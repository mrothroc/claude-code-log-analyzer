#!/usr/bin/env python3
"""
Arc Analyzer - Analyzes Claude Code logs for work patterns and autonomous hours.

Two complementary analyses:
1. ARCS - Coherent units of work from user prompts (what kind of work)
2. AGENTS - Autonomous agent sessions and their durations (how much autonomous work)

Uses Gemini Flash Lite for semantic boundary detection - no hardcoded patterns.

Usage:
    python arc_analyzer.py extract     # Extract arcs from parent session files
    python arc_analyzer.py agents      # Analyze agent sessions for autonomous hours
    python arc_analyzer.py stats       # Show combined statistics
    python arc_analyzer.py list        # List all arcs
    python arc_analyzer.py detail ID   # Show details for specific arc
    python arc_analyzer.py report      # Generate markdown report
"""

import json
import re
import sqlite3
import os
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse
from typing import Optional, Generator

from tqdm import tqdm

# Constants
CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"
DB_PATH = Path(__file__).parent / "arc_analytics.db"

# Additional Claude directories to scan (set via --extra-dirs flag or environment variable)
# Example: CLAUDE_EXTRA_DIRS=/Users/other/.claude/projects:/shared/.claude/projects
ADDITIONAL_PROJECT_DIRS = [
    Path(p) for p in os.environ.get("CLAUDE_EXTRA_DIRS", "").split(":") if p
]

# Model for boundary detection
BOUNDARY_MODEL = "gemini-flash-lite-latest"
BATCH_SIZE = 100  # Messages per API call
MAX_OUTPUT_TOKENS = 8000  # ~80 bytes per message result in JSON

# Arc duration thresholds (minutes)
ARC_DURATION_QUICK = 15      # < 15 min with agents = quick
ARC_DURATION_BUILD = 60      # 15-60 min = build
ARC_DURATION_FEATURE = 240   # 1-4 hours = feature, 4+ = release

# Confirmation patterns (short affirmations don't change topic)
CONFIRMATION_WORDS = {
    'yes', 'ok', 'okay', 'continue', 'proceed', 'go ahead', 'do it',
    'yes please', 'sounds good', 'perfect', 'great', 'thanks', 'thank you',
    'yep', 'yup', 'sure', 'agreed', 'correct', 'right', 'exactly'
}

# Schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS arcs (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    autonomy_level TEXT NOT NULL,  -- interactive/quick/build/feature/release (based on duration+agents)
    start_time TEXT NOT NULL,
    end_time TEXT,
    duration_minutes REAL,
    trigger_prompt TEXT,
    agents_spawned INTEGER DEFAULT 0,
    human_interrupts INTEGER DEFAULT 0,
    completion_status TEXT,
    intent TEXT,                   -- semantic intent from Gemini (implement/fix/refactor/test/review/deploy/docs/explore/config/other)
    session_file TEXT
);

CREATE TABLE IF NOT EXISTS agent_sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    start_time TEXT,
    end_time TEXT,
    duration_minutes REAL,
    message_count INTEGER,
    tool_calls INTEGER,
    file_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_arcs_project ON arcs(project);
CREATE INDEX IF NOT EXISTS idx_arcs_autonomy ON arcs(autonomy_level);
CREATE INDEX IF NOT EXISTS idx_arcs_intent ON arcs(intent);
CREATE INDEX IF NOT EXISTS idx_arcs_start ON arcs(start_time);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_project ON agent_sessions(project);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_duration ON agent_sessions(duration_minutes);
"""


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def generate_arc_id(project: str, timestamp: str, session_file: str = "", index: int = 0) -> str:
    """Generate a unique arc ID from project, timestamp, session file, and index."""
    hash_input = f"{project}:{session_file}:{timestamp}:{index}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


def is_confirmation(content: str) -> bool:
    """Check if message is just a short confirmation."""
    content = content.strip().lower().rstrip('.!?')
    return content in CONFIRMATION_WORDS


def parse_timestamp(ts) -> Optional[datetime]:
    """Parse timestamp string or numeric epoch to datetime."""
    if not ts:
        return None
    try:
        # Handle numeric timestamps (epoch milliseconds or seconds)
        if isinstance(ts, (int, float)):
            # Assume milliseconds if > year 3000 in seconds
            if ts > 32503680000:
                return datetime.fromtimestamp(ts / 1000)
            return datetime.fromtimestamp(ts)

        # Handle string numeric timestamps
        if isinstance(ts, str) and ts.replace('.', '').isdigit():
            numeric = float(ts)
            if numeric > 32503680000:
                return datetime.fromtimestamp(numeric / 1000)
            return datetime.fromtimestamp(numeric)

        # Handle ISO format
        if isinstance(ts, str) and 'T' in ts:
            if '.' in ts:
                base, frac = ts.split('.')
                frac_part = frac.rstrip('Z')[:6]
                ts = f"{base}.{frac_part}"
            ts = ts.rstrip('Z')
            return datetime.fromisoformat(ts)
        return None
    except (ValueError, OSError):
        return None


def stream_jsonl(file_path: Path) -> Generator[dict, None, None]:
    """Stream JSONL file line by line to handle large files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


def extract_user_messages(session_file: Path) -> list[dict]:
    """Extract all user messages from a session file."""
    messages = []

    for entry in stream_jsonl(session_file):
        entry_type = entry.get('type')
        timestamp = entry.get('timestamp')

        if entry_type == 'user':
            message = entry.get('message', {})
            content = ''

            # Extract text content
            if isinstance(message.get('content'), list):
                for block in message['content']:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        content += block.get('text', '')
                    elif isinstance(block, str):
                        content += block
            elif isinstance(message.get('content'), str):
                content = message['content']

            content = content.strip()
            if content:
                messages.append({
                    'timestamp': timestamp,
                    'content': content,
                    'is_topic_start': None,  # To be classified
                    'intent': None  # To be classified by Gemini for arc starts
                })

    return messages


def extract_assistant_data(session_file: Path) -> dict:
    """Extract agent spawns, completion signals, and activity timestamps from assistant messages."""
    data = {
        'agent_spawns': [],  # List of (timestamp, count)
        'completions': [],   # List of timestamps where completion detected
        'activity_timestamps': []  # All assistant message timestamps for duration calculation
    }

    completion_phrases = [
        'complete!', 'all tasks done', 'all tasks complete',
        'finished', 'release complete', 'work complete'
    ]

    for entry in stream_jsonl(session_file):
        if entry.get('type') != 'assistant':
            continue

        timestamp = entry.get('timestamp')
        if timestamp:
            data['activity_timestamps'].append(timestamp)

        message = entry.get('message', {})
        content_blocks = message.get('content', [])

        if not isinstance(content_blocks, list):
            continue

        spawn_count = 0
        for block in content_blocks:
            if isinstance(block, dict):
                # Count Task tool calls
                if block.get('type') == 'tool_use' and block.get('name') == 'Task':
                    spawn_count += 1

                # Check for completion
                if block.get('type') == 'text':
                    text = block.get('text', '').lower()
                    if any(phrase in text for phrase in completion_phrases):
                        data['completions'].append(timestamp)

        if spawn_count > 0:
            data['agent_spawns'].append((timestamp, spawn_count))

    return data


def classify_boundaries_with_gemini(messages: list[dict], verbose: bool = False) -> list[dict]:
    """Use Gemini Flash Lite to classify topic boundaries."""
    from google import genai
    from google.genai import types

    # Initialize client
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

    client = genai.Client(api_key=api_key)

    # Skip confirmations - they never start topics
    for msg in messages:
        if is_confirmation(msg['content']):
            msg['is_topic_start'] = False

    # First message is always a topic start
    if messages and messages[0]['is_topic_start'] is None:
        messages[0]['is_topic_start'] = True

    # Collect messages that need classification
    to_classify = [(i, msg) for i, msg in enumerate(messages) if msg['is_topic_start'] is None]

    if not to_classify and verbose:
        print("  All messages classified by heuristics")
        return messages

    num_batches = (len(to_classify) + BATCH_SIZE - 1) // BATCH_SIZE
    if verbose:
        print(f"  Classifying {len(to_classify)} messages in {num_batches} batches...")

    parsed_count = 0
    fallback_count = 0

    # Process in batches
    for batch_num, batch_start in enumerate(range(0, len(to_classify), BATCH_SIZE)):
        batch = to_classify[batch_start:batch_start + BATCH_SIZE]
        if verbose:
            print(f"    Batch {batch_num + 1}/{num_batches} ({len(batch)} messages)...", end=" ", flush=True)

        # Build prompt with context
        prompt_parts = [
            "You are analyzing a conversation to identify work 'arcs' - discrete quests or journeys.",
            "Each arc is a unit of work: the user starts a conversational thread with the LLM to accomplish a goal.",
            "",
            "For each message, determine: Is this STARTING a new quest, or CONTINUING an existing one?",
            "If it STARTS a new quest, also classify the INTENT (what kind of work is being requested).",
            "",
            "STARTS a new arc (new quest) when the user:",
            "- Gives a new command, request, or directive (even if related to previous work)",
            "- Asks to implement, fix, build, review, or investigate something",
            "- Delegates a task ('please do X', 'can you Y', 'I need Z')",
            "- Shares new information that requires action ('here is an error', 'this broke')",
            "- Changes direction ('actually, let's do X instead')",
            "- Returns after a break with a new request",
            "",
            "CONTINUES the current arc when the user:",
            "- Confirms or approves ('yes', 'ok', 'proceed', 'looks good')",
            "- Asks a clarifying question about work in progress",
            "- Provides requested information the AI asked for",
            "- Gives feedback on just-completed work ('that's not quite right')",
            "- Uses slash commands (/exit, /compact, /context)",
            "",
            "INTENT categories (only for 'start' messages):",
            "- implement: Building new features, adding functionality",
            "- fix: Bug fixes, error resolution, debugging",
            "- refactor: Code cleanup, restructuring, optimization",
            "- test: Writing or running tests, verification",
            "- review: Code review, design review, checking work",
            "- deploy: Deployment, release, shipping to production",
            "- docs: Documentation, comments, README updates",
            "- explore: Research, investigation, understanding code",
            "- config: Configuration, setup, environment changes",
            "- other: Anything that doesn't fit above categories",
            "",
            "KEY INSIGHT: Two similar tasks are TWO arcs. 'Fix bug A' then 'Fix bug B' = 2 arcs.",
            "Each new directive from the user starts a new quest, even if the topic is similar.",
            "",
            "Respond with ONLY a JSON array: [{\"index\": N, \"boundary\": \"start\" or \"continue\", \"intent\": \"category\"}]",
            "Include 'intent' only for 'start' boundaries.",
            "",
            "Messages to classify:"
        ]

        for idx, (orig_idx, msg) in enumerate(batch):
            # Include previous message for context
            prev_content = ""
            if orig_idx > 0:
                prev_content = messages[orig_idx - 1]['content'][:200]
                prompt_parts.append(f"\n[Previous context]: {prev_content}")

            content_preview = msg['content'][:500]
            prompt_parts.append(f"\n[Message {idx}]: {content_preview}")

        prompt = "\n".join(prompt_parts)

        try:
            response = client.models.generate_content(
                model=BOUNDARY_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for consistent classification
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )
            )

            # Parse response - extract JSON array using regex for robustness
            response_text = response.text.strip()

            # Try to find JSON array using regex (handles code fences, preamble, etc.)
            json_match = re.search(r'\[[\s\S]*?\]', response_text)
            if json_match:
                response_text = json_match.group(0)

            try:
                classifications = json.loads(response_text)
                starts = sum(1 for item in classifications if item.get('boundary', '').lower() == 'start')
                for item in classifications:
                    idx = item.get('index')
                    boundary = item.get('boundary', '').lower()
                    intent = item.get('intent', 'other').lower()
                    if idx is not None and idx < len(batch):
                        orig_idx = batch[idx][0]
                        messages[orig_idx]['is_topic_start'] = (boundary == 'start')
                        if boundary == 'start':
                            messages[orig_idx]['intent'] = intent
                parsed_count += len(batch)
                if verbose:
                    print(f"OK ({starts} starts)")
            except json.JSONDecodeError:
                # Fallback: treat unclear responses as continue
                fallback_count += len(batch)
                if verbose:
                    print(f"PARSE ERROR (fallback)")
                for orig_idx, _ in batch:
                    if messages[orig_idx]['is_topic_start'] is None:
                        messages[orig_idx]['is_topic_start'] = False

        except Exception as e:
            fallback_count += len(batch)
            if verbose:
                print(f"ERROR: {e}")
            # Fallback on error
            for orig_idx, _ in batch:
                if messages[orig_idx]['is_topic_start'] is None:
                    messages[orig_idx]['is_topic_start'] = False

    # Any remaining unclassified messages default to continue
    for msg in messages:
        if msg['is_topic_start'] is None:
            msg['is_topic_start'] = False

    if verbose:
        topic_starts = sum(1 for m in messages if m['is_topic_start'])
        print(f"  Done: {topic_starts} topic starts, {parsed_count} parsed, {fallback_count} fallback")

    return messages


def segment_into_arcs(messages: list[dict], assistant_data: dict,
                      session_file: Path, project_name: str) -> list[dict]:
    """Segment classified messages into arcs."""
    arcs = []
    current_arc = None
    arc_index = 0  # Track arc index within session for unique IDs
    completion_times = set(assistant_data['completions'])
    activity_timestamps = sorted(assistant_data['activity_timestamps'])

    for i, msg in enumerate(messages):
        timestamp = msg['timestamp']
        content = msg['content']

        if msg['is_topic_start']:
            # Close previous arc - record next arc's start as boundary
            if current_arc:
                current_arc['_next_arc_start'] = timestamp
                arcs.append(current_arc)

            # Start new arc
            current_arc = {
                'id': generate_arc_id(project_name, timestamp, str(session_file), arc_index),
                'project': project_name,
                'start_time': timestamp,
                'end_time': None,
                '_next_arc_start': None,  # Will be set when next arc starts
                'trigger_prompt': content[:500],
                'agents_spawned': 0,
                'human_prompts': 1,
                'completion_status': 'unknown',
                'session_file': str(session_file),
                'intent': msg.get('intent', 'other')  # Semantic intent from Gemini
            }
            arc_index += 1
        elif current_arc:
            # Continue current arc
            current_arc['human_prompts'] = current_arc.get('human_prompts', 0) + 1

    # Close final arc
    if current_arc:
        arcs.append(current_arc)

    # Compute proper end_time using last activity timestamp within each arc window
    for arc in arcs:
        start = parse_timestamp(arc['start_time'])
        next_start = parse_timestamp(arc['_next_arc_start']) if arc['_next_arc_start'] else None

        # Find last activity timestamp within this arc's window
        last_activity = None
        for ts in activity_timestamps:
            ts_dt = parse_timestamp(ts)
            if ts_dt and start:
                if ts_dt >= start and (next_start is None or ts_dt < next_start):
                    last_activity = ts

        # Use last activity timestamp as end_time (captures autonomous work)
        if last_activity:
            arc['end_time'] = last_activity
        elif arc['_next_arc_start']:
            # Fallback to next arc start if no activity found
            arc['end_time'] = arc['_next_arc_start']

        # Clean up temporary field
        del arc['_next_arc_start']

        # Count agent spawns within arc window
        end = parse_timestamp(arc['end_time']) if arc['end_time'] else None
        for spawn_time, count in assistant_data['agent_spawns']:
            spawn_dt = parse_timestamp(spawn_time)
            if spawn_dt and start:
                if spawn_dt >= start and (end is None or spawn_dt <= end):
                    arc['agents_spawned'] += count

        # Check for completion
        for comp_time in completion_times:
            comp_dt = parse_timestamp(comp_time)
            if comp_dt and start:
                if comp_dt >= start and (end is None or comp_dt <= end):
                    arc['completion_status'] = 'complete'

    return arcs


def classify_autonomy_level(duration_minutes: float | None, agents_spawned: int) -> str:
    """Classify arc autonomy level based on duration and agent count.

    This measures HOW the work was done (autonomy structure), not WHAT was done (intent).
    """
    # Classify by duration and agents
    if agents_spawned == 0:
        return 'interactive'
    elif duration_minutes is None:
        return 'quick'  # Unknown duration with agents - assume quick
    elif duration_minutes < ARC_DURATION_QUICK:
        return 'quick'
    elif duration_minutes < ARC_DURATION_BUILD:
        return 'build'
    elif duration_minutes < ARC_DURATION_FEATURE:
        return 'feature'
    else:
        return 'release'


def calculate_arc_metrics(arc: dict) -> dict:
    """Calculate duration and other metrics for an arc."""
    start = parse_timestamp(arc.get('start_time'))
    end = parse_timestamp(arc.get('end_time'))

    duration_minutes = None
    if start and end:
        duration = end - start
        duration_minutes = duration.total_seconds() / 60

    autonomy_level = classify_autonomy_level(
        duration_minutes,
        arc.get('agents_spawned', 0)
    )

    # Intent comes from Gemini classification, default to 'other'
    intent = arc.get('intent', 'other')

    # Calculate human interrupts (prompts beyond the first)
    human_interrupts = max(0, arc.get('human_prompts', 1) - 1)

    return {
        **arc,
        'duration_minutes': duration_minutes,
        'autonomy_level': autonomy_level,
        'intent': intent,
        'human_interrupts': human_interrupts
    }


def extract_all_arcs(verbose: bool = True, limit_files: int = None):
    """Extract arcs from all project session files."""
    if verbose:
        print(f"Initializing database at {DB_PATH}")
    conn = init_db()
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM arcs")
    conn.commit()

    total_arcs = 0

    # Collect all session files from all project directories
    session_files = []
    all_project_dirs = [PROJECTS_DIR] + [p for p in ADDITIONAL_PROJECT_DIRS if p.exists()]

    for projects_dir in all_project_dirs:
        if not projects_dir.exists():
            if verbose:
                print(f"Warning: {projects_dir} does not exist, skipping")
            continue
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            for session_file in project_dir.glob("*.jsonl"):
                if not session_file.name.startswith("agent-"):
                    session_files.append((session_file, project_dir.name))

    if limit_files:
        session_files = session_files[:limit_files]

    if verbose:
        print(f"Found {len(session_files)} session files")

    for session_file, project_name in tqdm(session_files, desc="Processing sessions", disable=not verbose):
        # Extract messages
        messages = extract_user_messages(session_file)
        if not messages:
            continue

        # Classify boundaries
        messages = classify_boundaries_with_gemini(messages, verbose=verbose)

        # Extract assistant data (agent spawns, completions)
        assistant_data = extract_assistant_data(session_file)

        # Segment into arcs
        arcs = segment_into_arcs(messages, assistant_data, session_file, project_name)

        # Calculate metrics and insert
        for arc in arcs:
            arc = calculate_arc_metrics(arc)

            cursor.execute("""
                INSERT OR REPLACE INTO arcs
                (id, project, autonomy_level, start_time, end_time, duration_minutes,
                 trigger_prompt, agents_spawned, human_interrupts,
                 completion_status, intent, session_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                arc['id'], arc['project'], arc['autonomy_level'],
                arc['start_time'], arc['end_time'], arc['duration_minutes'],
                arc['trigger_prompt'], arc['agents_spawned'],
                arc['human_interrupts'], arc['completion_status'],
                arc['intent'], arc['session_file']
            ))
            total_arcs += 1

        conn.commit()

    conn.close()

    if verbose:
        print(f"\nTotal arcs extracted: {total_arcs}")

    return total_arcs


def show_stats(db_path: Path = DB_PATH):
    """Show arc statistics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n=== Arc Analysis Summary ===\n")

    # Total arcs
    cursor.execute("SELECT COUNT(*) FROM arcs")
    total = cursor.fetchone()[0]
    print(f"Total arcs: {total}")

    if total == 0:
        print("\nNo arcs found. Run 'extract' first.")
        conn.close()
        return

    # Autonomy level distribution
    print("\n--- Autonomy Level Distribution ---")
    cursor.execute("""
        SELECT autonomy_level, COUNT(*) as count,
               ROUND(AVG(duration_minutes), 1) as avg_duration,
               ROUND(AVG(agents_spawned), 1) as avg_agents
        FROM arcs
        WHERE autonomy_level IS NOT NULL
        GROUP BY autonomy_level
        ORDER BY count DESC
    """)
    print(f"{'Level':<15} {'Count':>8} {'%':>8} {'Avg Duration':>15} {'Avg Agents':>12}")
    print("-" * 60)
    for row in cursor.fetchall():
        level, count, avg_dur, avg_agents = row
        pct = (count / total * 100) if total > 0 else 0
        dur_str = f"{avg_dur:.1f} min" if avg_dur else "N/A"
        agents_str = f"{avg_agents:.1f}" if avg_agents else "0"
        print(f"{level:<15} {count:>8} {pct:>7.1f}% {dur_str:>15} {agents_str:>12}")

    # Intent distribution (semantic, from Gemini)
    print("\n--- Intent Distribution (Semantic) ---")
    cursor.execute("""
        SELECT intent, COUNT(*) as count
        FROM arcs
        WHERE intent IS NOT NULL
        GROUP BY intent
        ORDER BY count DESC
    """)
    print(f"{'Intent':<15} {'Count':>8} {'%':>8}")
    print("-" * 35)
    for row in cursor.fetchall():
        intent, count = row
        pct = (count / total * 100) if total > 0 else 0
        print(f"{intent:<15} {count:>8} {pct:>7.1f}%")

    # Duration distribution
    print("\n--- Duration Distribution ---")
    cursor.execute("""
        SELECT
            CASE
                WHEN duration_minutes IS NULL THEN 'Unknown'
                WHEN duration_minutes < 5 THEN '< 5 min'
                WHEN duration_minutes < 15 THEN '5-15 min'
                WHEN duration_minutes < 60 THEN '15-60 min'
                WHEN duration_minutes < 240 THEN '1-4 hours'
                ELSE '4+ hours'
            END as bucket,
            COUNT(*) as count
        FROM arcs
        GROUP BY bucket
        ORDER BY
            CASE bucket
                WHEN '< 5 min' THEN 1
                WHEN '5-15 min' THEN 2
                WHEN '15-60 min' THEN 3
                WHEN '1-4 hours' THEN 4
                WHEN '4+ hours' THEN 5
                ELSE 6
            END
    """)
    print(f"{'Duration':<15} {'Count':>8} {'%':>8}")
    print("-" * 35)
    for row in cursor.fetchall():
        bucket, count = row
        pct = (count / total * 100) if total > 0 else 0
        print(f"{bucket:<15} {count:>8} {pct:>7.1f}%")

    # Autonomy summary
    print("\n--- Autonomy Summary ---")
    cursor.execute("""
        SELECT
            SUM(CASE WHEN agents_spawned > 0 THEN 1 ELSE 0 END) as autonomous_arcs,
            SUM(CASE WHEN agents_spawned = 0 THEN 1 ELSE 0 END) as interactive_arcs,
            SUM(CASE WHEN agents_spawned > 0 THEN duration_minutes ELSE 0 END) as autonomous_minutes,
            SUM(CASE WHEN agents_spawned = 0 THEN duration_minutes ELSE 0 END) as interactive_minutes
        FROM arcs
        WHERE duration_minutes IS NOT NULL
    """)
    row = cursor.fetchone()
    if row:
        auto_arcs, int_arcs, auto_min, int_min = row
        auto_min = auto_min or 0
        int_min = int_min or 0
        total_min = auto_min + int_min
        print(f"Autonomous arcs: {auto_arcs or 0} ({auto_min/60:.1f} hours)")
        print(f"Interactive arcs: {int_arcs or 0} ({int_min/60:.1f} hours)")
        if total_min > 0:
            print(f"Autonomy rate: {auto_min/total_min*100:.1f}% of time")

    # Longest arcs
    print("\n--- Longest Arcs (Top 10) ---")
    cursor.execute("""
        SELECT id, autonomy_level, intent, duration_minutes, agents_spawned,
               SUBSTR(trigger_prompt, 1, 50) as prompt_preview
        FROM arcs
        WHERE duration_minutes IS NOT NULL
        ORDER BY duration_minutes DESC
        LIMIT 10
    """)
    print(f"{'ID':<14} {'Level':<12} {'Intent':<10} {'Duration':>10} {'Agents':>7} {'Prompt':<25}")
    print("-" * 85)
    for row in cursor.fetchall():
        arc_id, level, intent, duration, agents, prompt = row
        dur_str = f"{duration:.0f} min" if duration < 120 else f"{duration/60:.1f} hrs"
        prompt_preview = (prompt[:22] + "...") if prompt and len(prompt) > 25 else (prompt or "")
        print(f"{arc_id:<14} {level:<12} {(intent or 'other'):<10} {dur_str:>10} {agents:>7} {prompt_preview:<25}")

    conn.close()


def list_arcs(db_path: Path = DB_PATH, autonomy_level: Optional[str] = None, limit: int = 50):
    """List arcs with optional autonomy level filter."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT id, autonomy_level, intent, start_time, duration_minutes, agents_spawned,
               SUBSTR(trigger_prompt, 1, 40) as prompt
        FROM arcs
    """
    params = []

    if autonomy_level:
        query += " WHERE autonomy_level = ?"
        params.append(autonomy_level)

    query += " ORDER BY start_time DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)

    print(f"\n{'ID':<14} {'Level':<12} {'Intent':<10} {'Start':<20} {'Duration':>8} {'Agents':>7} {'Prompt':<30}")
    print("-" * 110)

    for row in cursor.fetchall():
        arc_id, level, intent, start, duration, agents, prompt = row
        start_str = start[:19] if start else "Unknown"
        dur_str = f"{duration:.0f}m" if duration and duration < 120 else (f"{duration/60:.1f}h" if duration else "?")
        prompt_str = (prompt[:27] + "...") if prompt and len(prompt) > 30 else (prompt or "")
        print(f"{arc_id:<14} {level:<12} {(intent or 'other'):<10} {start_str:<20} {dur_str:>8} {agents:>7} {prompt_str:<30}")

    conn.close()


def show_detail(arc_id: str, db_path: Path = DB_PATH):
    """Show detailed information for a specific arc."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM arcs WHERE id = ?", (arc_id,))
    row = cursor.fetchone()

    if not row:
        print(f"Arc {arc_id} not found.")
        return

    columns = [desc[0] for desc in cursor.description]
    arc = dict(zip(columns, row))

    print(f"\n=== Arc Detail: {arc_id} ===\n")
    print(f"Autonomy Level: {arc['autonomy_level']}")
    print(f"Intent: {arc['intent']}")
    print(f"Project: {arc['project']}")
    print(f"Start: {arc['start_time']}")
    print(f"End: {arc['end_time']}")

    if arc['duration_minutes']:
        if arc['duration_minutes'] < 60:
            print(f"Duration: {arc['duration_minutes']:.1f} minutes")
        else:
            print(f"Duration: {arc['duration_minutes']/60:.2f} hours")

    print(f"Agents spawned: {arc['agents_spawned']}")
    print(f"Human interrupts: {arc['human_interrupts']}")
    print(f"Completion: {arc['completion_status']}")
    print(f"\nTrigger prompt:\n  {arc['trigger_prompt']}")
    print(f"\nSession file: {arc['session_file']}")

    conn.close()


def generate_report(db_path: Path = DB_PATH, output: Path = None):
    """Generate a markdown report of arc analysis."""
    if output is None:
        output = Path(__file__).parent / "ARC_REPORT.md"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM arcs")
    total_arcs = cursor.fetchone()[0]

    cursor.execute("""
        SELECT autonomy_level, COUNT(*) as count,
               ROUND(AVG(duration_minutes), 1) as avg_duration,
               ROUND(SUM(duration_minutes), 1) as total_duration,
               SUM(agents_spawned) as total_agents
        FROM arcs
        GROUP BY autonomy_level
        ORDER BY count DESC
    """)
    level_stats = cursor.fetchall()

    cursor.execute("""
        SELECT intent, COUNT(*) as count
        FROM arcs
        GROUP BY intent
        ORDER BY count DESC
    """)
    intent_stats = cursor.fetchall()

    cursor.execute("""
        SELECT id, autonomy_level, intent, duration_minutes, agents_spawned, trigger_prompt
        FROM arcs
        WHERE duration_minutes > 240
        ORDER BY duration_minutes DESC
        LIMIT 10
    """)
    long_arcs = cursor.fetchall()

    report = f"""# Arc Analysis Report

Generated: {datetime.now().isoformat()[:19]}

## Overview

**Total arcs analyzed:** {total_arcs}

## Autonomy Level Distribution

| Level | Count | % | Avg Duration | Total Duration | Total Agents |
|-------|-------|---|--------------|----------------|--------------|
"""

    for row in level_stats:
        level, count, avg_dur, total_dur, total_agents = row
        pct = (count / total_arcs * 100) if total_arcs > 0 else 0
        avg_str = f"{avg_dur:.1f} min" if avg_dur else "N/A"
        total_str = f"{total_dur/60:.1f} hrs" if total_dur else "N/A"
        report += f"| {level} | {count} | {pct:.1f}% | {avg_str} | {total_str} | {total_agents or 0} |\n"

    report += """
## Intent Distribution (Semantic)

| Intent | Count | % |
|--------|-------|---|
"""

    for row in intent_stats:
        intent, count = row
        pct = (count / total_arcs * 100) if total_arcs > 0 else 0
        report += f"| {intent} | {count} | {pct:.1f}% |\n"

    report += """
## Longest Arcs (4+ hours)

"""

    if long_arcs:
        for arc in long_arcs:
            arc_id, level, intent, duration, agents, prompt = arc
            dur_str = f"{duration/60:.1f} hours" if duration else "Unknown"
            report += f"""### {arc_id}
- **Autonomy Level:** {level}
- **Intent:** {intent}
- **Duration:** {dur_str}
- **Agents spawned:** {agents}
- **Trigger:** {(prompt or "")[:100]}{"..." if len(prompt or "") > 100 else ""}

"""
    else:
        report += "No arcs longer than 4 hours found.\n"

    report += """
## Key Insights

1. **Arc Type Distribution:** Shows how work breaks down between interactive and autonomous periods
2. **Intent Patterns:** What kinds of work you're doing (implementation, debugging, review, etc.)
3. **Long Arcs:** Your most extended autonomous work sessions

## About This Analysis

Arc boundaries were detected using semantic analysis (Gemini Flash Lite) rather than
hardcoded patterns. This means the analysis adapts to your personal workflow style.

Each arc represents a coherent unit of work - from when you started a task until you
shifted to something different.
"""

    conn.close()

    with open(output, 'w') as f:
        f.write(report)

    print(f"Report generated: {output}")


def extract_agent_session(agent_file: Path) -> Optional[dict]:
    """Extract metadata from an agent session file."""
    agent_id = agent_file.stem  # e.g., "agent-a015941"

    first_timestamp = None
    last_timestamp = None
    message_count = 0
    tool_calls = 0

    for entry in stream_jsonl(agent_file):
        timestamp = entry.get('timestamp')
        entry_type = entry.get('type')

        if timestamp:
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

        # Only count user/assistant messages, not tool results or other entries
        if entry_type in ('user', 'assistant'):
            message_count += 1

        # Count tool calls
        if entry_type == 'assistant':
            message = entry.get('message', {})
            content = message.get('content', [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        tool_calls += 1

    if not first_timestamp or not last_timestamp:
        return None

    # Calculate duration
    start = parse_timestamp(first_timestamp)
    end = parse_timestamp(last_timestamp)
    duration_minutes = None
    if start and end:
        duration_minutes = (end - start).total_seconds() / 60

    return {
        'id': agent_id,
        'start_time': first_timestamp,
        'end_time': last_timestamp,
        'duration_minutes': duration_minutes,
        'message_count': message_count,
        'tool_calls': tool_calls,
        'file_path': str(agent_file)
    }


def extract_all_agents(verbose: bool = True, projects: list[str] = None):
    """Extract all agent sessions from project directories."""
    if verbose:
        print(f"Initializing database at {DB_PATH}")
    conn = init_db()
    cursor = conn.cursor()

    # Clear existing agent data
    cursor.execute("DELETE FROM agent_sessions")
    conn.commit()

    # Collect all project directories to scan
    all_project_dirs = []
    if PROJECTS_DIR.exists():
        all_project_dirs.append(PROJECTS_DIR)
    elif verbose:
        print(f"Warning: {PROJECTS_DIR} does not exist")
    all_project_dirs.extend([p for p in ADDITIONAL_PROJECT_DIRS if p.exists()])

    if not all_project_dirs:
        if verbose:
            print("No Claude project directories found. Nothing to analyze.")
        conn.close()
        return

    # Collect all agent files
    agent_files = []
    for projects_dir in all_project_dirs:
        if verbose:
            print(f"Scanning {projects_dir}")
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_name = project_dir.name

            # Filter by project if specified
            if projects and not any(p in project_name for p in projects):
                continue

            for agent_file in project_dir.glob("agent-*.jsonl"):
                agent_files.append((agent_file, project_name))

    if verbose:
        print(f"Found {len(agent_files)} agent session files")

    total_sessions = 0
    total_hours = 0

    for agent_file, project_name in tqdm(agent_files, desc="Processing agents", disable=not verbose):
        session = extract_agent_session(agent_file)
        if not session:
            continue

        cursor.execute("""
            INSERT OR REPLACE INTO agent_sessions
            (id, project, start_time, end_time, duration_minutes, message_count, tool_calls, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session['id'],
            project_name,
            session['start_time'],
            session['end_time'],
            session['duration_minutes'],
            session['message_count'],
            session['tool_calls'],
            session['file_path']
        ))

        total_sessions += 1
        if session['duration_minutes']:
            total_hours += session['duration_minutes'] / 60

    conn.commit()
    conn.close()

    if verbose:
        print(f"\nTotal agent sessions: {total_sessions}")
        print(f"Total autonomous hours: {total_hours:.1f}")


def show_agent_stats():
    """Show statistics about agent sessions."""
    conn = init_db()
    cursor = conn.cursor()

    # Check if agent_sessions table has data
    cursor.execute("SELECT COUNT(*) FROM agent_sessions")
    count = cursor.fetchone()[0]

    if count == 0:
        print("No agent sessions found. Run 'agents' command first to extract agent data.")
        conn.close()
        return

    print("\n=== Agent Session Summary ===\n")

    # Total stats
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(duration_minutes) / 60.0 as total_hours,
            AVG(duration_minutes) as avg_minutes,
            MAX(duration_minutes) as max_minutes,
            SUM(message_count) as total_messages,
            SUM(tool_calls) as total_tools
        FROM agent_sessions
        WHERE duration_minutes IS NOT NULL
    """)
    row = cursor.fetchone()
    total, total_hours, avg_minutes, max_minutes, total_messages, total_tools = row

    print(f"Total agent sessions: {total:,}")
    print(f"Total autonomous hours: {total_hours:.1f}")
    print(f"Average session duration: {avg_minutes:.1f} minutes")
    print(f"Longest session: {max_minutes:.1f} minutes ({max_minutes/60:.1f} hours)")
    print(f"Total messages: {total_messages:,}")
    print(f"Total tool calls: {total_tools:,}")

    # Duration distribution
    print("\n--- Duration Distribution ---")
    print(f"{'Duration':<15} {'Count':>8} {'%':>8} {'Hours':>10}")
    print("-" * 45)

    duration_buckets = [
        ("< 1 min", 0, 1),
        ("1-2 min", 1, 2),
        ("2-5 min", 2, 5),
        ("5-10 min", 5, 10),
        ("10-20 min", 10, 20),
        ("20-40 min", 20, 40),
        ("40+ min", 40, 999999),
    ]

    for label, min_dur, max_dur in duration_buckets:
        cursor.execute("""
            SELECT COUNT(*), COALESCE(SUM(duration_minutes), 0) / 60.0
            FROM agent_sessions
            WHERE duration_minutes >= ? AND duration_minutes < ?
        """, (min_dur, max_dur))
        count, hours = cursor.fetchone()
        pct = (count / total * 100) if total > 0 else 0
        print(f"{label:<15} {count:>8} {pct:>7.1f}% {hours:>9.1f}h")

    # By project
    print("\n--- By Project ---")
    print(f"{'Project':<50} {'Sessions':>10} {'Hours':>10}")
    print("-" * 72)

    cursor.execute("""
        SELECT
            project,
            COUNT(*) as sessions,
            SUM(duration_minutes) / 60.0 as hours
        FROM agent_sessions
        WHERE duration_minutes IS NOT NULL
        GROUP BY project
        ORDER BY hours DESC
    """)

    for project, sessions, hours in cursor.fetchall():
        # Shorten project name for display (strip common path prefixes)
        short_name = project
        for prefix in ['-Users-', 'Users-']:
            if prefix in short_name:
                # Take everything after the last path-like segment
                parts = short_name.split('-')
                # Find where the actual project name starts (after IdeaProjects, Projects, etc.)
                for i, part in enumerate(parts):
                    if part in ('IdeaProjects', 'Projects', 'Shared'):
                        short_name = '-'.join(parts[i+1:])
                        break
        print(f"{short_name[:48]:<50} {sessions:>10} {hours:>9.1f}h")

    # Longest sessions
    print("\n--- Longest Agent Sessions (Top 10) ---")
    print(f"{'Agent ID':<20} {'Duration':>12} {'Messages':>10} {'Tools':>8}")
    print("-" * 54)

    cursor.execute("""
        SELECT id, duration_minutes, message_count, tool_calls
        FROM agent_sessions
        WHERE duration_minutes IS NOT NULL
        ORDER BY duration_minutes DESC
        LIMIT 10
    """)

    for agent_id, duration, messages, tools in cursor.fetchall():
        if duration >= 60:
            dur_str = f"{duration/60:.1f} hrs"
        else:
            dur_str = f"{duration:.0f} min"
        print(f"{agent_id:<20} {dur_str:>12} {messages:>10} {tools:>8}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arc Analyzer - Analyze work arcs and agent sessions in Claude Code logs")
    parser.add_argument("command", choices=["extract", "agents", "stats", "list", "detail", "report"],
                        help="Command to run")
    parser.add_argument("arc_id", nargs="?", help="Arc ID for detail command")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--data-dir", type=Path, help="Claude data directory (default: ~/.claude)")
    parser.add_argument("--level", help="Filter by autonomy level (for list command)")
    parser.add_argument("--limit", type=int, default=50, help="Limit results (for list command)")
    parser.add_argument("--files", type=int, help="Limit number of files to process (for extract command)")
    parser.add_argument("--projects", help="Comma-separated project name filters (for agents command)")

    args = parser.parse_args()

    # Override default data directory if specified
    if args.data_dir:
        CLAUDE_DIR = args.data_dir
        PROJECTS_DIR = CLAUDE_DIR / "projects"

    if args.command == "extract":
        extract_all_arcs(verbose=args.verbose, limit_files=args.files)
    elif args.command == "agents":
        projects = args.projects.split(',') if args.projects else None
        extract_all_agents(verbose=args.verbose, projects=projects)
    elif args.command == "stats":
        show_stats()
        show_agent_stats()
    elif args.command == "list":
        list_arcs(autonomy_level=args.level, limit=args.limit)
    elif args.command == "detail":
        if not args.arc_id:
            print("Error: arc_id required for detail command")
        else:
            show_detail(args.arc_id)
    elif args.command == "report":
        generate_report()

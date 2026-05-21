#!/usr/bin/env python3
"""
Gate Analyzer - Analyzes Claude Code review gate usage patterns.

Analyzes how review gates (code review, design review, precommit, etc.) are used
in Claude Code workflows to understand quality assurance patterns.

Usage:
    python gate_analyzer.py discover           # Discover available review gate tools
    python gate_analyzer.py extract            # Extract gate check usage from session logs
    python gate_analyzer.py classify           # Classify gate decisions and feedback
    python gate_analyzer.py classify-errors    # Classify error types in gate feedback
    python gate_analyzer.py compute-overlap    # Compute overlap ratio (ω) across gates
    python gate_analyzer.py stats              # Show gate usage statistics
    python gate_analyzer.py error-analysis     # Analyze error patterns by gate type
    python gate_analyzer.py report             # Generate markdown report
"""

# Standard library imports
import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Third-party imports
from tqdm import tqdm

# Constants
CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"
DB_PATH = Path(__file__).parent / "gate_analytics.db"

# Model for classification
DEFAULT_MODEL = "gemini-flash-lite-latest"


def _get_api_key() -> Optional[str]:
    """Get Gemini API key from environment, checking both common variable names."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _sanitize_path(p) -> str:
    """Replace home directory with ~ for display."""
    return str(p).replace(str(Path.home()), '~')


def _sanitize_project(project: str) -> str:
    """Sanitize project identifiers for display.

    Claude Code encodes local paths as project IDs (e.g. '-Users-alice-code-myapp').
    Extract just the final component to avoid leaking filesystem structure.
    """
    if not project:
        return ""
    # Split on path-separator-style dashes and take the last non-empty segment
    parts = [p for p in project.split("-") if p]
    if len(parts) <= 1:
        return project
    # Heuristic: project names are the last 1-2 segments after common prefixes
    # like Users/username/IdeaProjects/... or home/username/code/...
    # Return the last segment as the project name
    return parts[-1]

# Gate type inference patterns (pattern, gate_type) — used for manual --gates only
GATE_TYPE_PATTERNS = [
    (re.compile(r'.*review_plan.*', re.IGNORECASE), 'review_plan'),
    (re.compile(r'.*review_design.*', re.IGNORECASE), 'review_design'),
    (re.compile(r'.*review_code.*', re.IGNORECASE), 'review_code'),
    (re.compile(r'.*codereview.*', re.IGNORECASE), 'codereview'),
    (re.compile(r'.*precommit.*', re.IGNORECASE), 'precommit'),
    (re.compile(r'.*validate.*', re.IGNORECASE), 'validation'),
    (re.compile(r'.*qa.*', re.IGNORECASE), 'qa'),
    (re.compile(r'.*audit.*', re.IGNORECASE), 'audit'),
]

# Schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS gate_tools (
    tool_name TEXT PRIMARY KEY,
    gate_type TEXT NOT NULL,
    discovery_method TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gate_checks (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    project TEXT NOT NULL,
    gate_type TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    decision TEXT,
    feedback_text TEXT,
    feedback_length INTEGER,
    error_class TEXT,
    timestamp TEXT,
    session_file TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gate_issues (
    id INTEGER PRIMARY KEY,
    gate_check_id INTEGER,
    severity TEXT,
    title TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS gate_iterations (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    gate_type TEXT,
    iteration_number INTEGER,
    decision TEXT,
    gate_check_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_gate_checks_project ON gate_checks(project);
CREATE INDEX IF NOT EXISTS idx_gate_checks_gate_type ON gate_checks(gate_type);
CREATE INDEX IF NOT EXISTS idx_gate_checks_decision ON gate_checks(decision);
CREATE INDEX IF NOT EXISTS idx_gate_checks_error_class ON gate_checks(error_class);
CREATE INDEX IF NOT EXISTS idx_gate_issues_check ON gate_issues(gate_check_id);
CREATE INDEX IF NOT EXISTS idx_gate_iterations_session ON gate_iterations(session_id, gate_type);
"""


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def infer_gate_type(tool_name: str) -> str:
    """Infer gate type from tool name using pattern matching.

    Args:
        tool_name: Name of the tool to classify

    Returns:
        Inferred gate_type string. Defaults to 'other' if no pattern matches.
    """
    for pattern, gate_type in GATE_TYPE_PATTERNS:
        if pattern.match(tool_name):
            return gate_type

    # Default fallback
    return 'other'


def collect_jsonl_files() -> list[tuple[Path, str]]:
    """Collect all JSONL files across all projects, with project names.

    Returns list of (file_path, project_name) tuples.
    Skips .trimmed.jsonl if the original exists.
    """
    if not PROJECTS_DIR.exists():
        return []

    files = []
    seen_stems = set()

    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name

        # Collect top-level JSONL files
        for jsonl_file in sorted(project_dir.glob("*.jsonl")):
            stem = jsonl_file.stem
            if stem.endswith(".trimmed"):
                original_stem = stem.replace(".trimmed", "")
                if (project_dir, original_stem) in seen_stems:
                    continue
                stem = original_stem

            key = (project_dir, stem)
            if key not in seen_stems:
                seen_stems.add(key)
                files.append((jsonl_file, project_name))

        # Collect subagent JSONL files from session subdirectories
        for session_dir in sorted(project_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            subagent_dir = session_dir / "subagents"
            if subagent_dir.is_dir():
                for jsonl_file in sorted(subagent_dir.glob("*.jsonl")):
                    stem = jsonl_file.stem
                    if stem.endswith(".trimmed"):
                        original_stem = stem.replace(".trimmed", "")
                        if (subagent_dir, original_stem) in seen_stems:
                            continue
                        stem = original_stem

                    key = (subagent_dir, stem)
                    if key not in seen_stems:
                        seen_stems.add(key)
                        files.append((jsonl_file, project_name))

    return files


def stream_jsonl_with_line_numbers(file_path: Path):
    """Stream JSONL file line by line, yielding (line_number, parsed_dict).

    Handles malformed JSON gracefully by skipping bad lines.
    Uses streaming reads to handle files up to 2+ GB.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield (line_number, json.loads(line))
                except json.JSONDecodeError:
                    pass  # Silently skip malformed lines (common in large log files)
    except (IOError, OSError) as e:
        print(f"  Error reading {file_path.name}: {e}")


def collect_all_tool_names(verbose: bool = False) -> set[str]:
    """Scan all JSONL files and collect unique tool_use block names.

    Args:
        verbose: If True, show progress bar with tqdm

    Returns:
        Set of unique tool name strings
    """
    jsonl_files = collect_jsonl_files()

    if not jsonl_files:
        if verbose:
            print(f"No JSONL files found in {_sanitize_path(PROJECTS_DIR)}")
        return set()

    tool_names = set()

    iterator = tqdm(jsonl_files, desc="Scanning JSONL files", unit="file") if verbose else jsonl_files

    for file_path, project_name in iterator:
        for line_number, entry in stream_jsonl_with_line_numbers(file_path):
            message = entry.get("message", {})
            if not isinstance(message, dict):
                continue

            content_blocks = message.get("content", [])
            if not isinstance(content_blocks, list):
                continue

            for block in content_blocks:
                if not isinstance(block, dict):
                    continue

                if block.get("type") == "tool_use":
                    tool_name = block.get("name")
                    if tool_name:
                        tool_names.add(tool_name)

    return tool_names


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_go_duration(duration_str: str) -> Optional[float]:
    """Parse a Go-style duration string to seconds.

    Examples:
        "1m24.417249208s" -> 84.417249208
        "45.2s" -> 45.2
        "2m3.5s" -> 123.5
        "3h2m1s" -> 10921.0
        "0s" -> 0.0
        "xyz" -> None

    Args:
        duration_str: Go-style duration string (e.g., "1m24.5s")

    Returns:
        Total seconds as float, or None if no valid duration units found
    """
    if not duration_str or not isinstance(duration_str, str):
        return None

    total_seconds = 0.0
    matched_any = False

    # Tokenize with unit-safe regex: match number+unit pairs.
    # Order matters: ms/us/ns must match before bare m/s to avoid
    # "75ms" being parsed as 75 minutes.
    for value, unit in re.findall(r'(\d+(?:\.\d+)?)(ns|us|µs|ms|h|m|s)', duration_str):
        matched_any = True
        v = float(value)
        if unit == 'h':
            total_seconds += v * 3600
        elif unit == 'm':
            total_seconds += v * 60
        elif unit == 's':
            total_seconds += v
        elif unit == 'ms':
            total_seconds += v / 1000
        elif unit in ('us', 'µs'):
            total_seconds += v / 1_000_000
        elif unit == 'ns':
            total_seconds += v / 1_000_000_000

    return total_seconds if matched_any else None


def _canonicalize_decision(raw: str) -> Optional[str]:
    """Normalize a raw decision string to a canonical enum value.

    Returns APPROVED, NEEDS_REVISION, ESCALATE, or None if unrecognizable.
    """
    upper = raw.strip().upper()

    if upper in ("APPROVED", "PASS", "PASSED", "LGTM", "SUCCESS", "ACCEPT", "ACCEPTED"):
        return "APPROVED"
    if upper in ("NEEDS_REVISION", "REJECTED", "REJECT", "FAILED", "FAIL",
                 "NEEDS REVISION", "DENIED", "NOT_APPROVED"):
        return "NEEDS_REVISION"
    if upper in ("ESCALATE", "ESCALATED", "ESCALATION"):
        return "ESCALATE"

    return None  # Unrecognizable — let classify pipeline handle it


def _is_workflow_status_message(feedback_text: str) -> bool:
    """True if feedback_text is a pause/status/control message, not a substantive verdict.

    Conjunctive heuristics:
    - Parses as a JSON object (dict).
    - Has a 'status' string matching /(_ready_for_|complete|paused|pending|continued)/i.
    - Has 'next_step_required' or 'step_number'.
    - Lacks non-empty 'issues_found', 'findings', or 'issues'.
    - Lacks explicit 'decision', 'verdict', or '<disposition>' tag indicating APPROVED or NEEDS_REVISION.
    """
    if not feedback_text:
        return False

    try:
        data = json.loads(feedback_text)
    except (json.JSONDecodeError, TypeError):
        return False

    if not isinstance(data, dict):
        return False

    # Check status field
    status_val = data.get("status")
    if not isinstance(status_val, str):
        return False
    if not re.search(r'(_ready_for_|complete|paused|pending|continued)', status_val, re.IGNORECASE):
        return False

    # Check step keys
    if "next_step_required" not in data and "step_number" not in data:
        return False

    # Lacks non-empty issues_found or findings
    for key in ("issues_found", "findings", "issues"):
        val = data.get(key)
        if val:
            if isinstance(val, (list, dict, str)) and len(val) > 0:
                return False
            if isinstance(val, int) and val > 0:
                return False
            if isinstance(val, bool) and val is True:
                return False

    # Lacks explicit verdict/decision/disposition
    for key in ("decision", "verdict", "disposition"):
        val = data.get(key)
        if val and isinstance(val, str):
            canonical = _canonicalize_decision(val)
            if canonical in ("APPROVED", "NEEDS_REVISION"):
                return False

    if re.search(r'(?i)<disposition>\s*(APPROVED|NEEDS_REVISION)\s*</disposition>', feedback_text):
        return False

    return True


def _extract_two_phase_body_decision(feedback_text: str) -> Optional[str]:
    """Extract the actual body verdict from two-phase review-design feedback.

    The two-phase review-design wrapper can contain a stale top-level
    ``**Decision**: NEEDS_REVISION`` while the Phase 2 body under
    ``## AI Analysis`` explicitly approves the design. In that format, the body
    verdict is the substantive reviewer decision.
    """
    if not feedback_text:
        return None

    # Keep this intentionally narrow so generic "approved for implementation"
    # wording in ordinary rejection text does not override a real rejection.
    if not re.search(r'(?im)^\s*#\s+Design Review Analysis\s*$', feedback_text):
        return None

    ai_header = re.search(r'(?im)^\s*##\s+AI Analysis\s*$', feedback_text)
    if not ai_header:
        return None

    body = feedback_text[ai_header.end():]

    # If the body has an explicit DECISION line, the first one is authoritative.
    decision_line = re.search(
        r'(?im)^\s*(?:\*{0,2}\s*)?DECISION(?:\s*\*{0,2})?\s*:?'
        r'(?:\s*\*{0,2})?\s*'
        r'(APPROVED|PASS(?:ED)?|NEEDS[_ ]REVISION|REJECT(?:ED)?|FAIL(?:ED)?|ESCALATE)\b',
        body,
    )
    if decision_line:
        return _canonicalize_decision(decision_line.group(1))

    # Some older review-design bodies have no DECISION line but still contain a
    # clear approval conclusion. Only accept strong early-body approval language
    # when no blocking/revision wording appears near the conclusion.
    first_body = body[:2200]
    first_body_lower = first_body.lower()
    blocking_markers = (
        "decision: needs_revision",
        "needs revision",
        "required revisions",
        "requires revision",
        "must be fixed",
        "blocking issue",
        "critical issue",
    )
    if any(marker in first_body_lower for marker in blocking_markers):
        return None

    approval_patterns = (
        r'\bapproved\s+(?:to\s+move\s+to\s+the\s+next\s+stage|for\s+implementation|without\s+(?:reservation|revision))\b',
        r'\b(?:proposal|design|plan)\s+is\s+approved\b',
        r'\bit\s+is\s+approved\b',
        r'\bthis\s+proposal\s+can\s+be\s+implemented\s+with\s+high\s+confidence\b',
        r'\bthis\s+is\s+an\s+exemplary\s+design\s+document\b.{0,450}\b'
        r'(?:thoroughly\s+addresses\s+all\s+criteria|meets\s+all\s+requirements|'
        r'meticulously\s+addresses\s+all\s+official\s+acceptance\s+criteria|'
        r'is\s+thorough|demonstrates\s+a\s+clear\s+understanding)\b',
    )
    if any(re.search(pattern, first_body, re.IGNORECASE | re.DOTALL) for pattern in approval_patterns):
        return "APPROVED"

    return None


def parse_review_result(content_text: str) -> dict:
    """Parse a tool_result content text into structured gate check data.

    Generic parser that handles multiple response formats from different
    review gate tools. Tries patterns in order of specificity.

    Args:
        content_text: Raw content from tool_result block

    Returns:
        dict with keys:
            - decision: str or None (APPROVED, NEEDS_REVISION, ESCALATE)
            - execution_time_seconds: float or None
            - feedback_text: str or None
    """
    result = {
        "decision": None,
        "execution_time_seconds": None,
        "feedback_text": None,
    }

    if not content_text:
        return result

    if _is_workflow_status_message(content_text):
        result["decision"] = "STATUS_ONLY"
        result["feedback_text"] = content_text
        try:
            data = json.loads(content_text)
            exec_time = data.get("execution_time")
            if exec_time:
                result["execution_time_seconds"] = parse_go_duration(exec_time)
        except Exception:
            pass
        return result

    # Try to parse as JSON first
    try:
        data = json.loads(content_text)
    except (json.JSONDecodeError, TypeError):
        # Not JSON - treat whole content as feedback text
        result["feedback_text"] = content_text

        body_decision = _extract_two_phase_body_decision(content_text)
        if body_decision:
            result["decision"] = body_decision
            return result

        # Check for disposition tags in plain text
        disp_match = re.search(
            r'<disposition>(APPROVED|NEEDS_REVISION|ESCALATE)</disposition>',
            content_text,
            re.IGNORECASE
        )
        if disp_match:
            result["decision"] = disp_match.group(1).upper()

        return result

    if not isinstance(data, dict):
        result["feedback_text"] = content_text
        return result

    # Extract execution time if present
    exec_time = data.get("execution_time")
    if exec_time:
        result["execution_time_seconds"] = parse_go_duration(exec_time)

    # Extract feedback text - try multiple field names
    feedback = data.get("feedback") or data.get("review") or data.get("summary")
    if feedback:
        # Normalize to string — some tools return structured objects
        result["feedback_text"] = feedback if isinstance(feedback, str) else json.dumps(feedback, indent=2)

    body_decision = _extract_two_phase_body_decision(result["feedback_text"])

    # Decision extraction - try patterns in order of specificity

    # Pattern 1: Direct decision field — canonicalize to enum values
    raw_decision = data.get("decision")
    if raw_decision and isinstance(raw_decision, str):
        canonical = _canonicalize_decision(raw_decision)
        if canonical:
            if canonical == "NEEDS_REVISION" and body_decision == "APPROVED":
                result["decision"] = body_decision
                return result
            result["decision"] = canonical
            return result
        # Unrecognized value — fall through to other patterns

    if body_decision:
        result["decision"] = body_decision
        return result

    # Pattern 2: code_review_status (from codereview tools)
    code_review_status = data.get("code_review_status")
    if code_review_status and isinstance(code_review_status, dict):
        issues_found = code_review_status.get("issues_found", 0)
        confidence = code_review_status.get("current_confidence", "unknown")

        # Derive decision from codereview status
        if issues_found == 0:
            result["decision"] = "APPROVED"
        # issues_found > 0 is ambiguous regardless of confidence — leave as None
        # for the classify pipeline (regex + Gemini) to resolve

        # Build feedback from full response if not already set
        if not result["feedback_text"]:
            result["feedback_text"] = json.dumps(data, indent=2)

        return result

    # Pattern 3: workflow_guidance (from review_code tools)
    workflow_guidance = data.get("workflow_guidance")
    if workflow_guidance and isinstance(workflow_guidance, dict):
        guidance = workflow_guidance.get("guidance", "")
        if "PASSED" in guidance.upper():
            result["decision"] = "APPROVED"
        elif "FAILED" in guidance.upper() or "NEEDS_REVISION" in guidance.upper():
            result["decision"] = "NEEDS_REVISION"
        # No clear pass/fail signal — leave as None for classify pipeline

        if result["decision"]:
            return result

    # Pattern 4: Generic status field
    status = data.get("status")
    if status and isinstance(status, str):
        status_upper = status.upper()
        if "APPROVED" in status_upper or "PASS" in status_upper or "SUCCESS" in status_upper:
            result["decision"] = "APPROVED"
        elif "REJECT" in status_upper or "FAIL" in status_upper or "REVISION" in status_upper:
            result["decision"] = "NEEDS_REVISION"
        elif "ESCALATE" in status_upper:
            result["decision"] = "ESCALATE"

        if result["decision"]:
            return result

    # Pattern 5: Check for disposition tags in feedback text
    if result["feedback_text"]:
        disp_match = re.search(
            r'<disposition>(APPROVED|NEEDS_REVISION|ESCALATE)</disposition>',
            result["feedback_text"],
            re.IGNORECASE
        )
        if disp_match:
            result["decision"] = disp_match.group(1).upper()
            return result

    # If no decision found, return what we have (decision=None, feedback_text set)
    return result


def extract_issues_from_feedback(feedback: str) -> list[dict]:
    """Best-effort extraction of issues from review feedback text.

    Tries three pattern groups in order, returning after first success:
    1. Markdown table rows with severity columns
    2. Bold severity blocks (**Severity**: Level)
    3. issues_by_severity JSON (from codereview responses)

    Args:
        feedback: Raw feedback text from review gate

    Returns:
        List of dicts with keys: severity, title, description
        Empty list if no patterns match or feedback is None/empty
    """
    if not feedback:
        return []

    issues = []

    # Pattern Group 1: Markdown table rows (Findings Table format)
    # | File | Line | Category | Severity | Description | ...
    table_pattern = re.compile(
        r'\|\s*[^|]+\|\s*\d*\s*\|\s*(\w+)\s*\|\s*(CRITICAL|HIGH|MEDIUM|LOW|WARNING|INFO)\s*\|\s*([^|]+)',
        re.IGNORECASE
    )
    for match in table_pattern.finditer(feedback):
        category = match.group(1).strip()
        severity = match.group(2).strip().lower()
        description = match.group(3).strip()
        issues.append({
            "severity": severity,
            "title": category,
            "description": description[:500]  # Truncate long descriptions
        })

    if issues:
        return issues  # Early return after first successful pattern group

    # Pattern Group 2: "**Severity**: High/Medium/Low" blocks
    severity_block_pattern = re.compile(
        r'\*\*Severity\*\*:\s*(Critical|High|Medium|Low|Warning|Info)',
        re.IGNORECASE
    )
    issue_title_pattern = re.compile(
        r'(?:^|\n)#+\s*\d*\.?\s*(.+?)(?:\n|$)',
    )

    # Extract positions and values for both patterns
    titles = [(m.start(), m.group(1).strip()) for m in issue_title_pattern.finditer(feedback)]
    severities = [(m.start(), m.group(1).strip().lower()) for m in severity_block_pattern.finditer(feedback)]

    # Match each severity to nearest preceding title
    for sev_pos, severity in severities:
        best_title = None
        best_dist = float('inf')
        for title_pos, title in titles:
            if title_pos < sev_pos and (sev_pos - title_pos) < best_dist:
                best_dist = sev_pos - title_pos
                best_title = title
        issues.append({
            "severity": severity,
            "title": best_title or "Unnamed issue",
            "description": ""
        })

    if issues:
        return issues  # Early return after second pattern group

    # Pattern Group 3: issues_by_severity from codereview JSON
    severity_counts_pattern = re.compile(
        r'"issues_by_severity"\s*:\s*\{([^}]+)\}'
    )
    sev_match = severity_counts_pattern.search(feedback)
    if sev_match:
        sev_text = sev_match.group(1)
        for sev_item in re.finditer(r'"(\w+)"\s*:\s*(\d+)', sev_text):
            severity = sev_item.group(1).lower()
            count = int(sev_item.group(2))
            for _ in range(count):
                issues.append({
                    "severity": severity,
                    "title": f"{severity} issue",
                    "description": "From codereview issues_by_severity"
                })

    return issues


def get_gate_tools_map(conn: sqlite3.Connection) -> Optional[dict[str, str]]:
    """Read gate_tools table and return mapping of tool_name -> gate_type.

    Args:
        conn: Database connection

    Returns:
        Dict mapping tool_name to gate_type, or None if table is empty
    """
    cursor = conn.cursor()
    cursor.execute("SELECT tool_name, gate_type FROM gate_tools")
    rows = cursor.fetchall()

    if not rows:
        return None

    return {tool_name: gate_type for tool_name, gate_type in rows}


def insert_manual_gates(conn: sqlite3.Connection, gate_list: str):
    """Insert manually specified gates into gate_tools table.

    Args:
        conn: Database connection
        gate_list: Comma-separated list of tool names
    """
    cursor = conn.cursor()
    tool_names = [name.strip() for name in gate_list.split(',')]

    for tool_name in tool_names:
        # Infer gate type from tool name
        gate_type = infer_gate_type(tool_name)

        cursor.execute("""
            INSERT OR REPLACE INTO gate_tools (tool_name, gate_type, discovery_method)
            VALUES (?, ?, 'manual')
        """, (tool_name, gate_type))

    conn.commit()
    print(f"Inserted {len(tool_names)} manual gate tools")


def extract_gates_from_file(file_path: Path, project_name: str, gate_map: dict[str, str]) -> list[dict]:
    """Extract review gate calls and results from a single JSONL file.

    Scans JSONL file for tool_use/tool_result pairs matching discovered gates.
    Single-pass strategy:
    1. First pass: collect tool_use blocks where tool name is in gate_map, indexed by tool_use_id
    2. Second pass: match tool_result blocks to their tool_use via tool_use_id
    3. Parse each result with parse_review_result()
    4. Extract issues with extract_issues_from_feedback()
    5. Handle unmatched tool_use calls (no result) - still record with decision=None

    Args:
        file_path: Path to JSONL file
        project_name: str project name
        gate_map: dict[str, str] from get_gate_tools_map()

    Returns:
        List of dicts with keys:
            - session_id: Session ID
            - project: Project name
            - gate_type: Gate type (from gate_map)
            - tool_name: Tool name
            - decision: APPROVED, NEEDS_REVISION, ESCALATE, or None
            - execution_time_seconds: float or None
            - feedback_text: str or None
            - feedback_length: int
            - timestamp: ISO timestamp
            - session_file: str path to JSONL file
            - issues: list of issue dicts from extract_issues_from_feedback()
    """
    pending_calls = {}
    gate_checks = []

    for line_number, entry in stream_jsonl_with_line_numbers(file_path):
        # entry.get("type") not needed for gate extraction
        message = entry.get("message", {})
        timestamp = entry.get("timestamp")
        session_id = entry.get("sessionId") or file_path.stem

        if not isinstance(message, dict):
            continue

        content_blocks = message.get("content", [])
        if not isinstance(content_blocks, list):
            continue

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            # Check for tool_use (gate call)
            if block.get("type") == "tool_use":
                tool_name = block.get("name", "")
                if tool_name in gate_map:
                    tool_use_id = block.get("id")
                    if not tool_use_id:
                        # Synthetic key to avoid None collisions
                        tool_use_id = f"_synthetic_{line_number}:{tool_name}"

                    pending_calls[tool_use_id] = {
                        "tool_use_id": tool_use_id,
                        "tool_name": tool_name,
                        "gate_type": gate_map[tool_name],
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "line_number": line_number,
                    }

            # Check for tool_result (gate result)
            elif block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if tool_use_id and tool_use_id in pending_calls:
                    call_info = pending_calls.pop(tool_use_id)

                    # Extract content text from result (handles both string and list formats)
                    result_content = block.get("content", "")
                    content_text = ""

                    if isinstance(result_content, str):
                        content_text = result_content
                    elif isinstance(result_content, list):
                        text_parts = []
                        for content_block in result_content:
                            if isinstance(content_block, dict) and content_block.get("type") == "text":
                                text_parts.append(content_block.get("text", ""))
                            elif isinstance(content_block, str):
                                text_parts.append(content_block)
                        content_text = "\n".join(text_parts)

                    parsed = parse_review_result(content_text)

                    # Extract issues from feedback
                    issues = extract_issues_from_feedback(parsed["feedback_text"]) if parsed["feedback_text"] else []

                    gate_check = {
                        "session_id": call_info["session_id"],
                        "project": project_name,
                        "gate_type": call_info["gate_type"],
                        "tool_name": call_info["tool_name"],
                        "decision": parsed["decision"],
                        "execution_time_seconds": parsed["execution_time_seconds"],
                        "feedback_text": parsed["feedback_text"],
                        "feedback_length": len(parsed["feedback_text"]) if parsed["feedback_text"] else 0,
                        "timestamp": call_info["timestamp"] or timestamp,
                        "session_file": str(file_path),
                        "issues": issues,
                    }
                    gate_checks.append(gate_check)

    # Record remaining unmatched calls (no result found)
    for tool_use_id, call_info in pending_calls.items():
        gate_check = {
            "session_id": call_info["session_id"],
            "project": project_name,
            "gate_type": call_info["gate_type"],
            "tool_name": call_info["tool_name"],
            "decision": None,
            "execution_time_seconds": None,
            "feedback_text": None,
            "feedback_length": 0,
            "timestamp": call_info["timestamp"],
            "session_file": str(file_path),
            "issues": [],
        }
        gate_checks.append(gate_check)

    return gate_checks


def _insert_iterations_from_groups(cursor, groups: dict, verbose: bool = False):
    """Insert iteration rows from grouped gate checks, splitting on APPROVED boundaries.

    The private tool groups by (task_id, gate_type) — each task has its own review cycle.
    The public tool doesn't have task_id, so we split each group into sub-sequences
    bounded by APPROVED decisions. An APPROVED means that piece of work passed; the
    next check starts a new review cycle.

    This prevents false recoveries where APPROVED → NR → APPROVED spans different tasks.

    Args:
        cursor: Database cursor
        groups: Dict mapping (context_id, gate_type) -> list of (check_id, decision, timestamp, session_id)
        verbose: Show progress information
    """
    total_sequences = 0

    for (context_id, gate_type), checks in groups.items():
        # Split into sub-sequences: each sequence ends after an APPROVED.
        # Encode sub-sequence number into session_id so LEAD partitions correctly.
        seq_num = 0
        iteration_in_seq = 0

        for check_id, decision, timestamp, session_id in checks:
            iteration_in_seq += 1
            seq_session_id = f"{session_id}::{seq_num}"

            cursor.execute("""
                INSERT INTO gate_iterations (session_id, gate_type, iteration_number, decision, gate_check_id)
                VALUES (?, ?, ?, ?, ?)
            """, (seq_session_id, gate_type, iteration_in_seq, decision, check_id))

            # APPROVED ends a review cycle — reset for next piece of work
            if decision == 'APPROVED':
                seq_num += 1
                iteration_in_seq = 0
                total_sequences += 1

        # Count incomplete sequences (ended without APPROVED)
        if iteration_in_seq > 0:
            total_sequences += 1

    if verbose:
        print(f"  Created {total_sequences} review cycle sub-sequences")


def compute_iterations(conn: sqlite3.Connection, verbose: bool = False):
    """Compute iteration sequences for gate checks, splitting on APPROVED boundaries.

    Groups gate checks by context (arc or session) and gate type, then splits each
    group into sub-sequences bounded by APPROVED decisions. This simulates per-task
    grouping without requiring task_id.

    Uses arc boundaries from arc_analytics.db when available for initial grouping,
    falls back to session_id.

    Args:
        conn: Database connection
        verbose: Show progress information
    """
    cursor = conn.cursor()

    # Clear existing iterations
    cursor.execute("DELETE FROM gate_iterations")

    # Try arc-based grouping first
    arc_db_path = DB_PATH.parent / "arc_analytics.db"

    if arc_db_path.exists():
        try:
            cursor.execute(f"ATTACH DATABASE '{str(arc_db_path).replace(chr(39), chr(39)*2)}' AS arc_db")

            cursor.execute("""
                SELECT DISTINCT gc.id, a.id as arc_id, gc.gate_type, gc.decision, gc.timestamp, gc.session_id
                FROM gate_checks gc
                JOIN arc_db.arcs a
                    ON gc.session_file = a.session_file
                    AND gc.timestamp >= a.start_time
                    AND gc.timestamp <= a.end_time
                WHERE gc.session_id IS NOT NULL
                ORDER BY a.id, gc.gate_type, gc.timestamp
            """)
            rows = cursor.fetchall()

            if rows:
                if verbose:
                    print(f"  Using arc-based grouping ({len(rows)} checks matched to arcs)")

                groups = defaultdict(list)
                for check_id, arc_id, gate_type, decision, timestamp, session_id in rows:
                    groups[(arc_id, gate_type)].append((check_id, decision, timestamp, session_id))

                _insert_iterations_from_groups(cursor, groups, verbose)
                conn.commit()

                try:
                    cursor.execute("DETACH DATABASE arc_db")
                except sqlite3.Error:
                    pass
                return

            if verbose:
                print("  No gate checks matched to arcs, falling back to session-based grouping")

            try:
                cursor.execute("DETACH DATABASE arc_db")
            except sqlite3.Error:
                pass

        except sqlite3.Error as e:
            if verbose:
                print(f"  Error accessing arc database: {e}, falling back to session-based grouping")
            try:
                cursor.execute("DETACH DATABASE arc_db")
            except sqlite3.Error:
                pass

    elif verbose:
        print("  Using session-based grouping (arc_analytics.db not found)")
        print("  Note: Run 'arc_analyzer.py extract' first for more accurate recovery rates")

    # Fallback: group by (session_id, gate_type)
    cursor.execute("""
        SELECT id, session_id, gate_type, decision, timestamp
        FROM gate_checks
        WHERE session_id IS NOT NULL
        ORDER BY session_id, gate_type, timestamp
    """)

    rows = cursor.fetchall()
    if not rows:
        return

    groups = defaultdict(list)
    for check_id, session_id, gate_type, decision, timestamp in rows:
        groups[(session_id, gate_type)].append((check_id, decision, timestamp, session_id))

    _insert_iterations_from_groups(cursor, groups, verbose)
    conn.commit()


def extract_all(verbose: bool = True):
    """Full extraction orchestrator - extracts gate check data from all JSONL files.

    Steps:
    1. init_db()
    2. Clear existing gate_checks, gate_issues, gate_iterations data (idempotent re-run)
    3. Load gate_map via get_gate_tools_map() - exit if None
    4. collect_jsonl_files()
    5. Iterate files with tqdm progress bar
    6. Call extract_gates_from_file() per file
    7. Insert gate_checks rows and gate_issues rows (with gate_check_id foreign key)
    8. Call compute_iterations()
    9. Print summary: files processed, files with gates, total gate checks, total issues

    Args:
        verbose: Show progress bars and detailed output
    """
    if verbose:
        print(f"Initializing database at {_sanitize_path(DB_PATH)}")
    conn = init_db(DB_PATH)
    cursor = conn.cursor()

    # Clear existing data (idempotent)
    cursor.execute("DELETE FROM gate_iterations")
    cursor.execute("DELETE FROM gate_issues")
    cursor.execute("DELETE FROM gate_checks")
    conn.commit()

    # Load gate tools map
    gate_map = get_gate_tools_map(conn)
    if gate_map is None:
        print(f"No gate tools discovered in {_sanitize_path(DB_PATH)}. Run 'discover' first.")
        conn.close()
        return

    if verbose:
        print(f"Using {len(gate_map)} discovered gate tools")

    # Collect JSONL files
    files = collect_jsonl_files()
    if not files:
        print(f"No JSONL files found in {_sanitize_path(PROJECTS_DIR)}")
        conn.close()
        return

    if verbose:
        print(f"Found {len(files)} JSONL files to process\n")

    total_checks = 0
    total_issues = 0
    files_with_gates = 0

    # Process files with progress bar
    iterator = tqdm(files, desc="Processing files", unit="file") if verbose else files

    for file_path, project_name in iterator:
        gate_checks = extract_gates_from_file(file_path, project_name, gate_map)

        if gate_checks:
            files_with_gates += 1

        for gc in gate_checks:
            # Insert gate check
            cursor.execute("""
                INSERT INTO gate_checks
                (session_id, project, gate_type, tool_name, decision,
                 feedback_text, feedback_length, timestamp, session_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gc["session_id"], gc["project"], gc["gate_type"], gc["tool_name"],
                gc["decision"], gc["feedback_text"], gc["feedback_length"],
                gc["timestamp"], gc["session_file"]
            ))
            check_id = cursor.lastrowid
            total_checks += 1

            # Insert issues
            for issue in gc["issues"]:
                cursor.execute("""
                    INSERT INTO gate_issues (gate_check_id, severity, title, description)
                    VALUES (?, ?, ?, ?)
                """, (check_id, issue["severity"], issue["title"], issue["description"]))
                total_issues += 1

    conn.commit()

    # Compute iterations
    if verbose:
        print("\nComputing revision iteration sequences...")
    compute_iterations(conn, verbose=verbose)

    conn.close()

    # Print summary
    if verbose:
        print(f"\n{'='*50}")
        print(f"Extraction complete:")
        print(f"  Files processed:     {len(files)}")
        print(f"  Files with gates:    {files_with_gates}")
        print(f"  Total gate checks:   {total_checks}")
        print(f"  Total issues found:  {total_issues}")
        print(f"  Database:            {_sanitize_path(DB_PATH)}")


def _show_gate_tools(conn: sqlite3.Connection):
    """Display discovered gate tools in a formatted table.

    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT tool_name, gate_type, discovery_method
        FROM gate_tools
        ORDER BY gate_type, tool_name
    """)
    rows = cursor.fetchall()

    if not rows:
        print("No gate tools discovered.")
        return

    print(f"\nDiscovered {len(rows)} gate tools:\n")
    print(f"{'Tool Name':<50} {'Gate Type':<20} {'Method':<15}")
    print("-" * 85)
    for tool_name, gate_type, method in rows:
        print(f"{tool_name:<50} {gate_type:<20} {method:<15}")
    print()


def discover_gates(conn: sqlite3.Connection, verbose: bool = False, rediscover: bool = False, manual_gates: Optional[str] = None, model_name: str = DEFAULT_MODEL):
    """Discover review gate tools and cache results in gate_tools table.

    Full discovery orchestrator that:
    1. Handles manual gate registration (if provided)
    2. Uses cached results if available and not rediscovering
    3. Scans JSONL for tool names
    4. Applies heuristic pattern matching
    5. Uses Gemini disambiguation if API key available
    6. Caches results in database

    Args:
        conn: Database connection
        verbose: Show progress bars and detailed output
        rediscover: Force re-discovery even if cache exists
        manual_gates: Comma-separated list of tool names to register manually
        model_name: Model to use for Gemini classification
    """
    cursor = conn.cursor()

    # If manual gates provided, insert and return
    if manual_gates:
        insert_manual_gates(conn, manual_gates)
        _show_gate_tools(conn)
        return

    # Check for cached results
    cursor.execute("SELECT COUNT(*) FROM gate_tools")
    cached_count = cursor.fetchone()[0]

    if cached_count > 0 and not rediscover:
        print(f"Using cached gate tools ({cached_count} tools). Use --rediscover to re-scan.")
        _show_gate_tools(conn)
        return

    if cached_count == 0 and not rediscover:
        if verbose:
            print("No cached gate tools. Starting fresh discovery...")

    # Clear existing cache if rediscovering
    if rediscover and cached_count > 0:
        cursor.execute("DELETE FROM gate_tools")
        conn.commit()
        if verbose:
            print("Cleared existing cache")

    # Step 1: Scan JSONL files for all tool names
    if verbose:
        print("\nStep 1: Scanning JSONL files for tool names...")
    all_tool_names = collect_all_tool_names(verbose=verbose)

    if not all_tool_names:
        print("No tool names found in JSONL files")
        return

    if verbose:
        print(f"Found {len(all_tool_names)} unique tool names")

    # Step 2: Use Gemini to identify gate tools from the full tool list
    api_key = _get_api_key()

    if not api_key:
        print("\nError: No Gemini API key found.")
        print("Gate discovery requires Gemini to identify review gates from tool names.")
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY, or use --gates to specify gate tools manually.")
        return

    if verbose:
        print(f"\nStep 2: Sending {len(all_tool_names)} tool names to Gemini for classification...")

    try:
        discovered_gates = _gemini_classify_gate_tools(list(all_tool_names), api_key, model_name)
    except Exception as e:
        print(f"Error during Gemini classification: {e}")
        return

    if not discovered_gates:
        print("No gate tools identified by Gemini.")
        return

    if verbose:
        print(f"Gemini identified {len(discovered_gates)} gate tools")

    # Step 3: Cache results
    for tool_name, gate_type in discovered_gates.items():
        cursor.execute("""
            INSERT OR REPLACE INTO gate_tools (tool_name, gate_type, discovery_method)
            VALUES (?, ?, 'gemini')
        """, (tool_name, gate_type))

    conn.commit()

    print(f"\nDiscovery complete: {len(discovered_gates)} gate tools found")
    _show_gate_tools(conn)


def _gemini_classify_gate_tools(tool_names: list[str], api_key: str, model_name: str = DEFAULT_MODEL) -> dict[str, str]:
    """Use Gemini to classify which tool names are review/quality gates.

    Sends tool name list to Gemini asking which are review/quality gates and
    what gate_type they are. Returns dict mapping tool_name → gate_type (only gates).

    Args:
        tool_names: List of tool names to classify
        api_key: Gemini API key
        model_name: Model to use for classification

    Returns:
        Dict mapping tool_name -> gate_type for tools classified as gates
    """
    import google.genai as genai

    client = genai.Client(api_key=api_key)

    # Batch in groups of 200 to avoid hitting API limits
    BATCH_SIZE = 200
    all_gate_tools = {}

    for i in range(0, len(tool_names), BATCH_SIZE):
        batch = tool_names[i:i + BATCH_SIZE]

        prompt = f"""You are analyzing tool names from an AI development workflow to identify which are review or quality gates.

Review gates are tools that validate, review, or check code/designs/plans before proceeding. They typically have names like:
- review_code, review_design, review_plan
- codereview, precommit, validate
- qa, audit, check

Given this list of tool names, return a JSON object mapping ONLY the gate tools to their gate type.

Gate types should be one of: review_plan, review_design, review_code, codereview, precommit, validation, qa, audit

Tool names to classify:
{json.dumps(batch, indent=2)}

Return ONLY a JSON object with gate tools, like:
{{
  "mcp__sqlite-project__review_code": "review_code",
  "mcp__pal__codereview": "codereview"
}}

Do not include non-gate tools like Read, Write, Bash, Edit, etc.
"""

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            # Extract text from response
            response_text = response.text.strip()

            # Handle markdown code block wrapping
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1]
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1]
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]

            response_text = response_text.strip()

            # Parse JSON
            try:
                batch_gates = json.loads(response_text)
                if isinstance(batch_gates, dict):
                    all_gate_tools.update(batch_gates)
                else:
                    print(f"Warning: Batch {i//BATCH_SIZE + 1} response was not a JSON object, skipping.")
            except json.JSONDecodeError as e:
                print(f"JSON decode error in batch {i//BATCH_SIZE + 1}: {e}")
                print(f"Problematic text snippet: {response_text[:200]}...")

        except Exception as e:
            print(f"Gemini API error ({type(e).__name__}): {e}")
            # Continue with partial results

    return all_gate_tools


# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_decisions_regex(verbose: bool = False) -> int:
    """Classify UNKNOWN/NULL decisions using regex pattern matching.

    First pass classification that checks feedback_text against regex patterns
    for APPROVED, NEEDS_REVISION, and ESCALATE decisions. Also checks the
    decision field itself in case the tool returned the decision as text.

    This function only processes rows where decision IS NULL OR decision = 'UNKNOWN'.
    Previously classified non-UNKNOWN decisions are NOT overwritten.

    Args:
        verbose: If True, print detailed progress information

    Returns:
        Integer count of reclassified rows
    """
    conn = init_db(DB_PATH)
    cursor = conn.cursor()

    # Query for UNKNOWN/NULL decisions with feedback text.
    # Minimum 20 chars filters out empty/trivial responses (e.g. "ok", "done")
    # that lack enough signal for regex pattern matching.
    cursor.execute("""
        SELECT id, decision, feedback_text
        FROM gate_checks
        WHERE (decision IS NULL OR decision = 'UNKNOWN')
          AND feedback_text IS NOT NULL
          AND LENGTH(feedback_text) > 20
    """)
    rows = cursor.fetchall()

    if not rows:
        if verbose:
            print("No unknown decisions to classify with regex patterns.")
        conn.close()
        return 0

    if verbose:
        print(f"Pass 1 (regex): Processing {len(rows)} unknown decisions...")

    # Define regex patterns for each decision type
    # Patterns are checked in order - first match wins
    patterns = [
        # APPROVED patterns
        (r'(?i)\b(approved|pass(ed)?|accept(ed)?|lgtm|looks good)\b', 'APPROVED'),
        (r'(?i)<disposition>\s*APPROVED\s*</disposition>', 'APPROVED'),
        (r'(?i)\bCODE REVIEW (?:PASSED|APPROVED)\b', 'APPROVED'),

        # NEEDS_REVISION patterns
        (r'(?i)\b(needs?.revision|reject(ed)?|fail(ed)?|revis(e|ion)|denied|not.approved)\b', 'NEEDS_REVISION'),
        (r'(?i)<disposition>\s*NEEDS_REVISION\s*</disposition>', 'NEEDS_REVISION'),
        (r'(?i)\bCODE REVIEW (?:FAILED|REJECTED)\b', 'NEEDS_REVISION'),

        # ESCALATE patterns
        (r'(?i)\b(escalat(e|ed|ion)|manual|human.review)\b', 'ESCALATE'),
        (r'(?i)<disposition>\s*ESCALATE\s*</disposition>', 'ESCALATE'),
    ]

    reclassified = 0

    for row_id, current_decision, feedback_text in rows:
        new_decision = None

        # First check if it is a workflow status message
        if feedback_text and _is_workflow_status_message(feedback_text):
            new_decision = "STATUS_ONLY"

        # Check feedback_text against regex patterns
        if feedback_text and not new_decision:
            new_decision = _extract_two_phase_body_decision(feedback_text)

        if feedback_text and not new_decision:
            for pattern, decision_type in patterns:
                if re.search(pattern, feedback_text):
                    new_decision = decision_type
                    break

        # Update database if we found a decision
        if new_decision:
            cursor.execute(
                "UPDATE gate_checks SET decision = ? WHERE id = ?",
                (new_decision, row_id)
            )
            reclassified += 1

    conn.commit()
    conn.close()

    if verbose:
        print(f"Regex pass: reclassified {reclassified} of {len(rows)} unknown decisions")

    return reclassified


def classify_decisions_gemini(model_name: str = DEFAULT_MODEL, verbose: bool = False) -> int:
    """Classify remaining UNKNOWN decisions using Gemini API.

    Second pass classification that processes decisions still UNKNOWN after
    regex classification. Batches feedback text and sends to Gemini for
    classification into APPROVED, NEEDS_REVISION, or ESCALATE.

    Args:
        model_name: Gemini model to use (from --model CLI flag)
        verbose: If True, print detailed progress information

    Returns:
        Integer count of reclassified rows
    """
    # Check for API key
    api_key = _get_api_key()
    if not api_key:
        if verbose:
            print("\nSkipping Gemini pass: no API key set.")
            print("Set GEMINI_API_KEY or GOOGLE_API_KEY to enable Gemini classification.")
        return 0

    conn = init_db(DB_PATH)
    cursor = conn.cursor()

    # Query for remaining unknowns after regex pass
    cursor.execute("""
        SELECT id, feedback_text
        FROM gate_checks
        WHERE (decision IS NULL OR decision = 'UNKNOWN')
          AND feedback_text IS NOT NULL
          AND LENGTH(feedback_text) > 20
    """)
    rows = cursor.fetchall()

    if not rows:
        if verbose:
            print("\nNo remaining unknown decisions for Gemini classification.")
        conn.close()
        return 0

    if verbose:
        print(f"\nPass 2 (Gemini): Processing {len(rows)} remaining unknowns...")
        print(f"Using model: {model_name}")

    # Import Gemini client
    try:
        from google import genai
    except ImportError:
        print("Error: google-genai package not installed. Run: pip install google-genai")
        conn.close()
        return 0

    client = genai.Client(api_key=api_key)

    # Classification prompt
    CLASSIFY_PROMPT = """You are classifying the outcome of an AI code/design/plan review.
Given the review feedback text below, determine whether the reviewer APPROVED the work or requested REVISION.

Respond with EXACTLY one word: APPROVED, NEEDS_REVISION, ESCALATE, or UNKNOWN

Rules:
- APPROVED: The review passes the work, possibly with minor suggestions that don't block progress.
- NEEDS_REVISION: The review identifies issues that must be fixed before the work can proceed. Any "critical" or "blocking" issues mean NEEDS_REVISION.
- ESCALATE: The review indicates the work cannot be evaluated or needs human intervention.
- UNKNOWN: The text is an error message, doesn't contain a review, or cannot be classified.

Review feedback:
"""

    reclassified = 0
    errors = 0
    BATCH_SIZE = 20

    # Process in batches
    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

    iterator = range(0, len(rows), BATCH_SIZE)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Gemini batches", total=total_batches, unit="batch")

    for i in iterator:
        batch = rows[i:i + BATCH_SIZE]

        for row_id, feedback_text in batch:
            # Truncate very long feedback to stay within token limits
            truncated = feedback_text[:8000] if len(feedback_text) > 8000 else feedback_text

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=CLASSIFY_PROMPT + truncated,
                )
                answer = response.text.strip().upper()

                # Normalize the response
                if "APPROVED" in answer:
                    decision = "APPROVED"
                elif "NEEDS_REVISION" in answer or "REVISION" in answer:
                    decision = "NEEDS_REVISION"
                elif "ESCALATE" in answer:
                    decision = "ESCALATE"
                elif "UNKNOWN" in answer:
                    decision = None  # Leave as unknown
                else:
                    decision = None

                if decision:
                    cursor.execute(
                        "UPDATE gate_checks SET decision = ? WHERE id = ?",
                        (decision, row_id)
                    )
                    reclassified += 1

            except Exception as e:
                errors += 1
                if verbose and errors <= 3:
                    print(f"\n  Warning: Gemini error: {e}")
                elif verbose and errors == 4:
                    print("\n  (suppressing further error warnings)")

        # Commit after each batch
        conn.commit()

    conn.close()

    if verbose:
        print(f"\nGemini pass: classified {reclassified} of {len(rows)} unknowns")
        if errors > 0:
            print(f"  API errors: {errors}")

    return reclassified


def classify_error_types(model_name: str = DEFAULT_MODEL, verbose: bool = False) -> int:
    """Classify error types for NEEDS_REVISION decisions.

    Classifies the WHY behind rejections into one of four categories:
    - SYSTEMATIC: Wrong approach, coherently executed
    - INCOHERENT: Internally contradictory, self-inconsistent
    - OMISSION: Missing required components
    - API_ERROR: Tool/infrastructure failure

    Requires GEMINI_API_KEY to be set. Processes NEEDS_REVISION decisions
    where error_class IS NULL.

    Args:
        model_name: Gemini model to use (from --model CLI flag)
        verbose: If True, print detailed progress information

    Returns:
        Integer count of classified error types
    """
    # API key is REQUIRED here (unlike classify_decisions_gemini which is an
    # optional second pass after regex). Error classification is 100% Gemini —
    # there's no regex fallback for determining SYSTEMATIC vs INCOHERENT vs OMISSION.
    api_key = _get_api_key()
    if not api_key:
        print("Error: No Gemini API key found.")
        print("This command requires the Gemini API. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        return 0

    conn = init_db(DB_PATH)
    cursor = conn.cursor()

    # Query for NEEDS_REVISION decisions without error classification
    cursor.execute("""
        SELECT id, feedback_text
        FROM gate_checks
        WHERE decision = 'NEEDS_REVISION'
          AND error_class IS NULL
          AND feedback_text IS NOT NULL
          AND LENGTH(feedback_text) > 50
    """)
    rows = cursor.fetchall()

    if not rows:
        if verbose:
            print("No unclassified rejections found.")
        conn.close()
        return 0

    if verbose:
        print(f"\n=== Classifying Error Types ===\n")
        print(f"Processing {len(rows)} NEEDS_REVISION decisions...")
        print(f"Using model: {model_name}")

    # Import Gemini client
    try:
        from google import genai
    except ImportError:
        print("Error: google-genai package not installed. Run: pip install google-genai")
        conn.close()
        return 0

    client = genai.Client(api_key=api_key)

    # Error classification prompt
    CLASSIFY_PROMPT = """You are classifying AI agent errors caught by a code/design/plan review gate.

Given this review feedback that REJECTED the agent's work, classify the PRIMARY error type:

SYSTEMATIC — The agent's approach was fundamentally wrong but internally consistent. It misunderstood requirements, chose a wrong architecture, or applied a pattern where it doesn't belong. The work is coherent but incorrect.

INCOHERENT — The agent's work is internally inconsistent. It handled something correctly in one place but not another, contradicted its own plan, or produced randomly varying quality across components. This is a "hot mess" error.

OMISSION — Something was simply left out. Not wrong, not inconsistent, just missing entirely. A required component, test, security check, or documentation was skipped.

API_ERROR — This is not an agent reasoning error but a tool or infrastructure failure. The review tool itself crashed, API calls failed, or there was a system error.

Respond with EXACTLY one word: SYSTEMATIC, INCOHERENT, OMISSION, or API_ERROR

Review feedback:
"""

    classified = 0
    errors = 0
    BATCH_SIZE = 10  # Smaller batches for error classification (needs more careful analysis)

    # Process in batches
    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

    iterator = range(0, len(rows), BATCH_SIZE)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Classifying errors", total=total_batches, unit="batch")

    for i in iterator:
        batch = rows[i:i + BATCH_SIZE]

        for row_id, feedback_text in batch:
            # Truncate very long feedback
            truncated = feedback_text[:8000] if len(feedback_text) > 8000 else feedback_text

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=CLASSIFY_PROMPT + truncated,
                )
                answer = response.text.strip().upper()

                # Normalize the response
                if "SYSTEMATIC" in answer:
                    error_class = "SYSTEMATIC"
                elif "INCOHERENT" in answer:
                    error_class = "INCOHERENT"
                elif "OMISSION" in answer:
                    error_class = "OMISSION"
                elif "API_ERROR" in answer:
                    error_class = "API_ERROR"
                else:
                    error_class = None

                if error_class:
                    cursor.execute(
                        "UPDATE gate_checks SET error_class = ? WHERE id = ?",
                        (error_class, row_id)
                    )
                    classified += 1

            except Exception as e:
                errors += 1
                if verbose and errors <= 3:
                    print(f"\n  Warning: Gemini error: {e}")
                elif verbose and errors == 4:
                    print("\n  (suppressing further error warnings)")

        # Commit after each batch
        conn.commit()

    conn.close()

    if verbose:
        print(f"\nClassified {classified} rejection error types")
        if errors > 0:
            print(f"  API errors: {errors}")

    return classified


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def cmd_discover(args):
    """Discover available review gate tools."""
    conn = init_db()
    discover_gates(
        conn=conn,
        verbose=args.verbose,
        rediscover=args.rediscover,
        manual_gates=args.gates,
        model_name=args.model
    )
    conn.close()


def cmd_extract(args):
    """Extract gate check usage from session logs."""
    # Handle --gates flag: manually insert gates if provided
    if args.gates:
        conn = init_db(DB_PATH)
        insert_manual_gates(conn, args.gates)
        conn.close()

    # Run extraction
    extract_all(verbose=args.verbose)


def cmd_classify(args):
    """Classify gate decisions and feedback."""
    print("\n=== Classifying Gate Decisions ===\n")

    # Pass 1: Regex-based classification (always runs, no API key needed)
    regex_count = classify_decisions_regex(verbose=args.verbose)

    # Pass 2: Gemini-based classification for remaining unknowns
    gemini_count = classify_decisions_gemini(model_name=args.model, verbose=args.verbose)

    total_count = regex_count + gemini_count

    # Recompute iterations now that decisions are classified.
    # compute_iterations splits on APPROVED boundaries, which requires
    # classified decisions to be accurate.
    if total_count > 0:
        print("\nRecomputing iteration sequences with classified decisions...")
        conn = init_db(DB_PATH)
        compute_iterations(conn, verbose=args.verbose)
        conn.close()

    print(f"\n{'='*50}")
    print(f"Classification complete:")
    print(f"  Regex pass:    {regex_count} reclassified")
    print(f"  Gemini pass:   {gemini_count} reclassified")
    print(f"  Total:         {total_count} reclassified this run")
    print(f"\nRun 'stats' to see updated decision distribution.")


def cmd_classify_errors(args):
    """Classify error types in gate feedback."""
    classified_count = classify_error_types(model_name=args.model, verbose=args.verbose)

    if classified_count > 0:
        print(f"\n{'='*50}")
        print(f"Error classification complete: {classified_count} rejections classified")
        print("\nRun 'stats' or 'error-analysis' to see error type distribution.")


def show_stats():
    """Print summary statistics from the gate analytics database.

    Sections:
    1. Overview: Total gate checks count
    2. Gate type distribution: type, count, approved, revised, approve%, avg_time
    3. Decision distribution: decision, count, percentage
    4. Project distribution: project, count, approved, revised, approve%
    5. Feedback length stats: min, max, avg

    If database is empty, prints "No data. Run 'extract' first." and returns.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n=== Review Gate Analysis Summary ===\n")

    # Overview
    cursor.execute("SELECT COUNT(*) FROM gate_checks")
    total = cursor.fetchone()[0]
    print(f"Total gate checks: {total}")

    if total == 0:
        print(f"No data in {_sanitize_path(DB_PATH)}. Run 'extract' first.")
        conn.close()
        return

    # Gate type distribution
    print("\n--- Gate Type Distribution ---")
    cursor.execute("""
        SELECT gate_type, COUNT(*) as count,
               SUM(CASE WHEN decision = 'APPROVED' THEN 1 ELSE 0 END) as approved,
               SUM(CASE WHEN decision = 'NEEDS_REVISION' THEN 1 ELSE 0 END) as needs_revision,
               SUM(CASE WHEN decision = 'ESCALATE' THEN 1 ELSE 0 END) as escalate
        FROM gate_checks
        GROUP BY gate_type
        ORDER BY count DESC
    """)
    print(f"{'Type':<18} {'Count':>6} {'Approved':>9} {'Revised':>9} {'Escalate':>9} {'Approve%':>9}")
    print("-" * 70)
    for row in cursor.fetchall():
        gate_type, count, approved, needs_rev, escalate = row
        approved = approved or 0
        needs_rev = needs_rev or 0
        escalate = escalate or 0
        pct = (approved / count * 100) if count > 0 else 0
        print(f"{gate_type:<18} {count:>6} {approved:>9} {needs_rev:>9} {escalate:>9} {pct:>8.1f}%")

    # Decision distribution
    print("\n--- Decision Distribution ---")
    cursor.execute("""
        SELECT COALESCE(decision, 'UNKNOWN') as decision, COUNT(*) as count
        FROM gate_checks
        GROUP BY decision
        ORDER BY count DESC
    """)
    print(f"{'Decision':<20} {'Count':>8} {'%':>8}")
    print("-" * 38)
    for row in cursor.fetchall():
        decision, count = row
        pct = (count / total * 100) if total > 0 else 0
        print(f"{decision:<20} {count:>8} {pct:>7.1f}%")

    # Project distribution
    print("\n--- Project Distribution ---")
    cursor.execute("""
        SELECT project, COUNT(*) as count,
               SUM(CASE WHEN decision = 'APPROVED' THEN 1 ELSE 0 END) as approved,
               SUM(CASE WHEN decision = 'NEEDS_REVISION' THEN 1 ELSE 0 END) as needs_revision,
               SUM(CASE WHEN decision = 'ESCALATE' THEN 1 ELSE 0 END) as escalate
        FROM gate_checks
        GROUP BY project
        ORDER BY count DESC
    """)
    print(f"{'Project':<30} {'Count':>6} {'Approved':>9} {'Revised':>9} {'Escalate':>9} {'Approve%':>9}")
    print("-" * 77)
    for row in cursor.fetchall():
        project, count, approved, needs_rev, escalate = row
        approved = approved or 0
        needs_rev = needs_rev or 0
        escalate = escalate or 0
        pct = (approved / count * 100) if count > 0 else 0
        display_name = _sanitize_project(project)
        print(f"{display_name:<30} {count:>6} {approved:>9} {needs_rev:>9} {escalate:>9} {pct:>8.1f}%")

    # Feedback length stats
    print("\n--- Feedback Length ---")
    cursor.execute("""
        SELECT
            ROUND(AVG(feedback_length), 0) as avg_len,
            MIN(feedback_length) as min_len,
            MAX(feedback_length) as max_len
        FROM gate_checks
        WHERE feedback_length > 0
    """)
    row = cursor.fetchone()
    if row and row[0]:
        print(f"  Average: {row[0]:.0f} chars")
        print(f"  Min:     {row[1]} chars")
        print(f"  Max:     {row[2]} chars")

    # Error class distribution (for NEEDS_REVISION decisions only)
    print("\n--- Error Class Distribution (NEEDS_REVISION only) ---")
    cursor.execute("""
        SELECT error_class, COUNT(*) as count
        FROM gate_checks
        WHERE decision = 'NEEDS_REVISION' AND error_class IS NOT NULL
        GROUP BY error_class
        ORDER BY count DESC
    """)
    error_rows = cursor.fetchall()

    if error_rows:
        # Calculate total NEEDS_REVISION with error_class
        total_classified = sum(count for _, count in error_rows)
        print(f"{'Error Class':<20} {'Count':>8} {'%':>8}")
        print("-" * 38)
        for error_class, count in error_rows:
            pct = (count / total_classified * 100) if total_classified > 0 else 0
            print(f"{error_class:<20} {count:>8} {pct:>7.1f}%")
    else:
        print("  No error classifications found.")
        print("  Run 'classify-errors' to see error class breakdown.")

    # Overall approval rate (excludes UNKNOWN/STATUS_ONLY from denominator)
    print("\n--- Overall Approval Rate ---")
    cursor.execute("""
        SELECT
            SUM(CASE WHEN decision = 'APPROVED' THEN 1 ELSE 0 END) as approved,
            COUNT(*) as total,
            SUM(CASE WHEN decision IS NULL OR decision = 'UNKNOWN' OR decision = 'STATUS_ONLY' THEN 1 ELSE 0 END) as excluded
        FROM gate_checks
    """)
    row = cursor.fetchone()
    if row:
        approved, total_checks, excluded = row
        approved = approved or 0
        excluded = excluded or 0
        known_total = total_checks - excluded
        if known_total > 0:
            approval_rate = (approved / known_total * 100)
            print(f"  Approved:     {approved} / {known_total} ({approval_rate:.1f}%)")
            print(f"  (Excluded {excluded} UNKNOWN/STATUS_ONLY decisions from denominator)")
        else:
            print("  No classified decisions yet.")

    # Recovery rate (NEEDS_REVISION followed by APPROVED)
    print("\n--- Recovery Rate ---")
    cursor.execute("""
        WITH iterations_with_next AS (
            SELECT
                gi.session_id,
                gi.gate_type,
                gi.iteration_number,
                gi.decision as current_decision,
                LEAD(gi.decision) OVER (
                    PARTITION BY gi.session_id, gi.gate_type
                    ORDER BY gi.iteration_number
                ) as next_decision
            FROM gate_iterations gi
        )
        SELECT
            COUNT(*) as total_revisions,
            SUM(CASE WHEN next_decision = 'APPROVED' THEN 1 ELSE 0 END) as recovered,
            SUM(CASE WHEN next_decision = 'NEEDS_REVISION' THEN 1 ELSE 0 END) as repeated_revision
        FROM iterations_with_next
        WHERE current_decision = 'NEEDS_REVISION'
    """)
    row = cursor.fetchone()
    if row and row[0] and row[0] > 0:
        total_revisions, recovered, repeated = row
        recovered = recovered or 0
        repeated = repeated or 0
        recovery_rate = (recovered / total_revisions * 100) if total_revisions > 0 else 0
        print(f"  After NEEDS_REVISION: {recovered}/{total_revisions} recovered to APPROVED ({recovery_rate:.1f}%)")
        if repeated > 0:
            repeat_rate = (repeated / total_revisions * 100)
            print(f"  Repeated rejections:  {repeated}/{total_revisions} ({repeat_rate:.1f}%)")
    else:
        print("  No iteration data available.")

    conn.close()


def cmd_stats(args):
    """Show gate usage statistics."""
    show_stats()


def show_error_analysis():
    """Show detailed error analysis including recovery rates and arc correlation.

    Sections:
    1. Recovery rate: After NEEDS_REVISION, % that become APPROVED vs repeated NEEDS_REVISION
    2. Recovery rate by error class: Breakdown by SYSTEMATIC, INCOHERENT, OMISSION
    3. Repeat rejection patterns: Sessions with 3+ consecutive NEEDS_REVISION
    4. Optional arc correlation: If arc_analytics.db exists, show error distribution by arc type
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n=== Error Analysis ===\n")

    # Section 1: Overall recovery rate
    print("--- Recovery Rate ---")
    cursor.execute("""
        WITH iterations_with_next AS (
            SELECT
                gi.session_id,
                gi.gate_type,
                gi.iteration_number,
                gi.decision as current_decision,
                gi.gate_check_id,
                LEAD(gi.decision) OVER (
                    PARTITION BY gi.session_id, gi.gate_type
                    ORDER BY gi.iteration_number
                ) as next_decision
            FROM gate_iterations gi
        )
        SELECT
            COUNT(*) as total_revisions,
            SUM(CASE WHEN next_decision = 'APPROVED' THEN 1 ELSE 0 END) as recovered,
            SUM(CASE WHEN next_decision = 'NEEDS_REVISION' THEN 1 ELSE 0 END) as repeated_revision,
            SUM(CASE WHEN next_decision IS NULL THEN 1 ELSE 0 END) as no_follow_up
        FROM iterations_with_next
        WHERE current_decision = 'NEEDS_REVISION'
    """)
    row = cursor.fetchone()

    if row and row[0] and row[0] > 0:
        total_revisions, recovered, repeated, no_follow_up = row
        recovered = recovered or 0
        repeated = repeated or 0
        no_follow_up = no_follow_up or 0

        recovery_rate = (recovered / total_revisions * 100) if total_revisions > 0 else 0
        repeat_rate = (repeated / total_revisions * 100) if total_revisions > 0 else 0

        print(f"Total NEEDS_REVISION decisions with follow-up: {total_revisions}")
        print(f"  Recovered to APPROVED:       {recovered:>4} ({recovery_rate:>5.1f}%)")
        print(f"  Repeated NEEDS_REVISION:     {repeated:>4} ({repeat_rate:>5.1f}%)")
        if no_follow_up > 0:
            print(f"  No follow-up (last in session): {no_follow_up:>4}")
    else:
        print("No recovery data available (no iteration sequences found).")

    # Section 2: Recovery rate by error class
    print("\n--- Recovery Rate by Error Class ---")
    cursor.execute("""
        WITH iterations_with_next AS (
            SELECT
                gi.session_id,
                gi.gate_type,
                gi.iteration_number,
                gi.decision as current_decision,
                gi.gate_check_id,
                LEAD(gi.decision) OVER (
                    PARTITION BY gi.session_id, gi.gate_type
                    ORDER BY gi.iteration_number
                ) as next_decision
            FROM gate_iterations gi
        )
        SELECT
            gc.error_class,
            COUNT(*) as total,
            SUM(CASE WHEN iwn.next_decision = 'APPROVED' THEN 1 ELSE 0 END) as recovered
        FROM iterations_with_next iwn
        JOIN gate_checks gc ON iwn.gate_check_id = gc.id
        WHERE iwn.current_decision = 'NEEDS_REVISION'
          AND gc.error_class IS NOT NULL
          AND iwn.next_decision IS NOT NULL
        GROUP BY gc.error_class
        ORDER BY total DESC
    """)
    error_class_rows = cursor.fetchall()

    if error_class_rows:
        print(f"{'Error Class':<20} {'Total':>6} {'Recovered':>10} {'Recovery%':>10}")
        print("-" * 48)
        for error_class, total, recovered in error_class_rows:
            recovered = recovered or 0
            recovery_pct = (recovered / total * 100) if total > 0 else 0
            print(f"{error_class:<20} {total:>6} {recovered:>10} {recovery_pct:>9.1f}%")
    else:
        print("No error class data available.")
        print("Run 'classify-errors' to classify error types first.")

    # Section 3: Repeat rejection patterns (3+ consecutive NEEDS_REVISION)
    print("\n--- Repeat Rejection Patterns (3+ consecutive NEEDS_REVISION) ---")
    cursor.execute("""
        WITH numbered_iterations AS (
            SELECT
                session_id,
                gate_type,
                decision,
                iteration_number,
                -- Create a grouping key for consecutive decisions (gaps and islands)
                iteration_number - ROW_NUMBER() OVER (
                    PARTITION BY session_id, gate_type, decision
                    ORDER BY iteration_number
                ) as island_group
            FROM gate_iterations
        ),
        streaks AS (
            SELECT
                session_id,
                gate_type,
                decision,
                COUNT(*) as streak_length
            FROM numbered_iterations
            GROUP BY session_id, gate_type, decision, island_group
        )
        SELECT
            s.session_id,
            s.gate_type,
            s.streak_length as revision_count,
            (SELECT project FROM gate_checks gc WHERE gc.session_id = s.session_id LIMIT 1) as project
        FROM streaks s
        WHERE s.decision = 'NEEDS_REVISION' AND s.streak_length >= 3
        ORDER BY s.streak_length DESC
        LIMIT 20
    """)
    repeat_rows = cursor.fetchall()

    if repeat_rows:
        print(f"{'Session ID':<30} {'Gate Type':<18} {'Project':<20} {'Count':>6}")
        print("-" * 77)
        for session_id, gate_type, revision_count, project in repeat_rows:
            # Truncate session_id if too long
            session_display = session_id[:28] + '..' if len(session_id) > 30 else session_id
            project_display = _sanitize_project(project)
            project_display = project_display[:18] + '..' if len(project_display) > 20 else project_display
            print(f"{session_display:<30} {gate_type:<18} {project_display:<20} {revision_count:>6}")

        print(f"\nTotal sessions with 3+ consecutive rejections: {len(repeat_rows)}")
    else:
        print("No sessions with 3+ consecutive rejections found.")

    # Section 4: Optional arc correlation
    print("\n--- Arc Correlation (autonomy levels) ---")

    # Check if arc_analytics.db exists in the same directory
    arc_db_path = DB_PATH.parent / "arc_analytics.db"

    if not arc_db_path.exists():
        print(f"Arc analytics database not found at {_sanitize_path(arc_db_path)}")
        print("Skipping arc correlation. (This is optional - no error)")
    else:
        try:
            # Attach arc database
            cursor.execute(f"ATTACH DATABASE '{str(arc_db_path).replace(chr(39), chr(39)*2)}' AS arc_db")

            # Query for arc correlation - join on timestamp overlap
            # Arc has start_time and end_time, gate_checks has timestamp
            cursor.execute("""
                SELECT
                    a.autonomy_level,
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN gc.decision = 'NEEDS_REVISION' THEN 1 ELSE 0 END) as needs_revision,
                    SUM(CASE WHEN gc.decision = 'APPROVED' THEN 1 ELSE 0 END) as approved,
                    gc.error_class,
                    COUNT(CASE WHEN gc.error_class IS NOT NULL THEN 1 END) as classified_errors
                FROM arc_db.arcs a
                JOIN gate_checks gc ON
                    gc.timestamp >= a.start_time
                    AND gc.timestamp <= a.end_time
                WHERE gc.decision IS NOT NULL AND gc.decision != 'UNKNOWN'
                GROUP BY a.autonomy_level, gc.error_class
                ORDER BY a.autonomy_level, classified_errors DESC
            """)
            arc_rows = cursor.fetchall()

            if arc_rows:
                print(f"{'Autonomy Level':<20} {'Total':>6} {'NeedsRev':>10} {'Approved':>10} {'Error Class':<20} {'Count':>6}")
                print("-" * 80)
                for autonomy_level, total, needs_rev, approved, error_class, classified in arc_rows:
                    needs_rev = needs_rev or 0
                    approved = approved or 0
                    error_display = error_class if error_class else "(unclassified)"
                    print(f"{autonomy_level:<20} {total:>6} {needs_rev:>10} {approved:>10} {error_display:<20} {classified:>6}")
            else:
                print("No overlapping data between arcs and gate checks.")

        except sqlite3.Error as e:
            print(f"Error accessing arc database: {e}")
            print("Skipping arc correlation.")
        finally:
            # Ensure database is detached even if errors occur
            try:
                cursor.execute("DETACH DATABASE arc_db")
            except sqlite3.Error:
                # This may fail if the ATTACH failed; safe to ignore
                pass

    conn.close()


def cmd_error_analysis(args):
    """Analyze error patterns by gate type."""
    show_error_analysis()


def show_overlap():
    """Compute overlap ratio (omega) across review gates.

    The overlap ratio measures how much different gates overlap in what they catch.
    In a multi-stage pipeline, a task (identified by session_id) flows through stages,
    producing artifacts that are checked by different gates. A task can be rejected by
    plan review, revised, and then rejected again by code review for a different reason.

    Omega asks: when two gates both rejected artifacts from the same task, were they
    catching the same problem or different ones?

    - Low omega (near 0) = gates catch different things (complementary, good)
    - High omega (near 1) = gates catch the same things (redundant, wasteful)

    Two levels of analysis:
    - Session-level: did multiple gates reject artifacts from the same session?
    - Error-class: when gates reject the same session, do they flag the same error type?
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n=== Overlap Analysis (ω) ===\n")

    # Check we have data
    cursor.execute("""
        SELECT COUNT(DISTINCT session_id)
        FROM gate_checks
        WHERE decision = 'NEEDS_REVISION' AND session_id IS NOT NULL
    """)
    total_rejected = cursor.fetchone()[0]
    if total_rejected == 0:
        print("No rejected sessions found. Run 'extract' first.")
        conn.close()
        return

    # --- Session-level overlap ---

    # Global omega: fraction of rejected sessions rejected by >1 gate type
    cursor.execute("""
        WITH rejections AS (
            SELECT DISTINCT session_id, gate_type
            FROM gate_checks
            WHERE decision = 'NEEDS_REVISION' AND session_id IS NOT NULL
        ),
        sessions_by_gate_count AS (
            SELECT session_id, COUNT(DISTINCT gate_type) as gate_count
            FROM rejections
            GROUP BY session_id
        )
        SELECT
            SUM(CASE WHEN gate_count > 1 THEN 1 ELSE 0 END) as multi_gate,
            COUNT(*) as total
        FROM sessions_by_gate_count
    """)
    multi_gate, total = cursor.fetchone()
    global_omega = multi_gate / total if total > 0 else 0

    print(f"--- Session-Level Overlap ---")
    print(f"Global ω: {global_omega:.3f}")
    print(f"  Sessions with artifacts rejected by multiple gates: {multi_gate} / {total}")

    # Per-gate rejection counts
    cursor.execute("""
        SELECT gate_type, COUNT(DISTINCT session_id) as rejected
        FROM gate_checks
        WHERE decision = 'NEEDS_REVISION' AND session_id IS NOT NULL
        GROUP BY gate_type
        ORDER BY gate_type
    """)
    gate_rejection_counts = {row[0]: row[1] for row in cursor.fetchall()}
    gate_types = sorted(gate_rejection_counts.keys())

    if not gate_types:
        print("No gate types found.")
        conn.close()
        return

    # Per-gate rejection rates
    print(f"\nPer-gate rejection rates:")
    print(f"{'Gate':<30} {'Rejected':>10} {'Total':>10} {'Rate':>10}")
    print("-" * 62)
    for gt in gate_types:
        cursor.execute("""
            SELECT COUNT(DISTINCT session_id)
            FROM gate_checks
            WHERE gate_type = ? AND session_id IS NOT NULL
        """, (gt,))
        total_seen = cursor.fetchone()[0]
        rejected = gate_rejection_counts[gt]
        rate = rejected / total_seen if total_seen > 0 else 0
        print(f"{gt:<30} {rejected:>10} {total_seen:>10} {rate:>9.1%}")

    # Pairwise overlap (Jaccard: shared / union)
    cursor.execute("""
        WITH rejections AS (
            SELECT DISTINCT session_id, gate_type
            FROM gate_checks
            WHERE decision = 'NEEDS_REVISION' AND session_id IS NOT NULL
        )
        SELECT a.gate_type, b.gate_type, COUNT(DISTINCT a.session_id) as shared
        FROM rejections a
        JOIN rejections b ON a.session_id = b.session_id AND a.gate_type < b.gate_type
        GROUP BY a.gate_type, b.gate_type
    """)
    pairwise_shared = {}
    for gate_a, gate_b, shared in cursor.fetchall():
        pairwise_shared[(gate_a, gate_b)] = shared

    if len(gate_types) > 1:
        print(f"\nPairwise ω (Jaccard: shared / union):")
        col_width = max(len(g) for g in gate_types) + 2
        header = " " * col_width
        for g in gate_types:
            header += f"{g:>{col_width}}"
        print(header)

        for ga in gate_types:
            row = f"{ga:<{col_width}}"
            for gb in gate_types:
                if ga == gb:
                    row += f"{'-':>{col_width}}"
                else:
                    key = (min(ga, gb), max(ga, gb))
                    shared = pairwise_shared.get(key, 0)
                    union = gate_rejection_counts[ga] + gate_rejection_counts[gb] - shared
                    omega = shared / union if union > 0 else 0
                    row += f"{omega:>{col_width}.3f}"
            print(row)

        # Raw counts table
        print(f"\nPairwise raw counts:")
        print(f"{'Pair':<40} {'Shared':>8} {'Union':>8} {'ω':>8}")
        print("-" * 66)
        for i, ga in enumerate(gate_types):
            for gb in gate_types[i + 1:]:
                key = (min(ga, gb), max(ga, gb))
                shared = pairwise_shared.get(key, 0)
                union = gate_rejection_counts[ga] + gate_rejection_counts[gb] - shared
                omega = shared / union if union > 0 else 0
                pair_label = f"{ga} ↔ {gb}"
                print(f"{pair_label:<40} {shared:>8} {union:>8} {omega:>7.3f}")

    # --- Error-class overlap ---
    cursor.execute("""
        SELECT COUNT(*)
        FROM gate_checks
        WHERE decision = 'NEEDS_REVISION'
          AND session_id IS NOT NULL
          AND error_class IS NOT NULL
          AND error_class != 'API_ERROR'
    """)
    classified_count = cursor.fetchone()[0]

    if classified_count > 0 and len(gate_types) > 1:
        print(f"\n--- Error Class Overlap ---")
        print(f"(Using {classified_count} classified rejections)\n")

        error_classes = ['OMISSION', 'SYSTEMATIC', 'INCOHERENT']

        cursor.execute("""
            SELECT gate_type, error_class, COUNT(DISTINCT session_id) as count
            FROM gate_checks
            WHERE decision = 'NEEDS_REVISION'
              AND session_id IS NOT NULL
              AND error_class IS NOT NULL
              AND error_class != 'API_ERROR'
            GROUP BY gate_type, error_class
        """)
        gate_error_counts = {}
        for gate_type, error_class, count in cursor.fetchall():
            gate_error_counts[(gate_type, error_class)] = count

        cursor.execute("""
            WITH classified AS (
                SELECT DISTINCT session_id, gate_type, error_class
                FROM gate_checks
                WHERE decision = 'NEEDS_REVISION'
                  AND session_id IS NOT NULL
                  AND error_class IS NOT NULL
                  AND error_class != 'API_ERROR'
            )
            SELECT a.gate_type, b.gate_type, a.error_class,
                   COUNT(DISTINCT a.session_id) as shared_count
            FROM classified a
            JOIN classified b ON a.session_id = b.session_id
                AND a.error_class = b.error_class
                AND a.gate_type < b.gate_type
            GROUP BY a.gate_type, b.gate_type, a.error_class
        """)
        shared_by_error = {}
        for ga, gb, ec, count in cursor.fetchall():
            shared_by_error[(ga, gb, ec)] = count

        for i, ga in enumerate(gate_types):
            for gb in gate_types[i + 1:]:
                key = (min(ga, gb), max(ga, gb))
                shared_total = pairwise_shared.get(key, 0)
                if shared_total == 0:
                    continue

                print(f"  {ga} ↔ {gb}:")
                print(f"    Same error class (true redundancy):")
                any_same = False
                for ec in error_classes:
                    shared = shared_by_error.get((key[0], key[1], ec), 0)
                    total_a = gate_error_counts.get((ga, ec), 0)
                    total_b = gate_error_counts.get((gb, ec), 0)
                    union = total_a + total_b - shared
                    if shared > 0 or (total_a > 0 and total_b > 0):
                        any_same = True
                        print(f"      {ec:<14} {shared:>3} shared / {union:>3} total")
                if not any_same:
                    print(f"      (none)")
                print()

    # --- Interpretation ---
    print("--- Interpretation ---")
    if global_omega < 0.15:
        print(f"  Global ω = {global_omega:.3f} → LOW overlap. Gates are largely complementary.")
        print(f"  Each gate catches different problems. Minimal redundancy.")
    elif global_omega < 0.35:
        print(f"  Global ω = {global_omega:.3f} → MODERATE overlap. Some redundancy exists.")
        print(f"  Gates share some catches but still provide distinct value.")
    else:
        print(f"  Global ω = {global_omega:.3f} → HIGH overlap. Significant redundancy.")
        print(f"  Consider whether all gates are necessary or could be consolidated.")

    conn.close()


def cmd_compute_overlap(args):
    """Compute overlap ratio across gates."""
    show_overlap()


def generate_report() -> str:
    """Generate markdown report with gate analysis summary.

    Creates a comprehensive report with 7 sections:
    1. Header with summary stats
    2. Gate Discovery Summary
    3. Decision Distribution
    4. Error Classification
    5. Recovery Analysis
    6. Per-Project Summary
    7. Methodology Note

    Returns:
        Markdown-formatted report string
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if we have data
    cursor.execute("SELECT COUNT(*) FROM gate_checks")
    total_checks = cursor.fetchone()[0]

    if total_checks == 0:
        conn.close()
        return "# Gate Analysis Report\n\nNo data available. Run `extract` to populate the database.\n"

    lines = []

    # ========================================================================
    # SECTION 1: HEADER
    # ========================================================================
    lines.append("# Gate Analysis Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Get summary stats
    cursor.execute("SELECT COUNT(DISTINCT project) FROM gate_checks")
    total_projects = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM gate_checks WHERE decision = 'NEEDS_REVISION'")
    total_rejections = cursor.fetchone()[0]

    lines.append("## Summary")
    lines.append(f"- **Total Gate Checks:** {total_checks}")
    lines.append(f"- **Total Rejections:** {total_rejections}")
    lines.append(f"- **Projects Analyzed:** {total_projects}")
    lines.append("")

    # ========================================================================
    # SECTION 2: GATE DISCOVERY SUMMARY
    # ========================================================================
    lines.append("## Gate Discovery Summary")
    lines.append("")
    cursor.execute("""
        SELECT tool_name, gate_type, discovery_method
        FROM gate_tools
        ORDER BY gate_type, tool_name
    """)
    rows = cursor.fetchall()

    if rows:
        lines.append("| Tool Name | Gate Type | Discovery Method |")
        lines.append("|-----------|-----------|------------------|")
        for tool_name, gate_type, discovery_method in rows:
            lines.append(f"| {tool_name} | {gate_type} | {discovery_method} |")
    else:
        lines.append("*No gate tools discovered. Run `discover` first.*")

    lines.append("")

    # ========================================================================
    # SECTION 3: DECISION DISTRIBUTION
    # ========================================================================
    lines.append("## Decision Distribution")
    lines.append("")

    cursor.execute("""
        SELECT COALESCE(decision, 'UNKNOWN') as decision, COUNT(*) as count
        FROM gate_checks
        GROUP BY decision
        ORDER BY count DESC
    """)
    decision_rows = cursor.fetchall()

    if decision_rows:
        lines.append("| Decision | Count | Percentage |")
        lines.append("|----------|-------|------------|")

        approved_count = 0
        for decision, count in decision_rows:
            pct = (count / total_checks * 100) if total_checks > 0 else 0
            lines.append(f"| {decision} | {count} | {pct:.1f}% |")
            if decision == "APPROVED":
                approved_count = count

        # Overall approval rate
        # Exclude UNKNOWN and STATUS_ONLY from denominator (consistent with stats command)
        excluded_count = sum(count for decision, count in decision_rows if decision in ("UNKNOWN", "STATUS_ONLY"))
        known_total = total_checks - excluded_count
        approval_rate = (approved_count / known_total * 100) if known_total > 0 else 0
        lines.append("")
        lines.append(f"**Overall Approval Rate:** {approval_rate:.1f}% (excluding {excluded_count} UNKNOWN/STATUS_ONLY)")
    else:
        lines.append("*No decision data available.*")

    lines.append("")

    # ========================================================================
    # SECTION 4: ERROR CLASSIFICATION
    # ========================================================================
    lines.append("## Error Classification")
    lines.append("")

    cursor.execute("""
        SELECT COALESCE(error_class, 'UNCLASSIFIED') as error_class, COUNT(*) as count
        FROM gate_checks
        WHERE decision = 'NEEDS_REVISION'
        GROUP BY error_class
        ORDER BY count DESC
    """)
    error_rows = cursor.fetchall()

    if error_rows:
        lines.append("| Error Class | Count | Percentage |")
        lines.append("|-------------|-------|------------|")

        dominant_class = None
        dominant_count = 0

        for error_class, count in error_rows:
            pct = (count / total_rejections * 100) if total_rejections > 0 else 0
            lines.append(f"| {error_class} | {count} | {pct:.1f}% |")

            if count > dominant_count and error_class != 'UNCLASSIFIED':
                dominant_class = error_class
                dominant_count = count

        lines.append("")
        if dominant_class:
            lines.append(f"**Dominant Error Type:** {dominant_class} ({dominant_count} occurrences)")
        else:
            lines.append("**Note:** Run `classify-errors` to populate error classification data.")
    else:
        lines.append("*No rejection data available. Run `classify` and `classify-errors` to populate classification data.*")

    lines.append("")

    # ========================================================================
    # SECTION 5: RECOVERY ANALYSIS
    # ========================================================================
    lines.append("## Recovery Analysis")
    lines.append("")

    # Calculate recovery rate (NEEDS_REVISION → APPROVED transitions)
    cursor.execute("""
        WITH iterations_with_next AS (
            SELECT
                gi.session_id,
                gi.gate_type,
                gi.iteration_number,
                gi.decision as current_decision,
                gi.gate_check_id,
                LEAD(gi.decision) OVER (
                    PARTITION BY gi.session_id, gi.gate_type
                    ORDER BY gi.iteration_number
                ) as next_decision
            FROM gate_iterations gi
        )
        SELECT
            COUNT(*) as total_revisions,
            SUM(CASE WHEN next_decision = 'APPROVED' THEN 1 ELSE 0 END) as recovered
        FROM iterations_with_next
        WHERE current_decision = 'NEEDS_REVISION'
    """)
    recovery_row = cursor.fetchone()

    if recovery_row and recovery_row[0] and recovery_row[0] > 0:
        total_revisions, recovered = recovery_row
        recovered = recovered or 0
        recovery_rate = (recovered / total_revisions * 100) if total_revisions > 0 else 0
        lines.append(f"**Overall Recovery Rate:** {recovery_rate:.1f}% ({recovered}/{total_revisions} sessions)")
        lines.append("")

        # Recovery rate by error class
        lines.append("### Recovery Rate by Error Class")
        lines.append("")
        cursor.execute("""
            WITH iterations_with_next AS (
                SELECT
                    gi.session_id,
                    gi.gate_type,
                    gi.iteration_number,
                    gi.decision as current_decision,
                    gi.gate_check_id,
                    LEAD(gi.decision) OVER (
                        PARTITION BY gi.session_id, gi.gate_type
                        ORDER BY gi.iteration_number
                    ) as next_decision
                FROM gate_iterations gi
            )
            SELECT
                gc.error_class,
                COUNT(*) as total_revisions,
                SUM(CASE WHEN iwn.next_decision = 'APPROVED' THEN 1 ELSE 0 END) as recovered
            FROM iterations_with_next iwn
            JOIN gate_checks gc ON iwn.gate_check_id = gc.id
            WHERE iwn.current_decision = 'NEEDS_REVISION'
              AND gc.error_class IS NOT NULL
            GROUP BY gc.error_class
            ORDER BY recovered DESC
        """)
        recovery_class_rows = cursor.fetchall()

        if recovery_class_rows:
            lines.append("| Error Class | Recovered | Total | Recovery Rate |")
            lines.append("|-------------|-----------|-------|---------------|")
            for error_class, total, recovered in recovery_class_rows:
                recovered = recovered or 0
                rate = (recovered / total * 100) if total > 0 else 0
                lines.append(f"| {error_class} | {recovered} | {total} | {rate:.1f}% |")
        else:
            lines.append("*No error classification data available for recovery analysis. Run `classify-errors`.*")
    else:
        lines.append("*No revision iterations found. Run `extract` to populate iteration data.*")

    lines.append("")

    # ========================================================================
    # SECTION 6: PER-PROJECT SUMMARY
    # ========================================================================
    lines.append("## Per-Project Summary")
    lines.append("")

    cursor.execute("""
        SELECT project,
               COUNT(*) as total,
               SUM(CASE WHEN decision = 'APPROVED' THEN 1 ELSE 0 END) as approved,
               gate_type
        FROM gate_checks
        GROUP BY project, gate_type
    """)
    project_gate_rows = cursor.fetchall()

    if project_gate_rows:
        # Aggregate by project to find dominant gate type
        project_stats = defaultdict(lambda: {'total': 0, 'approved': 0, 'gate_types': defaultdict(int)})

        for project, total, approved, gate_type in project_gate_rows:
            project_stats[project]['total'] += total
            project_stats[project]['approved'] += approved or 0
            project_stats[project]['gate_types'][gate_type] += total

        lines.append("| Project | Total Checks | Approval Rate | Dominant Gate Type |")
        lines.append("|---------|--------------|---------------|-------------------|")

        for project in sorted(project_stats.keys(), key=lambda p: project_stats[p]['total'], reverse=True):
            stats = project_stats[project]
            total = stats['total']
            approved = stats['approved']
            approval_rate = (approved / total * 100) if total > 0 else 0

            # Find dominant gate type
            dominant_gate = max(stats['gate_types'].items(), key=lambda x: x[1])[0] if stats['gate_types'] else 'N/A'

            display_name = _sanitize_project(project)
            lines.append(f"| {display_name} | {total} | {approval_rate:.1f}% | {dominant_gate} |")
    else:
        lines.append("*No project data available.*")

    lines.append("")

    # ========================================================================
    # SECTION 7: METHODOLOGY NOTE
    # ========================================================================
    lines.append("## Methodology")
    lines.append("")
    lines.append("This analysis uses a two-pass classification approach:")
    lines.append("")
    lines.append("1. **Regex Pattern Matching:** Fast heuristic classification of common decision patterns")
    lines.append("2. **Gemini Classification:** AI-powered classification for ambiguous cases")
    lines.append("")
    lines.append("### Error Taxonomy")
    lines.append("")
    lines.append("- **SYSTEMATIC:** Wrong approach, coherently executed")
    lines.append("- **INCOHERENT:** Internally contradictory, self-inconsistent")
    lines.append("- **OMISSION:** Missing required components")
    lines.append("- **API_ERROR:** Tool/infrastructure failure")
    lines.append("")
    lines.append("### Data Sources")
    lines.append("")
    lines.append(f"- Session logs from: `{_sanitize_path(PROJECTS_DIR)}`")
    lines.append(f"- Analysis database: `{_sanitize_path(DB_PATH)}`")
    lines.append("")

    conn.close()

    return "\n".join(lines)


def cmd_report(args):
    """Generate markdown report."""
    report = generate_report()

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {_sanitize_path(args.output)}")
    else:
        print(report)


def main():
    """Main entry point with argparse dispatch."""
    global CLAUDE_DIR, PROJECTS_DIR, DB_PATH

    parser = argparse.ArgumentParser(
        description="Analyze Claude Code review gate usage patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gate_analyzer.py discover --rediscover
  python gate_analyzer.py extract --data-dir /custom/path
  python gate_analyzer.py stats --gates codereview,precommit
  python gate_analyzer.py error-analysis -v
"""
    )

    # Global flags
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override default .claude directory path"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model for classification (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--gates",
        help="Comma-separated list of gate tool names to process (skips discovery)"
    )
    parser.add_argument(
        "--rediscover",
        action="store_true",
        help="Force rediscovery of gate tools"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("discover", help="Discover available review gate tools")
    subparsers.add_parser("extract", help="Extract gate check usage from session logs")
    subparsers.add_parser("classify", help="Classify gate decisions and feedback [R2]")
    subparsers.add_parser("classify-errors", help="Classify error types in gate feedback [R2]")
    subparsers.add_parser("stats", help="Show gate usage statistics")
    subparsers.add_parser("error-analysis", help="Analyze error patterns by gate type [R2]")
    subparsers.add_parser("compute-overlap", help="Compute overlap ratio (ω) across gates")

    report_parser = subparsers.add_parser("report", help="Generate markdown report [R2]")
    report_parser.add_argument("--output", "-o", help="Write report to file instead of stdout")

    args = parser.parse_args()

    # Override global paths if --data-dir specified
    if args.data_dir:
        CLAUDE_DIR = args.data_dir
        PROJECTS_DIR = CLAUDE_DIR / "projects"

    # Dispatch to command handler
    if not args.command:
        parser.print_help()
        return

    command_map = {
        "discover": cmd_discover,
        "extract": cmd_extract,
        "classify": cmd_classify,
        "classify-errors": cmd_classify_errors,
        "stats": cmd_stats,
        "error-analysis": cmd_error_analysis,
        "compute-overlap": cmd_compute_overlap,
        "report": cmd_report,
    }

    handler = command_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

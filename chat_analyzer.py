#!/usr/bin/env python3
"""
Claude Code Chat Log Analyzer

Extracts patterns from Claude Code conversation history using embeddings and clustering.

Usage:
    python chat_analyzer.py extract    # Parse JSONL files into SQLite
    python chat_analyzer.py embed      # Generate embeddings via Gemini API
    python chat_analyzer.py cluster    # Run HDBSCAN clustering
    python chat_analyzer.py stats      # Show statistics
    python chat_analyzer.py visualize  # Generate UMAP visualization
"""

import json
import sqlite3
import struct
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse
from typing import Optional

import numpy as np
from tqdm import tqdm

# Constants
CLAUDE_DIR = Path.home() / ".claude"
HISTORY_FILE = CLAUDE_DIR / "history.jsonl"
PROJECTS_DIR = CLAUDE_DIR / "projects"
DB_PATH = Path(__file__).parent / "chat_analytics.db"

# Schema
SCHEMA = """
-- Prompts from history.jsonl (user inputs)
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    project TEXT,
    content TEXT NOT NULL,
    turn_number INTEGER DEFAULT 1,
    embedding BLOB,
    cluster_id INTEGER
);

-- Sessions extracted from project JSONL files
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    summary TEXT,
    start_time INTEGER,
    end_time INTEGER,
    message_count INTEGER DEFAULT 0,
    is_subagent INTEGER DEFAULT 0
);

-- Tool usage extracted from sessions
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    tool_name TEXT NOT NULL,
    timestamp INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_prompts_project ON prompts(project);
CREATE INDEX IF NOT EXISTS idx_prompts_cluster ON prompts(cluster_id);
CREATE INDEX IF NOT EXISTS idx_prompts_timestamp ON prompts(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name);
"""


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def parse_history_jsonl(history_file: Path = HISTORY_FILE) -> list[dict]:
    """Parse history.jsonl and extract prompts."""
    prompts = []
    if not history_file.exists():
        print(f"Warning: {history_file} does not exist")
        return prompts
    with open(history_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if 'display' in entry and 'timestamp' in entry:
                    prompts.append({
                        'timestamp': entry['timestamp'],
                        'project': entry.get('project', 'unknown'),
                        'content': entry['display']
                    })
            except json.JSONDecodeError:
                continue
    return prompts


def extract_sessions_from_project(project_dir: Path) -> list[dict]:
    """Extract session metadata from project JSONL files."""
    sessions = []

    for jsonl_file in project_dir.glob("*.jsonl"):
        is_subagent = jsonl_file.name.startswith("agent-")
        session_id = jsonl_file.stem

        summary = None
        start_time = None
        end_time = None
        message_count = 0
        tool_calls = []

        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())

                        # Get summary
                        if entry.get('type') == 'summary':
                            summary = entry.get('summary')

                        # Track timestamps
                        ts = entry.get('timestamp')
                        if ts:
                            if start_time is None or ts < start_time:
                                start_time = ts
                            if end_time is None or ts > end_time:
                                end_time = ts

                        # Count messages
                        if entry.get('type') in ('user', 'assistant'):
                            message_count += 1

                            # Extract tool calls from assistant messages
                            if entry.get('type') == 'assistant':
                                msg = entry.get('message', {})
                                content = msg.get('content', [])
                                if isinstance(content, list):
                                    for block in content:
                                        if isinstance(block, dict) and block.get('type') == 'tool_use':
                                            tool_calls.append({
                                                'tool_name': block.get('name'),
                                                'timestamp': ts
                                            })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {jsonl_file}: {e}")
            continue

        sessions.append({
            'id': session_id,
            'project': project_dir.name,  # Use dir name for consistency with prompts
            'summary': summary,
            'start_time': start_time,
            'end_time': end_time,
            'message_count': message_count,
            'is_subagent': is_subagent,
            'tool_calls': tool_calls
        })

    return sessions


def ingest_prompts(conn: sqlite3.Connection, prompts: list[dict]):
    """Insert prompts into database."""
    cursor = conn.cursor()

    # Group by project to assign turn numbers
    by_project = defaultdict(list)
    for p in prompts:
        by_project[p['project']].append(p)

    # Sort each project's prompts by timestamp and assign turn numbers
    for project, project_prompts in by_project.items():
        project_prompts.sort(key=lambda x: x['timestamp'])
        for i, p in enumerate(project_prompts, 1):
            p['turn_number'] = i

    # Insert all prompts
    for p in prompts:
        cursor.execute("""
            INSERT OR REPLACE INTO prompts (timestamp, project, content, turn_number)
            VALUES (?, ?, ?, ?)
        """, (p['timestamp'], p['project'], p['content'], p['turn_number']))

    conn.commit()
    return len(prompts)


def ingest_sessions(conn: sqlite3.Connection, sessions: list[dict]):
    """Insert sessions and tool calls into database."""
    cursor = conn.cursor()

    for s in sessions:
        cursor.execute("""
            INSERT OR REPLACE INTO sessions
            (id, project, summary, start_time, end_time, message_count, is_subagent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (s['id'], s['project'], s['summary'], s['start_time'],
              s['end_time'], s['message_count'], 1 if s['is_subagent'] else 0))

        for tc in s.get('tool_calls', []):
            cursor.execute("""
                INSERT INTO tool_calls (session_id, tool_name, timestamp)
                VALUES (?, ?, ?)
            """, (s['id'], tc['tool_name'], tc['timestamp']))

    conn.commit()
    return len(sessions)


def run_extraction(verbose: bool = True):
    """Run full extraction pipeline."""
    if verbose:
        print(f"Initializing database at {DB_PATH}")
    conn = init_db()
    cursor = conn.cursor()

    # Clear existing data for idempotent extraction
    cursor.execute("DELETE FROM tool_calls")
    cursor.execute("DELETE FROM sessions")
    cursor.execute("DELETE FROM prompts")
    conn.commit()

    # Extract prompts from history
    if verbose:
        print(f"Parsing {HISTORY_FILE}...")
    prompts = parse_history_jsonl()
    prompt_count = ingest_prompts(conn, prompts)
    if verbose:
        print(f"  Ingested {prompt_count} prompts")

    # Extract sessions from each project
    if verbose:
        print(f"Scanning {PROJECTS_DIR}...")

    total_sessions = 0
    if not PROJECTS_DIR.exists():
        print(f"Warning: {PROJECTS_DIR} does not exist")
        conn.close()
        return prompt_count, 0

    for project_dir in PROJECTS_DIR.iterdir():
        if project_dir.is_dir():
            sessions = extract_sessions_from_project(project_dir)
            count = ingest_sessions(conn, sessions)
            total_sessions += count
            if verbose and count > 0:
                print(f"  {project_dir.name}: {count} sessions")

    if verbose:
        print(f"Total: {total_sessions} sessions")

    conn.close()
    return prompt_count, total_sessions


def show_stats(db_path: Path = DB_PATH):
    """Show database statistics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n=== Chat Analytics Summary ===\n")

    # Prompt stats
    cursor.execute("SELECT COUNT(*) FROM prompts")
    print(f"Total prompts: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(DISTINCT project) FROM prompts")
    print(f"Unique projects: {cursor.fetchone()[0]}")

    # Session stats
    cursor.execute("SELECT COUNT(*) FROM sessions WHERE is_subagent = 0")
    print(f"Main sessions: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM sessions WHERE is_subagent = 1")
    print(f"Subagent sessions: {cursor.fetchone()[0]}")

    # Tool usage
    cursor.execute("""
        SELECT tool_name, COUNT(*) as count
        FROM tool_calls
        GROUP BY tool_name
        ORDER BY count DESC
        LIMIT 10
    """)
    print("\nTop 10 tools used:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    # Project activity
    cursor.execute("""
        SELECT project, COUNT(*) as count
        FROM prompts
        GROUP BY project
        ORDER BY count DESC
        LIMIT 10
    """)
    print("\nTop 10 projects by prompt count:")
    for row in cursor.fetchall():
        # Shorten project path for display
        project = row[0].replace(str(Path.home()), '~')
        print(f"  {project}: {row[1]}")

    # Time range
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM prompts")
    min_ts, max_ts = cursor.fetchone()
    if min_ts and max_ts:
        start = datetime.fromtimestamp(min_ts / 1000)
        end = datetime.fromtimestamp(max_ts / 1000)
        print(f"\nDate range: {start.date()} to {end.date()}")

    conn.close()


# ============================================================================
# Phase 2: Embedding Generation
# ============================================================================

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMS = 768  # Recommended for efficiency
EMBEDDING_TASK = "CLASSIFICATION"  # Good for clustering
BATCH_SIZE = 100  # Prompts per API call


def embedding_to_blob(embedding: list[float]) -> bytes:
    """Convert embedding list to SQLite BLOB."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Convert SQLite BLOB back to numpy array."""
    n = len(blob) // 4  # 4 bytes per float
    return np.array(struct.unpack(f'{n}f', blob))


def generate_embeddings(db_path: Path = DB_PATH, batch_size: int = BATCH_SIZE):
    """Generate embeddings for prompts without them."""
    from google import genai
    from google.genai import types

    # Initialize client with API key from env
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    client = genai.Client(api_key=api_key)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get prompts without embeddings
    cursor.execute("""
        SELECT id, content FROM prompts
        WHERE embedding IS NULL AND content IS NOT NULL AND content != ''
        ORDER BY id
    """)
    rows = cursor.fetchall()

    if not rows:
        print("All prompts already have embeddings.")
        conn.close()
        return

    print(f"Generating embeddings for {len(rows)} prompts...")

    # Process in batches
    for i in tqdm(range(0, len(rows), batch_size)):
        batch = rows[i:i + batch_size]
        ids = [r[0] for r in batch]
        contents = [r[1] for r in batch]

        try:
            # Call Gemini API with batch
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type=EMBEDDING_TASK,
                    output_dimensionality=EMBEDDING_DIMS
                )
            )

            # Store embeddings
            for idx, embedding_obj in enumerate(result.embeddings):
                embedding = embedding_obj.values
                blob = embedding_to_blob(embedding)
                cursor.execute(
                    "UPDATE prompts SET embedding = ? WHERE id = ?",
                    (blob, ids[idx])
                )

            conn.commit()

        except Exception as e:
            print(f"\nError at batch {i//batch_size}: {e}")
            conn.commit()  # Save progress
            continue

    conn.close()
    print("Embedding generation complete.")


# ============================================================================
# Phase 3: Clustering
# ============================================================================

def run_clustering(db_path: Path = DB_PATH, min_cluster_size: int = 50):
    """Run HDBSCAN clustering on embeddings."""
    import hdbscan

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Load all embeddings
    cursor.execute("""
        SELECT id, embedding FROM prompts
        WHERE embedding IS NOT NULL
        ORDER BY id
    """)
    rows = cursor.fetchall()

    if not rows:
        print("No embeddings found. Run 'embed' first.")
        conn.close()
        return

    print(f"Clustering {len(rows)} prompts...")

    ids = [r[0] for r in rows]
    embeddings = np.array([blob_to_embedding(r[1]) for r in rows])

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(embeddings)

    # Update database
    for idx, cluster_id in enumerate(labels):
        cursor.execute(
            "UPDATE prompts SET cluster_id = ? WHERE id = ?",
            (int(cluster_id), ids[idx])
        )

    conn.commit()

    # Stats
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Found {n_clusters} clusters, {n_noise} noise points ({n_noise*100/len(labels):.1f}%)")

    conn.close()


def show_clusters(db_path: Path = DB_PATH, samples: int = 5):
    """Show sample prompts from each cluster."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get cluster sizes
    cursor.execute("""
        SELECT cluster_id, COUNT(*) as size
        FROM prompts
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
        ORDER BY size DESC
    """)
    clusters = cursor.fetchall()

    print(f"\n=== Cluster Analysis ({len(clusters)} clusters) ===\n")

    for cluster_id, size in clusters[:20]:  # Top 20 clusters
        label = "NOISE" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"\n--- {label} ({size} prompts) ---")

        cursor.execute("""
            SELECT content FROM prompts
            WHERE cluster_id = ?
            ORDER BY RANDOM()
            LIMIT ?
        """, (cluster_id, samples))

        for row in cursor.fetchall():
            content = row[0][:100] + "..." if len(row[0]) > 100 else row[0]
            print(f"  - {content}")

    conn.close()


# ============================================================================
# Phase 4: Visualization
# ============================================================================

def generate_visualization(db_path: Path = DB_PATH, output: str = "clusters.html"):
    """Generate UMAP visualization of clusters."""
    import umap

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT embedding, cluster_id, content FROM prompts
        WHERE embedding IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No embeddings found.")
        return

    print(f"Generating UMAP projection for {len(rows)} points...")

    embeddings = np.array([blob_to_embedding(r[0]) for r in rows])
    clusters = [r[1] if r[1] is not None else -1 for r in rows]
    contents = [r[2][:50] + "..." if len(r[2]) > 50 else r[2] for r in rows]

    # UMAP reduction to 2D
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding_2d = reducer.fit_transform(embeddings)

    # Generate HTML with Plotly
    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'cluster': [str(c) for c in clusters],
            'content': contents
        })

        fig = px.scatter(
            df, x='x', y='y', color='cluster',
            hover_data=['content'],
            title='Claude Code Prompt Clusters'
        )
        fig.write_html(output)
        print(f"Visualization saved to {output}")

    except ImportError:
        # Fallback: save coordinates to CSV
        import csv
        csv_output = output.replace('.html', '.csv')
        with open(csv_output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'cluster', 'content'])
            for i, (x, y) in enumerate(embedding_2d):
                writer.writerow([x, y, clusters[i], contents[i]])
        print(f"Coordinates saved to {csv_output} (install plotly for HTML)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude Code Chat Log Analyzer")
    parser.add_argument("command", choices=["extract", "embed", "cluster", "clusters", "stats", "visualize"],
                        help="Command to run")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--data-dir", type=Path,
                        help="Claude data directory (default: ~/.claude)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for embedding generation")
    parser.add_argument("--min-cluster-size", type=int, default=50,
                        help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of sample prompts per cluster")

    args = parser.parse_args()

    # Override default data directory if specified
    if args.data_dir:
        CLAUDE_DIR = args.data_dir
        HISTORY_FILE = CLAUDE_DIR / "history.jsonl"
        PROJECTS_DIR = CLAUDE_DIR / "projects"

    if args.command == "extract":
        run_extraction(verbose=args.verbose)
    elif args.command == "embed":
        generate_embeddings(batch_size=args.batch_size)
    elif args.command == "cluster":
        run_clustering(min_cluster_size=args.min_cluster_size)
    elif args.command == "clusters":
        show_clusters(samples=args.samples)
    elif args.command == "stats":
        show_stats()
    elif args.command == "visualize":
        generate_visualization()

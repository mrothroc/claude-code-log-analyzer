# Claude Code Log Analyzer

Python tools for analyzing Claude Code conversation history. Extract patterns, measure autonomous work hours, and understand your AI-assisted development workflow.

This toolkit was developed during a 97-day research project analyzing 543 hours of autonomous AI-assisted work across 165 releases. The methodology and findings are documented in the [accompanying presentation](https://michael.roth.rocks/research/543-hours/).

## What It Does

Two complementary analyses:

1. **Arc Analysis** - Identifies coherent units of work ("arcs") from your prompts. What kind of work are you doing?
2. **Agent Analysis** - Counts autonomous agent sessions and their durations. How much autonomous work happened?

Tools:
- **arc_analyzer.py** - Detects work arcs using semantic analysis (Gemini Flash Lite), counts agent session hours
- **chat_analyzer.py** - Clusters prompts using Gemini embeddings + HDBSCAN to reveal interaction patterns

## Quick Start

```bash
# Clone and setup
git clone https://github.com/mrothroc/claude-code-log-analyzer.git
cd claude-code-log-analyzer

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your Gemini API key (either works)
export GEMINI_API_KEY="your-key"
# or: export GOOGLE_API_KEY="your-key"

# Analyze agent sessions (autonomous hours)
python arc_analyzer.py agents    # Required first for agent stats
python arc_analyzer.py stats     # Shows arc + agent statistics

# Extract and analyze arcs (work patterns)
python arc_analyzer.py extract   # Requires GEMINI_API_KEY
python arc_analyzer.py stats     # Shows arc + agent statistics

# Cluster prompts by similarity
python chat_analyzer.py extract
python chat_analyzer.py embed
python chat_analyzer.py cluster
python chat_analyzer.py visualize
```

## Data Sources

The scripts read from your local Claude Code data directory (`~/.claude/`):

| File | Content |
|------|---------|
| `projects/*/` | Project session directories |
| `projects/*/*.jsonl` | Full conversation transcripts |
| `projects/*/agent-*.jsonl` | Subagent (autonomous) session logs |

### Custom Data Directory

To analyze logs from a different location (backup, server, another user):

```bash
python arc_analyzer.py agents --data-dir /path/to/claude-data
python chat_analyzer.py extract --data-dir /path/to/claude-data
```

### Multiple Claude Directories

If you have Claude Code data in multiple locations (e.g., different user accounts on the same machine), set:

```bash
export CLAUDE_EXTRA_DIRS="/Users/other/.claude/projects:/shared/.claude/projects"
```

## Arc Analyzer

Identifies "arcs" - coherent units of work that may span multiple prompts, tool uses, and agent spawns. Also counts autonomous agent session hours directly.

```bash
python arc_analyzer.py agents      # Count agent sessions and autonomous hours
python arc_analyzer.py extract     # Detect arcs from user prompts (uses Gemini)
python arc_analyzer.py stats       # Show combined statistics
python arc_analyzer.py list        # List all arcs
python arc_analyzer.py detail ID   # Show details for specific arc
python arc_analyzer.py report      # Generate markdown report
```

### How Arc Detection Works

This tool uses **semantic boundary detection** with Gemini Flash Lite. For each user message, it asks: "Is this starting a new quest/journey, or continuing the current one?"

This generic approach adapts to any workflow style. The model considers:
- Is this a new command, request, or directive?
- Is this a follow-up, confirmation, or clarification?
- Did the user change direction or return from a break?

### About the Presentation Numbers

The [543 Hours presentation](https://michael.roth.rocks/research/543-hours/) used a **pattern-based** approach tuned to a specific workflow. That original analysis detected arcs using regex patterns like:

```python
# Examples from the original pattern-based detection:
r'burn\s+down.*(?:release|R\d+|tasks)'    # "burn down release R5"
r'spawn.*(?:restricted|foreground).*agent' # "spawn a restricted agent"
r'please.*(?:implement|complete).*T\d+'    # "please implement T103"
```

These patterns were specific to a delegation-heavy workflow with structured task management. The semantic approach in this public tool is more generic but may produce different arc counts for the same data.

**Key difference:** Pattern-based detection found 650 arcs in the original dataset. Semantic detection on the same data finds fewer arcs because it groups more liberally. Both approaches are valid - they measure slightly different things.

### Autonomy Levels (HOW work was done)

| Level | Description |
|-------|-------------|
| `interactive` | No agents spawned, direct human-AI collaboration |
| `quick` | < 15 minutes with agents |
| `build` | 15-60 minutes |
| `feature` | 1-4 hours |
| `release` | 4+ hours of sustained autonomous work |

### Intent Categories (WHAT kind of work)

| Intent | Description |
|--------|-------------|
| `implement` | Building new features, adding functionality |
| `fix` | Bug fixes, error resolution, debugging |
| `refactor` | Code cleanup, restructuring, optimization |
| `test` | Writing or running tests, verification |
| `review` | Code review, design review, checking work |
| `deploy` | Deployment, release, shipping to production |
| `docs` | Documentation, comments, README updates |
| `explore` | Research, investigation, understanding code |
| `config` | Configuration, setup, environment changes |
| `other` | Anything that doesn't fit above categories |

## Chat Analyzer

Clusters prompts using semantic embeddings to find patterns in how you interact with Claude Code.

```bash
python chat_analyzer.py extract    # Parse JSONL into SQLite
python chat_analyzer.py embed      # Generate Gemini embeddings
python chat_analyzer.py cluster    # Run HDBSCAN clustering
python chat_analyzer.py visualize  # Generate UMAP visualization
python chat_analyzer.py stats      # Database statistics
python chat_analyzer.py clusters   # Show sample prompts per cluster
```

### Options

```bash
# Adjust clustering granularity
python chat_analyzer.py cluster --min-cluster-size 30  # Smaller clusters
python chat_analyzer.py cluster --min-cluster-size 100 # Larger clusters

# Embedding batch size
python chat_analyzer.py embed --batch-size 50
```

## Output Files

| File | Description |
|------|-------------|
| `arc_analytics.db` | SQLite database of arcs and agent sessions |
| `chat_analytics.db` | SQLite database of prompts, sessions, tool calls |
| `clusters.html` | Interactive UMAP visualization |

## Requirements

- Python 3.10+
- Gemini API key (for semantic detection and embeddings)
- Claude Code with existing conversation history

## Database Schema

### arc_analytics.db

```sql
CREATE TABLE arcs (
    id TEXT PRIMARY KEY,
    project TEXT,
    autonomy_level TEXT,    -- interactive/quick/build/feature/release (based on duration + agents)
    start_time TEXT,
    end_time TEXT,
    duration_minutes REAL,
    trigger_prompt TEXT,
    agents_spawned INTEGER,
    human_interrupts INTEGER,
    completion_status TEXT,
    intent TEXT,            -- semantic intent from Gemini (implement/fix/refactor/test/review/deploy/docs/explore/config/other)
    session_file TEXT       -- source session file path
);

CREATE TABLE agent_sessions (
    id TEXT PRIMARY KEY,
    project TEXT,
    start_time TEXT,
    end_time TEXT,
    duration_minutes REAL,
    message_count INTEGER,
    tool_calls INTEGER,
    file_path TEXT          -- source agent file path
);
```

### chat_analytics.db

```sql
CREATE TABLE prompts (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    project TEXT,
    content TEXT NOT NULL,
    turn_number INTEGER DEFAULT 1,
    embedding BLOB,         -- 768-dim Gemini embedding
    cluster_id INTEGER
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    project TEXT,
    summary TEXT,
    start_time INTEGER,
    end_time INTEGER,
    message_count INTEGER,
    is_subagent INTEGER
);

CREATE TABLE tool_calls (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    tool_name TEXT,
    timestamp INTEGER
);
```

## Methodology

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for detailed analysis methodology.

## Privacy Note

These tools only read your local Claude Code data. Data is sent to Gemini API for:
- Semantic boundary classification (arc detection)
- Embedding generation (prompt clustering)

The generated databases contain your actual prompts. Do not share them publicly.

## License

MIT License - See [LICENSE](LICENSE)

## Related

- [543 Hours of Autonomous Work](https://michael.roth.rocks/research/543-hours/) - Research presentation using these tools
- [Claude Code](https://github.com/anthropics/claude-code) - Anthropic's CLI for Claude

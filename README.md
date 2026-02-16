# Claude Code Log Analyzer

Python tools for analyzing Claude Code conversation history. Extract patterns, measure autonomous work hours, and understand your AI-assisted development workflow.

This toolkit was developed during a 97-day research project analyzing 543 hours of autonomous AI-assisted work across 165 releases. The methodology and findings are documented in the [accompanying presentation](https://michael.roth.rocks/research/543-hours/).

## What It Does

Three complementary analyses:

1. **Arc Analysis** - Identifies coherent units of work ("arcs") from your prompts. What kind of work are you doing?
2. **Agent Analysis** - Counts autonomous agent sessions and their durations. How much autonomous work happened?
3. **Gate Analysis** - Discovers and analyzes review gate tools in your sessions. How effective are your quality gates?

Tools:
- **arc_analyzer.py** - Detects work arcs using semantic analysis (Gemini Flash Lite), counts agent session hours
- **chat_analyzer.py** - Clusters prompts using Gemini embeddings + HDBSCAN to reveal interaction patterns
- **gate_analyzer.py** - Discovers review gate tools, extracts gate check results, classifies decisions and error types

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

# Analyze review gate effectiveness
python gate_analyzer.py discover          # Find gate tools in your sessions
python gate_analyzer.py extract           # Extract gate check results
python gate_analyzer.py classify          # Classify decisions (regex + Gemini)
python gate_analyzer.py classify-errors   # Classify error types on rejections
python gate_analyzer.py stats             # Show gate statistics
python gate_analyzer.py error-analysis    # Analyze rejection patterns
python gate_analyzer.py report            # Generate markdown report
```

## Data Sources

The scripts read from your local Claude Code data directory (`~/.claude/`):

| File | Content |
|------|---------|
| `history.jsonl` | User prompts with timestamps (used by chat_analyzer) |
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
export CLAUDE_EXTRA_DIRS="/home/other/.claude/projects:/shared/.claude/projects"
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

## Gate Analyzer

Discovers review gate tools in your Claude Code sessions and analyzes their effectiveness. Works with any MCP-based review tool — no configuration required.

```bash
python gate_analyzer.py discover           # Auto-discover gate tools
python gate_analyzer.py extract            # Extract gate check results
python gate_analyzer.py classify           # Classify decisions (regex then Gemini)
python gate_analyzer.py classify-errors    # Classify error types on rejections
python gate_analyzer.py stats              # Show statistics
python gate_analyzer.py error-analysis     # Analyze rejection patterns and recovery rates
python gate_analyzer.py report             # Generate markdown report
python gate_analyzer.py report -o out.md   # Write report to file
```

### How Gate Discovery Works

**Automatic discovery** — the tool finds review gates using Gemini:

1. **Scan** — Collects all `tool_use` names from JSONL files
2. **Classify** — Sends tool names to Gemini Flash Lite (batched), asking which are quality review gates and what type they are
3. **Cache** — Results stored in `gate_tools` table. Reuse on subsequent runs. `--rediscover` to refresh.

Override with `--gates tool1,tool2` to skip discovery and specify gate tools manually.

### Decision Classification

Decisions are classified from gate tool responses:

| Decision | Meaning |
|----------|---------|
| `APPROVED` | Work passed review |
| `NEEDS_REVISION` | Work requires changes |
| `ESCALATE` | Issue requires human attention |
| `UNKNOWN` | Could not determine decision |

Two-pass classification: regex patterns on response text first, then Gemini for remaining unknowns.

### Error Type Classification

Rejections are classified into error types using Gemini:

| Type | Description |
|------|-------------|
| `SYSTEMATIC` | Wrong but internally consistent — misunderstood requirements, wrong architecture |
| `INCOHERENT` | Internally inconsistent — contradicts own plan, random quality variation |
| `OMISSION` | Simply left out — required component, test, or documentation skipped |
| `API_ERROR` | Infrastructure failure, not a real rejection |

### Options

```bash
--data-dir PATH      # Claude data directory (default: ~/.claude)
--gates TOOLS        # Manual gate tool list (comma-separated, skips discovery)
--model MODEL        # Gemini model (default: gemini-flash-lite-latest)
--rediscover         # Re-run gate discovery
--output FILE        # Write report to file (report command only)
```

### Related Research

This tool was built to support the research in [AI Agents Aren't a Hot Mess](https://michael.roth.rocks/research/gate-analysis/) — an analysis of 4,918 cross-model review gate checks testing Anthropic's incoherence hypothesis.

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
| `gate_analytics.db` | SQLite database of gate tools, checks, issues |
| `clusters.html` | Interactive UMAP visualization |

## Requirements

- Python 3.10+
- Gemini API key (for semantic detection and embeddings)
- Claude Code with existing conversation history

## Database Schema

### gate_analytics.db

```sql
CREATE TABLE gate_tools (
    tool_name TEXT PRIMARY KEY,
    gate_type TEXT,           -- review_plan, review_design, review_code, codereview, precommit, validation, qa, audit
    discovery_method TEXT     -- gemini, manual
);

CREATE TABLE gate_checks (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    project TEXT NOT NULL,
    gate_type TEXT,
    tool_name TEXT,
    decision TEXT,            -- APPROVED, NEEDS_REVISION, ESCALATE, UNKNOWN
    feedback_text TEXT,
    feedback_length INTEGER,
    error_class TEXT,         -- SYSTEMATIC, INCOHERENT, OMISSION, API_ERROR
    timestamp TEXT,
    session_file TEXT
);

CREATE TABLE gate_issues (
    id INTEGER PRIMARY KEY,
    gate_check_id INTEGER,
    severity TEXT,
    title TEXT,
    description TEXT
);

CREATE TABLE gate_iterations (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    gate_type TEXT,
    iteration_number INTEGER,
    decision TEXT,
    gate_check_id INTEGER
);
```

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
- [AI Agents Aren't a Hot Mess](https://michael.roth.rocks/research/gate-analysis/) - Gate analysis research (4,918 cross-model review checks)
- [Claude Code](https://github.com/anthropics/claude-code) - Anthropic's CLI for Claude

# Claude Code Log Analyzer

Python tools for analyzing Claude Code conversation history.

## Quick Commands

```bash
source .venv/bin/activate
export GEMINI_API_KEY="your-key"

# Arc analysis (work patterns + autonomous hours)
python arc_analyzer.py agents && python arc_analyzer.py extract && python arc_analyzer.py stats

# Chat analysis (prompt clustering)
python chat_analyzer.py extract && python chat_analyzer.py embed && python chat_analyzer.py cluster

# Gate analysis (review gate effectiveness)
python gate_analyzer.py discover && python gate_analyzer.py extract && python gate_analyzer.py stats
```

## Tools

| Tool | Database | Purpose |
|------|----------|---------|
| `arc_analyzer.py` | `arc_analytics.db` | Work arcs, autonomous hours, intent classification |
| `chat_analyzer.py` | `chat_analytics.db` | Prompt clustering via Gemini embeddings + HDBSCAN |
| `gate_analyzer.py` | `gate_analytics.db` | Review gate discovery, extraction, error classification |

## Key Conventions

- `--data-dir` flag overrides default `~/.claude` directory on all tools
- `--model` flag selects Gemini model for classification in `gate_analyzer.py` (default: `gemini-flash-lite-latest`)
- Each tool owns its own SQLite database in the repo root
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` env var required for semantic features

## Full Documentation

- [README.md](README.md) — complete usage guide with schemas and examples
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — data sources, analysis phases, reproducibility

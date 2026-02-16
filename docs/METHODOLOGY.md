# Methodology

How the Claude Code usage analysis is performed.

## Data Sources

### 1. History File (`~/.claude/history.jsonl`)
- **Content**: User prompts with timestamps
- **Format**: JSONL with `display`, `timestamp`, `project` fields

### 2. Project Session Files (`~/.claude/projects/*/*.jsonl`)
- **Content**: Full conversation transcripts
- **Format**: JSONL with messages, tool calls, responses
- **Note**: Main sessions are long-running terminal instances (weeks to months each)

### 3. Agent Logs (`~/.claude/projects/*/agent-*.jsonl`)
- **Content**: Subagent conversation transcripts
- **Format**: Same as session files
- **Purpose**: Autonomous work analysis

## Analysis Tools

### arc_analyzer.py

Identifies autonomous work "arcs" - periods of work initiated by a single prompt, potentially spanning multiple agent spawns.

```bash
python arc_analyzer.py extract   # Extract arcs from session files
python arc_analyzer.py stats     # Show arc statistics
python arc_analyzer.py list      # List all arcs
python arc_analyzer.py report    # Generate markdown report
```

### chat_analyzer.py

Main analysis script for prompt clustering and session analysis.

```bash
python chat_analyzer.py extract    # Parse JSONL into SQLite
python chat_analyzer.py embed      # Generate Gemini embeddings
python chat_analyzer.py cluster    # Run HDBSCAN clustering
python chat_analyzer.py visualize  # Generate UMAP visualization
```

### gate_analyzer.py

Discovers and analyzes review gate tools in Claude Code sessions. Measures gate effectiveness, classifies decisions, and categorizes error types.

```bash
python gate_analyzer.py discover         # Auto-discover gate tools via Gemini
python gate_analyzer.py extract          # Extract gate checks from session JSONL
python gate_analyzer.py classify         # Classify decisions (regex then Gemini)
python gate_analyzer.py classify-errors  # Classify error types on rejections
python gate_analyzer.py stats            # Show gate usage statistics
python gate_analyzer.py error-analysis   # Analyze rejection patterns and recovery rates
python gate_analyzer.py report           # Generate markdown report
```

**Pipeline:** `discover → extract → classify → classify-errors → stats/report`

**Key design decisions:**
- Zero-config gate discovery: sends all tool names to Gemini Flash Lite for classification
- Generic JSON response parsing: tries common field names (`decision`, `status`, `code_review_status`, `workflow_guidance`), falls back to regex
- Two-pass decision classification: regex patterns first (no API cost), Gemini for remaining unknowns
- Error taxonomy: SYSTEMATIC (wrong approach), INCOHERENT (self-contradictory), OMISSION (incomplete)
- Optional arc correlation: joins with `arc_analytics.db` if present, gracefully skips otherwise

## Technical Stack

### Python Dependencies
```
google-genai       # Gemini API client
hdbscan            # Density-based clustering
scikit-learn       # ML utilities
umap-learn         # Dimensionality reduction
plotly             # Interactive visualization
numpy              # Numerical computing
tqdm               # Progress bars
```

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
```

## Analysis Phases

### Phase 1: Data Extraction
1. Parse all JSONL files from Claude Code directories
2. Extract prompts with timestamps and project context
3. Identify main vs subagent sessions
4. Extract tool calls for usage analysis

### Phase 2: Embedding Generation
1. Use Gemini `gemini-embedding-001` model
2. Task type: `CLASSIFICATION` (optimized for clustering)
3. 768-dimensional vectors
4. Batch processing for efficiency

### Phase 3: Clustering
1. HDBSCAN with configurable min_cluster_size
2. No forced cluster count (discovers natural groupings)
3. Noise points (-1) represent varied/unique prompts

### Phase 4: Analysis
1. Manual cluster labeling via sampling
2. Pattern identification in clustered prompts
3. "Noise" analysis to understand adaptive work
4. Temporal analysis for workflow evolution
5. Tool call analysis for capability usage

### Phase 5: Synthesis
1. Cross-reference patterns with timeline
2. Identify milestone events
3. Correlate tool adoption with workflow changes
4. Document findings in structured markdown

## Reproducibility

### Re-running Analysis
```bash
source .venv/bin/activate
export GEMINI_API_KEY="your-key"

# Full pipeline
python chat_analyzer.py extract
python chat_analyzer.py embed
python chat_analyzer.py cluster
python chat_analyzer.py visualize

# View results
python chat_analyzer.py stats
python chat_analyzer.py clusters --samples 5
```

### Modifying Clustering
```bash
# Smaller clusters (more granular)
python chat_analyzer.py cluster --min-cluster-size 20

# Larger clusters (more general)
python chat_analyzer.py cluster --min-cluster-size 100
```

## Limitations

1. **Timestamp granularity**: Some session timestamps may be imprecise
2. **Subagent task identification**: Many agents may have "Unknown" tasks (summary not captured)
3. **Tool call attribution**: Some tool calls lack session context
4. **Project boundaries**: Cross-project work not fully captured
5. **Embedding model choice**: Results may vary with different embedding models

## Gate Analysis Phases

### Phase 1: Gate Discovery
1. Scan all JSONL files for unique `tool_use` names
2. Send tool names to Gemini Flash Lite (batched in groups of 200), asking which are quality review gates and what type they are
3. Cache discovered gates in `gate_tools` table with type and discovery method

### Phase 2: Extraction
1. For each JSONL file, find `tool_use` blocks matching discovered gate tools
2. Match each `tool_use` to its `tool_result` via `tool_use_id`
3. Parse JSON response trying common decision fields, fall back to regex on text
4. Extract structured issues from feedback when available
5. Compute iteration sequences (consecutive checks of same gate type in same session)

### Phase 3: Classification
1. **Regex pass** — Pattern-match decision keywords in feedback text (e.g., "approved", "needs revision", "rejected")
2. **Gemini pass** — For remaining UNKNOWN decisions, send feedback text to Gemini for semantic classification
3. **Error classification** — For NEEDS_REVISION decisions, classify error type as SYSTEMATIC, INCOHERENT, OMISSION, or API_ERROR

### Phase 4: Analysis
1. Compute approval rates, rejection rates, error type distributions
2. Calculate recovery rates (what happens after rejection)
3. Optionally correlate with arc data for complexity analysis
4. Generate summary statistics and markdown reports

## Future Work

1. Improve subagent task extraction from conversation content
2. Add time-series visualization of pattern evolution
3. Build interactive dashboard for exploration
4. Correlate patterns with outcome metrics (success/failure)
5. Semantic similarity search for pattern discovery

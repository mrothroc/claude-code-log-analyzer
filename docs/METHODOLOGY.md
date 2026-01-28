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

## Future Work

1. Improve subagent task extraction from conversation content
2. Add time-series visualization of pattern evolution
3. Build interactive dashboard for exploration
4. Correlate patterns with outcome metrics (success/failure)
5. Semantic similarity search for pattern discovery

# Industrial Knowledge Intelligence — Hybrid GraphRAG

A hybrid GraphRAG system (knowledge graph + vector search, fused via RRF) that answers questions about industrial pump maintenance by tracing relationships across documents that share no keywords with each other. Built for **ET AI Hackathon 2.0, Problem #8: Industrial Knowledge Intelligence**.

**[Full write-up](Industrial_Knowledge_Intelligence_Detailed_Document.pdf)** · **[Architecture diagram](architecture_diagram.png)**

**Presentation deck:** [Industrial_Knowledge_Intelligence_Deck.pptx](./Industrial_Knowledge_Intelligence_Deck.pptx)
**Demo video:** [demo_video.mp4](./demo_video.mp4)

## Quick start

```bash
pip install -r requirements.txt

# System dependency: Tesseract OCR (only needed when (re)building the vector
# store, since 3 corpus PDFs are scanned images):
#   macOS: brew install tesseract | Debian/Ubuntu: apt install tesseract-ocr
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki

# macOS/Linux
export ANTHROPIC_API_KEY="your-key"
# Windows PowerShell
$env:ANTHROPIC_API_KEY="your-key"

streamlit run app.py
```

The first run downloads the `BAAI/bge-small-en-v1.5` embedding model (~130MB) and builds the vector store/knowledge graph from the corpus if they don't already exist under `data/` (no API key needed for the build itself).

Run the test suite with `python -m pytest tests/` — it validates the eval dataset, graph invariants, chunking, citation filtering, and ranking determinism. CI runs it on every push (`.github/workflows/tests.yml`).

Or run it in Docker (embedding model and indexes baked into the image at build time):

```bash
docker build -t ikig .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY="your-key" ikig
```

Operational notes: answers are cached in `.synthesis_cache/` (repeat questions skip the API and show a ⚡ badge); every query appends a JSON trace line to `logs/query_traces.jsonl` (latency split, token usage, rankings, citations); model names and top-k are overridable via `SYNTHESIS_MODEL`, `EXTRACTION_MODEL`, and `RETRIEVAL_TOP_K` env vars. Model citations are validated against the retrieved context before rendering — a hallucinated doc reference can never display as a source. Each answer includes a "Why these documents" expander showing the knowledge-graph paths (works offline; all JS inlined).

Retrieval robustness: entity matching is alias- and punctuation-tolerant ("p204" finds P-204), scanned/clean duplicate documents collapse to one top-k slot, and follow-up questions reuse the conversation so "was it acted upon?" resolves against the previous answer. `evaluate.py` reports recall at both K=5 and K=12 — on a 17-document corpus the @5 figure is the meaningful one.

## Architecture

Seven stages, each a standalone module in `ingest/`:

1. **Document loader** (`loaders.py`) — PyMuPDF text extraction, with pytesseract OCR fallback for scanned pages
2. **Entity/relationship extractor** (`extractor.py`) — Claude API extracts typed entities and relationships per document, cached to disk
3. **Knowledge graph builder** (`graph_builder.py`) — assembles a NetworkX graph across the whole corpus, merging entities by canonical ID (e.g. "Pump P-204" and "P-204" resolve to one node)
4. **Vector store builder** (`vector_builder.py`) — embeds each page into a persistent ChromaDB collection
5. **RRF fusion retriever** (`retriever.py`) — combines semantic search with graph traversal via Reciprocal Rank Fusion
6. **Synthesis agent** (`synthesis.py`) — Claude API turns retrieved documents into an answer with citations and a confidence rating
7. **Streamlit chat UI** (`app.py`) — the user-facing entry point

`evaluate.py` is an eighth piece, sitting outside the numbered pipeline: a benchmark harness scoring retrieval quality against hand-labeled ground truth.

## The star demo question

> "Was there any early warning before the Pump P-204 failure, and what should have been done according to procedure?"

This is the flagship test of the whole hybrid approach. The answer requires connecting five documents that share no common keywords: an inspection report flags elevated bearing vibration three weeks before failure; a maintenance log defers the recommended bearing replacement due to a parts shortage; a vibration trend log shows the alarm threshold crossed five days out; an incident report records the resulting failure; and a procedure document specifies what should have happened instead. Pure keyword or vector search returns only the incident report — the graph traversal is what recovers the earlier warning chain.

## Benchmark results

Measured 17 Jul 2026 after the chunking/embedding upgrade (windowed page chunks + `bge-small-en-v1.5`), verified byte-identical across three consecutive runs:

- **Entity extraction Macro-F1: 0.8411** (excluding DATE, which has zero labeled instances in the ground truth set)
- **Star demo question (Q01) recall@12: 1.00 hybrid vs 0.80 vector-only** — the full five-document chain is retrieved; the two documents the vector baseline misses (VS-204 and M-118, whose text never mentions "P-204") are supplied by graph traversal
- **Overall recall@12 across 8 questions: 0.83 hybrid vs 0.81 vector-only**; **recall@5: 0.70 hybrid vs 0.76 vector-only**
- All control (keyword-answerable) questions: 1.00 in both modes

Honest reading: upgrading the embedding model lifted the *vector baseline* substantially (0.69 → 0.81 @12), so the hybrid's aggregate margin is now small — and at the stricter @5 cutoff, hop-0 graph noise costs a little precision on single-document questions. The graph earns its place on the multi-hop chain question this system was built for (Q01: 1.00 vs 0.80), not on aggregate averages. Q02 remains capped by hop-0 tie-flooding (root-cause trace in `CLAUDE.md`).

## Repo structure

```
ingest/          the 7-stage pipeline (see Architecture above)
data/corpus/     the locked document corpus (synthetic, real OEM manuals, scanned)
data/eval/       benchmark questions, ground truth, results, FailureSensorIQ reference data
app.py           Streamlit UI - the live app
evaluate.py      benchmark harness
scripts/         one-off scripts that built the locked dataset - not part of running the app
                 (generate_corpus.py, make_scanned.py, download_failuresensoriq.py)
```

## Dataset

The corpus is a locked mix of a synthetic P-204 pump near-miss chain (11 PDFs, planted for the star demo), 3 real OEM pump manuals, and 3 degraded scans for OCR testing, plus IBM's FailureSensorIQ dataset (CC-BY-4.0) as reference evaluation data. Full provenance, sources, and structure: [`data/DATASET_README.md`](data/DATASET_README.md). FailureSensorIQ-specific reference material (citation, dataset subsets, benchmark details): [`data/eval/FAILURESENSORIQ_NOTES.md`](data/eval/FAILURESENSORIQ_NOTES.md).

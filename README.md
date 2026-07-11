# Industrial Knowledge Intelligence — Hybrid GraphRAG

A hybrid GraphRAG system (knowledge graph + vector search, fused via RRF) that answers questions about industrial pump maintenance by tracing relationships across documents that share no keywords with each other. Built for **ET AI Hackathon 2.0, Problem #8: Industrial Knowledge Intelligence**.

## Quick start

```bash
pip install -r requirements.txt

# macOS/Linux
export ANTHROPIC_API_KEY="your-key"
# Windows PowerShell
$env:ANTHROPIC_API_KEY="your-key"

streamlit run app.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90MB) and builds the vector store/knowledge graph from the corpus if they don't already exist under `data/`.

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

- **Entity extraction Macro-F1: 0.8411** (excluding DATE, which has zero labeled instances in the ground truth set)
- **Star demo question (Q01) recall: 0.60**
- **Overall hybrid retrieval recall across 8 benchmark questions: 0.78**

Known, understood limitation: two questions (Q02, and partially Q01) are capped by "hop-0 tie-flooding" — many documents share an identical graph proximity score for merely mentioning the query's main entity, which can crowd out a more specific but structurally-further answer out of the top-K results. Full root-cause traces are in `CLAUDE.md`.

## Repo structure

```
ingest/          the 7-stage pipeline (see Architecture above)
data/corpus/     the locked document corpus (synthetic, real OEM manuals, scanned)
data/eval/       benchmark questions, ground truth, results, FailureSensorIQ reference data
app.py           Streamlit UI - the live app
evaluate.py      benchmark harness
generate_corpus.py, make_scanned.py, download_failuresensoriq.py
                 one-off scripts that built the locked dataset - not part of running the app
```

## Dataset

The corpus is a locked mix of a synthetic P-204 pump near-miss chain (11 PDFs, planted for the star demo), 3 real OEM pump manuals, and 3 degraded scans for OCR testing, plus IBM's FailureSensorIQ dataset (CC-BY-4.0) as reference evaluation data. Full provenance, sources, and structure: [`data/DATASET_README.md`](data/DATASET_README.md). FailureSensorIQ-specific reference material (citation, dataset subsets, benchmark details): [`data/eval/FAILURESENSORIQ_NOTES.md`](data/eval/FAILURESENSORIQ_NOTES.md).

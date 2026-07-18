"""
Lightweight observability for the Hybrid GraphRAG pipeline.

Every synthesized query appends one JSON line to logs/query_traces.jsonl:
timestamp, latency split (retrieval vs LLM), matched entities, retrieved
doc rankings, token usage, validated citations, confidence, and cache
status. No external tracing stack - a flat, greppable file that answers
"why did the system say that?" after the fact.

Usage:
    from ingest.tracing import trace_event
    trace_event({"event": "synthesize", "query": ..., "llm_ms": ...})
"""
import json
import os
import uuid
from datetime import datetime, timezone

TRACE_PATH = os.path.join("logs", "query_traces.jsonl")


def trace_event(record: dict, path: str = TRACE_PATH) -> None:
    record = {
        "trace_id": uuid.uuid4().hex[:12],
        "ts": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # tracing must never take down the answer path

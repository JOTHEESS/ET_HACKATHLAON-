"""
Synthesis agent for the Hybrid GraphRAG pipeline.

Takes a natural-language question, retrieves the most relevant documents
via ingest.retriever's RRF fusion, and asks Claude to synthesize an
answer that cites the specific source documents it drew from, with a
confidence rating.
"""
import hashlib
import json
import os
import re
import time

from anthropic import Anthropic

from ingest.graph_builder import load_or_build_graph
from ingest.retriever import match_entities, retrieve
from ingest.tracing import trace_event
from ingest.vector_builder import load_vector_store, query_store

ANSWER_CACHE_DIR = ".synthesis_cache"
SYNTHESIS_MODEL = os.environ.get("SYNTHESIS_MODEL", "claude-sonnet-4-5")
DEFAULT_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "12"))

SYNTHESIS_SYSTEM_PROMPT = """You are an industrial maintenance knowledge assistant. \
Answer the user's question using ONLY the provided source documents below - do not use \
outside knowledge about pumps, procedures, or regulations.

Respond with ONLY valid JSON, no markdown fences, no preamble, in this exact shape:
{
  "answer": "...",
  "citations": ["doc_id1", "doc_id2", ...],
  "confidence": "high" | "medium" | "low"
}

Rules:
- "citations" must list only doc_ids that were actually used to support the answer.
- Use "confidence": "low" if the source documents don't clearly answer the question.
- Cross-reference facts across documents when the question requires it (e.g. an equipment
  ID mentioned in one document that is governed by a procedure described in another).
"""


def _gather_context(collection, query: str, doc_ids: list,
                     n_vector_hits: int = 20, max_chars_per_doc: int = 4000) -> str:
    """Pull the most relevant chunk per doc_id. Docs that surfaced via vector
    search get their best-matching chunk; docs found only through graph
    traversal (no vector hit) fall back to their first page."""
    best_chunk_by_doc = {}
    for hit in query_store(collection, query, n_results=n_vector_hits):
        doc_id = hit["metadata"]["doc_id"]
        if doc_id not in best_chunk_by_doc:
            best_chunk_by_doc[doc_id] = hit["text"]

    sections = []
    for doc_id in doc_ids:
        if doc_id in best_chunk_by_doc:
            text = best_chunk_by_doc[doc_id]
        else:
            # Doc reached only via graph traversal: ask the vector store for
            # this doc's best chunk FOR THIS QUERY, instead of blindly taking
            # page 1 (wrong for multi-page manuals).
            hits = collection.query(query_texts=[query], n_results=1,
                                     where={"doc_id": doc_id})
            if hits["documents"] and hits["documents"][0]:
                text = hits["documents"][0][0]
            else:
                result = collection.get(where={"doc_id": doc_id})
                pairs = sorted(zip(result["metadatas"], result["documents"]),
                                key=lambda x: (x[0]["page_num"], x[0].get("window", 0)))
                text = pairs[0][1] if pairs else ""

        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "...[truncated]"
        sections.append(f"=== Document: {doc_id} ===\n{text}")

    return "\n\n".join(sections)


def _answer_cache_path(query: str, model: str, top_k: int, history: list) -> str:
    basis = json.dumps([model, top_k, query.strip().lower(),
                        [h.get("content", "")[:200] for h in history]])
    h = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    return os.path.join(ANSWER_CACHE_DIR, f"{h}.json")


def _parse_json_response(raw: str):
    """LLM output -> dict, tolerating markdown fences and stray prose
    around the JSON. Returns None only if no parseable object exists."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _load_answer_cache(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_answer_cache(path: str, result: dict) -> None:
    os.makedirs(ANSWER_CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def _validate_citations(citations: list, retrieved_doc_ids: list) -> list:
    """Keep only citations that name a document actually given to the model as
    context - a hallucinated doc_id must never render as a legitimate source.
    The model sometimes cites the short form ("IR-556") instead of the full
    filename it was shown; resolve those by prefix match."""
    valid = []
    for cite in citations:
        if cite in retrieved_doc_ids:
            match = cite
        else:
            match = next((d for d in retrieved_doc_ids
                          if d.lower().startswith(str(cite).lower())), None)
        if match and match not in valid:
            valid.append(match)
    return valid


def synthesize(query: str, client: Anthropic = None, collection=None, graph=None,
               model: str = SYNTHESIS_MODEL, top_k: int = DEFAULT_TOP_K,
               use_cache: bool = True, history: list = None) -> dict:
    """history: prior chat turns as [{"role", "content"}, ...] - used to
    resolve follow-ups ("was it acted on?") that carry no entity IDs."""
    history = history or []
    cache_path = _answer_cache_path(query, model, top_k, history)
    if use_cache:
        cached = _load_answer_cache(cache_path)
        if cached is not None:
            cached["cached"] = True
            trace_event({"event": "synthesize", "cache": "hit", "query": query,
                         "model": model, "top_k": top_k,
                         "confidence": cached.get("confidence"),
                         "citations": cached.get("citations", [])})
            return cached

    if client is None:
        # Explicit timeout + retries so a flaky network during a live demo
        # degrades to a caught exception, not an indefinite spinner.
        client = Anthropic(timeout=60.0, max_retries=3)
    if collection is None:
        collection = load_vector_store()
    if graph is None:
        graph = load_or_build_graph()

    # A follow-up question often has no entity mentions of its own - borrow
    # the previous user turn so graph traversal still gets its seeds.
    retrieval_query = query
    if history and not match_entities(graph, query):
        prev_user = [h["content"] for h in history if h.get("role") == "user"]
        if prev_user:
            retrieval_query = f"{prev_user[-1]} {query}"

    t0 = time.perf_counter()
    retrieval = retrieve(retrieval_query, collection=collection, graph=graph, top_k=top_k)
    context = _gather_context(collection, retrieval_query, retrieval["fused_doc_ids"])
    retrieval_ms = round((time.perf_counter() - t0) * 1000)

    history_block = ""
    if history:
        lines = [f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-6:]]
        history_block = ("Conversation so far (context for resolving references "
                         "like 'it' or 'the alarm' - not a source of facts):\n"
                         + "\n".join(lines) + "\n\n")

    t1 = time.perf_counter()
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"{history_block}Question: {query}\n\nSource documents:\n{context}"
        }]
    )
    llm_ms = round((time.perf_counter() - t1) * 1000)

    result = _parse_json_response(message.content[0].text)
    if result is None or not isinstance(result, dict):
        result = {"answer": message.content[0].text.strip(),
                  "citations": [], "confidence": "low"}

    result.setdefault("citations", [])
    result.setdefault("confidence", "low")
    result["citations"] = _validate_citations(result["citations"], retrieval["fused_doc_ids"])
    result["retrieved_doc_ids"] = retrieval["fused_doc_ids"]
    result["matched_entities"] = retrieval["matched_entities"]
    result["cached"] = False

    trace_event({
        "event": "synthesize", "cache": "miss", "query": query,
        "model": model, "top_k": top_k,
        "retrieval_ms": retrieval_ms, "llm_ms": llm_ms,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "matched_entities": retrieval["matched_entities"],
        "fused_doc_ids": retrieval["fused_doc_ids"],
        "confidence": result["confidence"],
        "citations": result["citations"],
    })
    if use_cache:
        _save_answer_cache(cache_path, result)
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    query = sys.argv[1] if len(sys.argv) > 1 else (
        "Was there any early warning before the Pump P-204 failure, "
        "and what should have been done according to procedure?"
    )

    print(f"Query: {query}\n")
    result = synthesize(query)
    print(json.dumps(result, indent=2))

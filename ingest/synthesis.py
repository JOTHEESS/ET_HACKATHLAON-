"""
Synthesis agent for the Hybrid GraphRAG pipeline.

Takes a natural-language question, retrieves the most relevant documents
via ingest.retriever's RRF fusion, and asks Claude to synthesize an
answer that cites the specific source documents it drew from, with a
confidence rating.
"""
import json
import os

from anthropic import Anthropic

from ingest.retriever import retrieve
from ingest.vector_builder import load_vector_store, query_store

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
            result = collection.get(where={"doc_id": doc_id})
            pairs = sorted(zip(result["metadatas"], result["documents"]),
                            key=lambda x: x[0]["page_num"])
            text = pairs[0][1] if pairs else ""

        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "...[truncated]"
        sections.append(f"=== Document: {doc_id} ===\n{text}")

    return "\n\n".join(sections)


def synthesize(query: str, client: Anthropic = None, collection=None, graph=None,
               model: str = "claude-sonnet-4-5", top_k: int = 12) -> dict:
    if client is None:
        client = Anthropic()
    if collection is None:
        collection = load_vector_store()

    retrieval = retrieve(query, collection=collection, graph=graph, top_k=top_k)
    context = _gather_context(collection, query, retrieval["fused_doc_ids"])

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Question: {query}\n\nSource documents:\n{context}"
        }]
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"answer": raw, "citations": [], "confidence": "low"}

    result.setdefault("citations", [])
    result.setdefault("confidence", "low")
    result["retrieved_doc_ids"] = retrieval["fused_doc_ids"]
    result["matched_entities"] = retrieval["matched_entities"]
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

"""
Entity & relationship extraction for the Hybrid GraphRAG pipeline.

Takes raw page text (from ingest.loaders) grouped by document, sends it to
Claude with a strict JSON schema, and returns structured entities +
relationships ready for the knowledge graph builder.

Entity types match the hand-labeled ground truth schema so extraction
accuracy (Macro-F1) can be scored directly against data/eval/ground_truth_entities.json.
"""
import json
import os
import hashlib
from anthropic import Anthropic

CACHE_DIR = ".extraction_cache"

EXTRACTION_SYSTEM_PROMPT = """You are an industrial document entity extraction system. \
Extract entities and relationships from the given document text with high precision.

ENTITY TYPES (use exactly these labels):
- EQUIPMENT: equipment tags, parts, components (e.g. "Pump P-204", "DE bearing", "SKF 6316")
- PROCEDURE: procedure IDs, clauses (e.g. "M-118", "clause 4.3")
- REGULATION: standards, regulatory codes (e.g. "OISD-132", "ISO 10816-3")
- PERSON: named individuals (e.g. "R. Krishnan")
- DOCUMENT_REF: document/report reference IDs (e.g. "IR-556", "ML-1183")
- MEASUREMENT: quantities with units (e.g. "4.6 mm/s RMS", "62 C", "14 days")
- DATE: dates mentioned
- LOCATION: physical locations (e.g. "Unit 2, Pump House Bay 4")
- STATUS: status/classification labels (e.g. "ELEVATED - MONITOR")
- EVENT: named events (e.g. "catastrophic DE bearing seizure")
- SENSOR: sensor/instrument tags (e.g. "VT-204-DE")

RELATIONSHIP TYPES (use exactly these labels where applicable, or a short
snake_case verb phrase if none fit):
- flagged_by, serviced_under, governed_by, deferred_in, drives,
  escalated_to, resulted_in, references, part_of, inspected_by

Respond with ONLY valid JSON, no markdown fences, no preamble, in this exact shape:
{
  "entities": [{"text": "...", "type": "EQUIPMENT"}, ...],
  "relationships": [{"source": "...", "relation": "governed_by", "target": "..."}, ...]
}

Rules:
- Extract entities exactly as they appear in the text (preserve original casing/spacing).
- Do not invent entities or relationships not supported by the text.
- Only include relationships where both source and target are entities you extracted.
- If a document references another document ID (e.g. "per IR-556"), extract that ID as
  a DOCUMENT_REF entity and add a "references" relationship.
"""


def _cache_key(doc_id: str, text: str) -> str:
    h = hashlib.sha256((doc_id + text).encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}__{h}.json"


def _load_cache(key: str):
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_cache(key: str, data: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, key), "w") as f:
        json.dump(data, f, indent=2)


def extract_document(doc_id: str, full_text: str, client: Anthropic = None,
                      model: str = "claude-sonnet-4-5", use_cache: bool = True) -> dict:
    key = _cache_key(doc_id, full_text)
    if use_cache:
        cached = _load_cache(key)
        if cached is not None:
            return cached

    if client is None:
        client = Anthropic()

    if not full_text.strip():
        result = {"entities": [], "relationships": []}
        if use_cache:
            _save_cache(key, result)
        return result

    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Document ID: {doc_id}\n\nDocument text:\n{full_text}"
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
        print(f"  [WARN] Failed to parse extraction JSON for {doc_id}, returning empty result")
        result = {"entities": [], "relationships": []}

    result.setdefault("entities", [])
    result.setdefault("relationships", [])

    if use_cache:
        _save_cache(key, result)
    return result


def extract_corpus(page_records: list, client: Anthropic = None,
                    model: str = "claude-sonnet-4-5") -> dict:
    by_doc = {}
    for rec in page_records:
        by_doc.setdefault(rec["doc_id"], []).append(rec)

    results = {}
    for doc_id, pages in sorted(by_doc.items()):
        pages_sorted = sorted(pages, key=lambda r: r["page_num"])
        full_text = "\n\n".join(p["text"] for p in pages_sorted)
        extraction = extract_document(doc_id, full_text, client=client, model=model)
        extraction["pages"] = len(pages_sorted)
        results[doc_id] = extraction
        n_e, n_r = len(extraction["entities"]), len(extraction["relationships"])
        print(f"  - {doc_id}: {n_e} entities, {n_r} relationships")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.loaders import load_corpus

    root = sys.argv[1] if len(sys.argv) > 1 else "data/corpus"
    print(f"Loading corpus from {root}...")
    pages = load_corpus(root)
    print(f"Loaded {len(pages)} pages. Extracting entities/relationships...")
    results = extract_corpus(pages)

    out_path = "data/extraction_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved extraction results -> {out_path}")

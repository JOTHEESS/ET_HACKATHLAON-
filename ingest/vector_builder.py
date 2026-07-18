"""
Vector store builder for the Hybrid GraphRAG pipeline.

Embeds the corpus (via sentence-transformers) into a persistent ChromaDB
collection, so the RRF fusion retriever can combine semantic search with
the knowledge graph traversal. Pages are the outer unit (page_num survives
into metadata for citations), but pages longer than the embedding model's
token window are split into overlapping word windows - previously the
whole page was embedded as one chunk and silently truncated at the model
limit, leaving most of each long OEM-manual page invisible to search.

Usage:
    from ingest.vector_builder import build_vector_store, query_store
    collection = build_vector_store(page_records)
    results = query_store(collection, "vibration alarm threshold")
"""
import os

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "et_corpus"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 512-token window (2x MiniLM)

# ~512 bge tokens with headroom for wordpiece expansion; ~15% overlap so a
# fact straddling a window boundary is fully inside at least one chunk.
WORDS_PER_CHUNK = 380
OVERLAP_WORDS = 57


def _split_page(text: str) -> list:
    """One chunk for a short page; overlapping word windows for a long one."""
    words = text.split()
    if len(words) <= WORDS_PER_CHUNK:
        return [text]
    chunks, start = [], 0
    step = WORDS_PER_CHUNK - OVERLAP_WORDS
    while start < len(words):
        chunks.append(" ".join(words[start:start + WORDS_PER_CHUNK]))
        if start + WORDS_PER_CHUNK >= len(words):
            break
        start += step
    return chunks


def _chunk_id(doc_id: str, page_num: int, window: int) -> str:
    return f"{doc_id}::p{page_num}.{window}"


def build_vector_store(page_records: list, persist_dir: str = CHROMA_DIR,
                        reset: bool = True) -> chromadb.api.models.Collection.Collection:
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn,
        metadata={"embedding_model": EMBEDDING_MODEL, "words_per_chunk": WORDS_PER_CHUNK})

    ids, documents, metadatas = [], [], []
    for rec in page_records:
        text = rec["text"].strip()
        if not text:
            continue
        for window, chunk in enumerate(_split_page(text)):
            ids.append(_chunk_id(rec["doc_id"], rec["page_num"], window))
            documents.append(chunk)
            metadatas.append({
                "doc_id": rec["doc_id"],
                "page_num": rec["page_num"],
                "window": window,
                "source_folder": rec["source_folder"],
                "is_ocr": rec["is_ocr"],
                "num_tables": len(rec["tables"]),
            })

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return collection


def load_vector_store(persist_dir: str = CHROMA_DIR) -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def load_or_build_vector_store(persist_dir: str = CHROMA_DIR,
                                corpus_root: str = "data/corpus") -> chromadb.api.models.Collection.Collection:
    """data/chroma_db is gitignored, so a fresh clone has no collection - build
    it from the corpus on first run instead of crashing. No API key needed:
    only PDF text extraction + local sentence-transformers embedding. A store
    built with a different embedding model/chunking config is stale - rebuild
    it too, or old and new vectors would silently mix."""
    try:
        collection = load_vector_store(persist_dir)
        meta = collection.metadata or {}
        if meta.get("embedding_model") == EMBEDDING_MODEL \
                and meta.get("words_per_chunk") == WORDS_PER_CHUNK:
            return collection
    except Exception:  # chromadb's missing-collection error type varies by version
        pass
    from ingest.loaders import load_corpus
    return build_vector_store(load_corpus(corpus_root), persist_dir=persist_dir)


def query_store(collection, query_text: str, n_results: int = 5) -> list:
    results = collection.query(query_texts=[query_text], n_results=n_results)
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.loaders import load_corpus

    print("Loading corpus from data/corpus...")
    pages = load_corpus("data/corpus")
    print(f"Loaded {len(pages)} pages. Building vector store (embedding model: {EMBEDDING_MODEL})...")
    collection = build_vector_store(pages)
    print(f"Vector store built -> {CHROMA_DIR}  ({collection.count()} chunks)")

    sample_query = "vibration alarm threshold for Pump P-204's DE bearing"
    print(f"\nSample query: {sample_query!r}")
    for hit in query_store(collection, sample_query, n_results=3):
        print(f"  - {hit['id']}  (distance={hit['distance']:.4f})")
        print(f"    {hit['text'][:150].replace(chr(10), ' ')}...")

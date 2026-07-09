"""
Vector store builder for the Hybrid GraphRAG pipeline.

Embeds each page of the corpus (via sentence-transformers) into a
persistent ChromaDB collection, so the RRF fusion retriever can combine
semantic search with the knowledge graph traversal. One chunk per page
keeps this simple - the synthetic/real docs are short enough per page
that finer-grained splitting isn't needed.

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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _chunk_id(doc_id: str, page_num: int) -> str:
    return f"{doc_id}::p{page_num}"


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

    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    ids, documents, metadatas = [], [], []
    for rec in page_records:
        text = rec["text"].strip()
        if not text:
            continue
        ids.append(_chunk_id(rec["doc_id"], rec["page_num"]))
        documents.append(text)
        metadatas.append({
            "doc_id": rec["doc_id"],
            "page_num": rec["page_num"],
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

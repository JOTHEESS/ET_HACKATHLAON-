"""
Benchmark harness for the Hybrid GraphRAG pipeline.

Scores vector-only retrieval against the RRF hybrid retriever on all 8
ground-truth benchmark questions, to prove (or disprove) the project's
core claim: graph traversal recovers documents that semantic search
alone misses. Vector search is the fairer baseline here (stronger than
literal keyword matching), so any hybrid improvement is a conservative
lower bound on the graph's real contribution.

Caveat: benchmark_questions.json's "hops" field is a graph traversal
path, not strictly a list of expected documents - some hops are entity/
regulation IDs with no corresponding document (e.g. Q02's hops include
"P-204" and "OISD-132", neither of which is a filename in data/corpus/;
only "M-118" is). Recall is scored by prefix-matching each hop against
doc_ids, so questions with non-document hops have a recall ceiling
below 1.0 even for perfect retrieval - read per-question scores next to
their hops list, not as raw pass/fail.

Usage:
    python evaluate.py
"""
import json
import os
from collections import defaultdict

from ingest.graph_builder import load_graph
from ingest.retriever import retrieve, vector_doc_ranking
from ingest.vector_builder import load_vector_store

BENCHMARK_PATH = "data/eval/benchmark_questions.json"
RESULTS_PATH = "data/eval/benchmark_results.json"
TOP_K = 12


def _hop_found(hop: str, doc_ids: list) -> bool:
    """hops in benchmark_questions.json are short refs (e.g. "IR-556") that
    prefix the actual doc_id (e.g. "IR-556_inspection_report.pdf")."""
    hop_lower = hop.lower()
    return any(doc_id.lower().startswith(hop_lower) for doc_id in doc_ids)


def _recall(hops: list, doc_ids: list) -> float:
    if not hops:
        return 1.0
    found = sum(1 for hop in hops if _hop_found(hop, doc_ids))
    return found / len(hops)


def run_benchmark(top_k: int = TOP_K) -> dict:
    collection = load_vector_store()
    graph = load_graph()

    with open(BENCHMARK_PATH, "r") as f:
        benchmark = json.load(f)

    per_question = []
    for q in benchmark["questions"]:
        # n_results=20 matches what retrieve() feeds the hybrid's vector arm -
        # giving the baseline fewer chunks than the hybrid gets would
        # artificially inflate the hybrid's margin.
        vector_ranking = vector_doc_ranking(collection, q["question"], n_results=20)[:top_k]
        hybrid_ranking = retrieve(q["question"], collection=collection, graph=graph,
                                   top_k=top_k)["fused_doc_ids"]

        per_question.append({
            "id": q["id"],
            "retrieval_type": q["retrieval_type"],
            "question": q["question"],
            "hops": q["hops"],
            "vector_recall": _recall(q["hops"], vector_ranking),
            "hybrid_recall": _recall(q["hops"], hybrid_ranking),
            # recall@5 is the stricter, more meaningful cut on a 17-doc
            # corpus - recall@12 leaves little room to differ from chance.
            "vector_recall_at5": _recall(q["hops"], vector_ranking[:5]),
            "hybrid_recall_at5": _recall(q["hops"], hybrid_ranking[:5]),
            "vector_ranking": vector_ranking,
            "hybrid_ranking": hybrid_ranking,
        })

    by_type = defaultdict(lambda: {"vector": [], "hybrid": []})
    for pq in per_question:
        by_type[pq["retrieval_type"]]["vector"].append(pq["vector_recall"])
        by_type[pq["retrieval_type"]]["hybrid"].append(pq["hybrid_recall"])

    by_retrieval_type = {
        rtype: {
            "n": len(scores["vector"]),
            "vector_avg_recall": sum(scores["vector"]) / len(scores["vector"]),
            "hybrid_avg_recall": sum(scores["hybrid"]) / len(scores["hybrid"]),
        }
        for rtype, scores in by_type.items()
    }

    n = len(per_question)
    overall = {
        "vector_avg_recall": sum(pq["vector_recall"] for pq in per_question) / n,
        "hybrid_avg_recall": sum(pq["hybrid_recall"] for pq in per_question) / n,
        "vector_avg_recall_at5": sum(pq["vector_recall_at5"] for pq in per_question) / n,
        "hybrid_avg_recall_at5": sum(pq["hybrid_recall_at5"] for pq in per_question) / n,
    }

    return {"top_k": top_k, "per_question": per_question,
            "by_retrieval_type": by_retrieval_type, "overall": overall}


def print_report(results: dict) -> None:
    k = results["top_k"]
    print(f"Recall - fraction of a question's required documents found in the top-K ranking (K=5 and K={k})\n")

    print(f"{'ID':<10} {'type':<18} {'vec@5':>7} {'hyb@5':>7} {'vec@' + str(k):>7} {'hyb@' + str(k):>7}")
    for pq in results["per_question"]:
        print(f"{pq['id']:<10} {pq['retrieval_type']:<18} "
              f"{pq['vector_recall_at5']:>7.2f} {pq['hybrid_recall_at5']:>7.2f} "
              f"{pq['vector_recall']:>7.2f} {pq['hybrid_recall']:>7.2f}")

    print(f"\n{'retrieval_type':<18} {'n':>3} {'vector avg':>12} {'hybrid avg':>12}")
    for rtype, s in results["by_retrieval_type"].items():
        print(f"{rtype:<18} {s['n']:>3} {s['vector_avg_recall']:>12.2f} {s['hybrid_avg_recall']:>12.2f}")

    o = results["overall"]
    print(f"\nOVERALL @5:  vector = {o['vector_avg_recall_at5']:.2f}  |  hybrid = {o['hybrid_avg_recall_at5']:.2f}")
    print(f"OVERALL @{k}: vector = {o['vector_avg_recall']:.2f}  |  hybrid = {o['hybrid_avg_recall']:.2f}")


if __name__ == "__main__":
    results = run_benchmark()
    print_report(results)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results -> {RESULTS_PATH}")

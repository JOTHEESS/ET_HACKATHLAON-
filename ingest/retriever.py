"""
RRF fusion retriever for the Hybrid GraphRAG pipeline.

Combines semantic search (ingest.vector_builder) with knowledge graph
traversal (ingest.graph_builder) via Reciprocal Rank Fusion, so a query
surfaces both keyword/semantic hits AND documents connected only through
entity relationships - e.g. the P-204 star chain, whose five documents
share no keywords with each other.

Usage:
    from ingest.retriever import retrieve
    result = retrieve("Was there early warning before the P-204 failure?")
"""
import os
from collections import defaultdict

from ingest.vector_builder import load_vector_store, query_store
from ingest.graph_builder import load_graph, neighbors

RRF_K = 60


def match_entities(graph, query: str) -> list:
    """Graph node texts that appear verbatim in the query, longest match first."""
    query_lower = query.lower()
    matches = [node for node in graph.nodes if len(node) >= 3 and node.lower() in query_lower]
    matches.sort(key=len, reverse=True)
    return matches


def graph_doc_ranking(graph, query: str, max_hops: int = 2) -> list:
    """Rank doc_ids by proximity to entities mentioned in the query.

    Runs a separate bounded BFS per matched entity rather than one shared
    frontier, so a mega-hub entity (e.g. "P-204", mentioned in nearly every
    doc) can't flood every doc to the same hop-0 tie and drown out the
    more specific co-matched entities (e.g. "Zone D", "DE bearing").

    Each seed entity contributes to a given doc's score AT MOST ONCE, at
    that doc's closest hop from the entity. Scoring every node visited
    (rather than every doc, once) would reward documents purely for
    having lots of internal entities - the real OEM manuals have 90-130
    entities each vs. ~10-20 in the synthetic narrative docs, so they'd
    structurally out-rank the tight narrative chain on entity-count alone,
    regardless of true relevance to the query.
    """
    seed_entities = match_entities(graph, query)
    if not seed_entities:
        return []

    doc_scores = defaultdict(float)
    for entity in seed_entities:
        if not graph.has_node(entity):
            continue

        hop_distance = {entity: 0}
        frontier = {entity}
        for hop in range(max_hops):
            next_frontier = set()
            for node in frontier:
                for nbr in neighbors(graph, node):
                    if nbr not in hop_distance:
                        hop_distance[nbr] = hop + 1
                        next_frontier.add(nbr)
            frontier = next_frontier

        best_hop_for_doc = {}
        for node, hop in hop_distance.items():
            for doc_id in graph.nodes[node].get("doc_ids", []):
                if doc_id not in best_hop_for_doc or hop < best_hop_for_doc[doc_id]:
                    best_hop_for_doc[doc_id] = hop
        for doc_id, hop in best_hop_for_doc.items():
            doc_scores[doc_id] += 1.0 / (hop + 1)

    return [doc_id for doc_id, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)]


def vector_doc_ranking(collection, query: str, n_results: int = 20) -> list:
    """De-duplicated doc_id ranking from the top n_results page-chunk hits."""
    seen = []
    for hit in query_store(collection, query, n_results=n_results):
        doc_id = hit["metadata"]["doc_id"]
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def rrf_fuse(rankings: list, k: int = RRF_K) -> list:
    """rankings: list of ranked doc_id lists (best first). Returns fused doc_ids, best first."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def retrieve(query: str, collection=None, graph=None, top_k: int = 12) -> dict:
    if collection is None:
        collection = load_vector_store()
    if graph is None:
        graph = load_graph()

    v_ranking = vector_doc_ranking(collection, query)
    g_ranking = graph_doc_ranking(graph, query)
    fused = rrf_fuse([v_ranking, g_ranking])

    return {
        "query": query,
        "matched_entities": match_entities(graph, query),
        "vector_ranking": v_ranking[:top_k],
        "graph_ranking": g_ranking[:top_k],
        "fused_doc_ids": fused[:top_k],
    }


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    collection = load_vector_store()
    graph = load_graph()

    with open("data/eval/benchmark_questions.json", "r") as f:
        benchmark = json.load(f)

    for q in benchmark["questions"]:
        result = retrieve(q["question"], collection=collection, graph=graph)
        print(f"{q['id']} [{q['retrieval_type']}] {q['question']}")
        print(f"  matched entities: {result['matched_entities']}")
        print(f"  vector-only top:  {result['vector_ranking']}")
        print(f"  graph-only top:   {result['graph_ranking']}")
        print(f"  fused top:        {result['fused_doc_ids']}")
        print()

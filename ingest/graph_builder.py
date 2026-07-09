"""
Knowledge graph builder for the Hybrid GraphRAG pipeline.

Takes the entity/relationship extraction results (ingest.extractor) and
assembles them into a single NetworkX MultiDiGraph spanning the whole
corpus. Entities are merged into one node per exact surface text (e.g.
every doc's "P-204" mention becomes the same node), which is what lets
the star demo chain (IR-556 -> ML-1183 -> VS-204 -> INC-2024-07 -> M-118)
connect across documents that share no keywords.

Usage:
    from ingest.graph_builder import build_graph, save_graph, load_graph
    graph = build_graph(extraction_results)
    save_graph(graph, "data/knowledge_graph.json")
"""
import json
import os
from collections import Counter

import networkx as nx
from networkx.readwrite import json_graph

GRAPH_PATH = "data/knowledge_graph.json"


def build_graph(extraction_results: dict) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for doc_id, extraction in extraction_results.items():
        for entity in extraction.get("entities", []):
            text, etype = entity["text"], entity["type"]
            if graph.has_node(text):
                graph.nodes[text]["doc_ids"].add(doc_id)
                graph.nodes[text]["types"].add(etype)
            else:
                graph.add_node(text, doc_ids={doc_id}, types={etype})

        for rel in extraction.get("relationships", []):
            source, target, relation = rel["source"], rel["target"], rel["relation"]
            for node in (source, target):
                if not graph.has_node(node):
                    graph.add_node(node, doc_ids={doc_id}, types={"UNKNOWN"})
                else:
                    graph.nodes[node]["doc_ids"].add(doc_id)
            key = f"{relation}@{doc_id}"
            graph.add_edge(source, target, key=key, relation=relation, doc_id=doc_id)

    return graph


def _serializable_copy(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    out = graph.copy()
    for _, data in out.nodes(data=True):
        data["doc_ids"] = sorted(data["doc_ids"])
        data["types"] = sorted(data["types"])
    return out


def save_graph(graph: nx.MultiDiGraph, path: str = GRAPH_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json_graph.node_link_data(_serializable_copy(graph), edges="edges")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str = GRAPH_PATH) -> nx.MultiDiGraph:
    with open(path, "r") as f:
        data = json.load(f)
    graph = json_graph.node_link_graph(data, edges="edges")
    for _, attrs in graph.nodes(data=True):
        attrs["doc_ids"] = set(attrs["doc_ids"])
        attrs["types"] = set(attrs["types"])
    return graph


def neighbors(graph: nx.MultiDiGraph, node: str) -> list:
    """Both incoming and outgoing edges, since traversal for retrieval doesn't care about direction."""
    undirected = graph.to_undirected(as_view=True)
    return list(undirected.neighbors(node)) if node in undirected else []


def find_path(graph: nx.MultiDiGraph, source: str, target: str):
    undirected = graph.to_undirected(as_view=True)
    if source not in undirected or target not in undirected:
        return None
    try:
        return nx.shortest_path(undirected, source, target)
    except nx.NetworkXNoPath:
        return None


def graph_stats(graph: nx.MultiDiGraph) -> dict:
    type_counts = Counter()
    for _, data in graph.nodes(data=True):
        type_counts.update(data["types"])
    degrees = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "entity_type_counts": dict(type_counts),
        "top_hub_nodes": degrees[:10],
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.loaders import load_corpus
    from ingest.extractor import extract_corpus

    extraction_path = "data/extraction_results.json"
    if os.path.exists(extraction_path):
        print(f"Loading cached extraction results from {extraction_path}...")
        with open(extraction_path, "r") as f:
            results = json.load(f)
    else:
        print("No extraction_results.json found - running extraction over data/corpus...")
        pages = load_corpus("data/corpus")
        results = extract_corpus(pages)
        with open(extraction_path, "w") as f:
            json.dump(results, f, indent=2)

    graph = build_graph(results)
    save_graph(graph)
    print(f"Saved knowledge graph -> {GRAPH_PATH}")

    stats = graph_stats(graph)
    print(f"\nNodes: {stats['num_nodes']}  Edges: {stats['num_edges']}")
    print(f"Entity types: {stats['entity_type_counts']}")
    print("Top hub nodes:")
    for node, degree in stats["top_hub_nodes"]:
        print(f"  - {node}: degree {degree}")

    path = find_path(graph, "IR-556", "M-118")
    print(f"\nStar chain check IR-556 -> M-118: {path}")

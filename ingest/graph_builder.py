"""
Knowledge graph builder for the Hybrid GraphRAG pipeline.

Takes the entity/relationship extraction results (ingest.extractor) and
assembles them into a single NetworkX MultiDiGraph spanning the whole
corpus. Entities are merged by a canonical key (see _canonical_key),
not exact surface text - the extractor emits different surface forms
of the same real-world entity per document (e.g. "P-204" in one doc,
"Pump P-204" in another, "Centrifugal Pump P-204" in a third), and
merging only on exact string match left those as three disconnected
node identities in practice. Canonicalizing by embedded ID token (e.g.
"Pump P-204" -> "P-204") is what actually lets the star demo chain
(IR-556 -> ML-1183 -> VS-204 -> INC-2024-07 -> M-118) connect across
documents that share no keywords.

Usage:
    from ingest.graph_builder import build_graph, save_graph, load_graph
    graph = build_graph(extraction_results)
    save_graph(graph, "data/knowledge_graph.json")
"""
import json
import os
import re
from collections import Counter

import networkx as nx
from networkx.readwrite import json_graph

GRAPH_PATH = "data/knowledge_graph.json"

# Matches equipment/document/procedure/regulation ID codes like "P-204",
# "M-118", "IR-556", "OISD-132", "INC-2024-07". Searched anywhere in the
# entity text, so "Pump P-204" and "Centrifugal Pump P-204" both resolve
# to the same key as bare "P-204".
_ID_TOKEN_RE = re.compile(r"\b[A-Za-z]{1,6}-\d{2,4}(?:-\d{1,4})?\b")


def _canonical_key(text: str) -> str:
    """Node identity for an entity mention. An embedded ID token (e.g. "P-204")
    takes priority since it's the actual real-world identifier; entities with no
    ID token (e.g. "DE bearing") fall back to a case-folded exact match, which
    still merges harmless casing variants without risking false merges between
    unrelated phrases."""
    match = _ID_TOKEN_RE.search(text)
    if match:
        return match.group(0).upper()
    return text.strip().lower()


def build_graph(extraction_results: dict) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    def _ensure_node(text: str, doc_id: str, etype: str = None) -> str:
        key = _canonical_key(text)
        if graph.has_node(key):
            data = graph.nodes[key]
            data["doc_ids"].add(doc_id)
            data["aliases"].add(text)
            if etype:
                data["types"].add(etype)
            if len(text) > len(data["display_text"]):
                data["display_text"] = text
        else:
            graph.add_node(key, doc_ids={doc_id}, types={etype} if etype else {"UNKNOWN"},
                            aliases={text}, display_text=text)
        return key

    for doc_id, extraction in extraction_results.items():
        for entity in extraction.get("entities", []):
            _ensure_node(entity["text"], doc_id, entity["type"])

        for rel in extraction.get("relationships", []):
            source_key = _ensure_node(rel["source"], doc_id)
            target_key = _ensure_node(rel["target"], doc_id)
            relation = rel["relation"]
            key = f"{relation}@{doc_id}"
            graph.add_edge(source_key, target_key, key=key, relation=relation, doc_id=doc_id)

    return graph


def _serializable_copy(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    out = graph.copy()
    for _, data in out.nodes(data=True):
        data["doc_ids"] = sorted(data["doc_ids"])
        data["types"] = sorted(data["types"])
        data["aliases"] = sorted(data["aliases"])
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
        attrs["aliases"] = set(attrs["aliases"])
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

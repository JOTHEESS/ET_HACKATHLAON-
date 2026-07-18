"""
Graph-path visualization for the Streamlit UI.

Explains WHY each document was retrieved: for every cited/retrieved
document, finds the shortest knowledge-graph path from an entity mentioned
in the query to an entity appearing in that document, then renders those
paths as an interactive pyvis network - entities as circles (query seeds
highlighted), documents as boxes, edges labeled with their relation
(governed_by, references, ...). Only path nodes are drawn, never the full
2-hop neighborhood, so hub entities like P-204 (degree ~38) don't turn the
view into a hairball.

Usage:
    from ingest.path_viz import build_path_html
    html = build_path_html(graph, matched_entities, doc_ids)
"""
import re

import networkx as nx
from pyvis.network import Network

# pyvis's template always injects Bootstrap <link>/<script> CDN tags, even
# with cdn_resources="in_line" (only vis-network itself gets inlined). We
# use none of Bootstrap's widgets, so strip them - the demo must not depend
# on venue wifi.
_CDN_TAG_RE = re.compile(
    r'<(?:link|script)[^>]*cdn\.jsdelivr\.net[^>]*>(?:\s*</script>)?',
    re.IGNORECASE,
)

MAX_DOCS = 6
MAX_HOPS = 3

SEED_COLOR = "#2e7d32"    # query entities - green
ENTITY_COLOR = "#1565c0"  # intermediate entities - blue
DOC_COLOR = "#ef6c00"     # retrieved documents - orange boxes


def _first_relation(graph: nx.MultiDiGraph, u: str, v: str) -> str:
    data = graph.get_edge_data(u, v) or graph.get_edge_data(v, u)
    if not data:
        return ""
    return next(iter(data.values())).get("relation", "")


def retrieval_paths(graph: nx.MultiDiGraph, seed_entities: list, doc_ids: list,
                    max_hops: int = MAX_HOPS) -> dict:
    """doc_id -> shortest node path [seed, ..., anchor] where the anchor
    entity appears in that document. A single-node path means the seed
    itself appears in the document (hop-0). Docs unreachable within
    max_hops of every seed are omitted."""
    undirected = graph.to_undirected(as_view=True)
    paths_by_seed = {
        seed: nx.single_source_shortest_path(undirected, seed, cutoff=max_hops)
        for seed in seed_entities if seed in undirected
    }

    best_paths = {}
    for doc_id in doc_ids:
        best = None
        for paths in paths_by_seed.values():
            for node, path in paths.items():
                if doc_id in graph.nodes[node].get("doc_ids", ()):
                    if best is None or len(path) < len(best):
                        best = path
        if best is not None:
            best_paths[doc_id] = best
    return best_paths


def build_path_html(graph: nx.MultiDiGraph, seed_entities: list, doc_ids: list,
                    max_docs: int = MAX_DOCS, height: str = "420px") -> str:
    """Self-contained HTML (vis.js inlined, no CDN - works offline at a
    demo venue) or None when no query entity matched the graph."""
    paths = retrieval_paths(graph, seed_entities, doc_ids[:max_docs])
    if not paths:
        return None

    net = Network(height=height, width="100%", directed=False,
                  cdn_resources="in_line")
    seeds = set(seed_entities)
    drawn_edges = set()

    def _add_entity(node: str) -> None:
        data = graph.nodes[node]
        net.add_node(
            node,
            label=data.get("display_text", node),
            color=SEED_COLOR if node in seeds else ENTITY_COLOR,
            shape="dot",
            size=18 if node in seeds else 12,
            title=(f"types: {', '.join(sorted(data.get('types', [])))}\n"
                   f"aliases: {', '.join(sorted(data.get('aliases', [])))}"),
        )

    for doc_id, path in paths.items():
        for node in path:
            _add_entity(node)
        for u, v in zip(path, path[1:]):
            if (u, v) not in drawn_edges and (v, u) not in drawn_edges:
                net.add_edge(u, v, label=_first_relation(graph, u, v),
                             font={"size": 10})
                drawn_edges.add((u, v))

        doc_node = f"doc::{doc_id}"
        net.add_node(doc_node, label=doc_id, color=DOC_COLOR, shape="box",
                     title="retrieved document")
        anchor = path[-1]  # the entity that appears in this document
        if (anchor, doc_node) not in drawn_edges:
            net.add_edge(anchor, doc_node, label="appears_in", dashes=True,
                         font={"size": 9})
            drawn_edges.add((anchor, doc_node))

    return _CDN_TAG_RE.sub("", net.generate_html())


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.graph_builder import load_or_build_graph
    from ingest.retriever import match_entities

    query = sys.argv[1] if len(sys.argv) > 1 else (
        "Which regulation governs bearing replacement timelines for Pump P-204?"
    )
    graph = load_or_build_graph()
    seeds = match_entities(graph, query)
    print(f"Query: {query}")
    print(f"Matched entities: {seeds}")
    paths = retrieval_paths(graph, seeds, ["M-118_procedure.pdf", "INC-2024-07_incident_report.pdf"])
    for doc_id, path in paths.items():
        print(f"  {doc_id}: {' -> '.join(path)}")

    html = build_path_html(graph, seeds, list(paths))
    out = "path_viz_test.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote {out} ({len(html)} bytes) - open in a browser to inspect.")

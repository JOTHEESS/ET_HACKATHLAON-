"""The knowledge graph invariants the demo depends on: the star chain is
connected, entity canonicalization merges surface forms, and the graph
loads from the committed JSON."""
import os

import pytest

pytest.importorskip("networkx")

from tests.conftest import ROOT
from ingest.graph_builder import _canonical_key, find_path, load_or_build_graph


@pytest.fixture(scope="module")
def graph():
    return load_or_build_graph(
        path=os.path.join(ROOT, "data", "knowledge_graph.json"),
        extraction_path=os.path.join(ROOT, "data", "extraction_results.json"))


def test_canonical_key_merges_surface_forms():
    assert _canonical_key("P-204") == "P-204"
    assert _canonical_key("Pump P-204") == "P-204"
    assert _canonical_key("Centrifugal Pump P-204") == "P-204"
    assert _canonical_key("INC-2024-07") == "INC-2024-07"
    # No embedded ID token -> case-folded text, no false merges
    assert _canonical_key("DE bearing") == "de bearing"


def test_p204_is_a_single_hub_node(graph):
    assert graph.has_node("P-204")
    assert not graph.has_node("Pump P-204"), "alias leaked in as its own node"
    assert graph.degree("P-204") >= 20, "P-204 should be the corpus hub"


def test_star_chain_is_connected(graph):
    path = find_path(graph, "IR-556", "M-118")
    assert path is not None, "star chain IR-556 -> M-118 must be connected"
    assert len(path) <= 4, f"star chain path unexpectedly long: {path}"
    for node in ("IR-556", "ML-1183", "VS-204", "INC-2024-07", "M-118", "OISD-132"):
        assert graph.has_node(node), f"star chain node {node} missing"

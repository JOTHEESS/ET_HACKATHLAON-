"""Ranking determinism: the exact bug class documented in CLAUDE.md (set
iteration order leaking through sort ties) must never regress."""
import pytest

pytest.importorskip("chromadb")

from ingest.retriever import rrf_fuse


def test_rrf_ties_break_alphabetically():
    # b and c receive identical RRF scores; order must not depend on input order
    fused1 = rrf_fuse([["a", "b"], ["a", "c"]])
    fused2 = rrf_fuse([["a", "c"], ["a", "b"]])
    assert fused1 == fused2 == ["a", "b", "c"]


def test_graph_ranking_stable_across_calls():
    import os
    from tests.conftest import ROOT
    from ingest.graph_builder import load_or_build_graph
    from ingest.retriever import graph_doc_ranking

    graph = load_or_build_graph(
        path=os.path.join(ROOT, "data", "knowledge_graph.json"),
        extraction_path=os.path.join(ROOT, "data", "extraction_results.json"))
    query = "Was there any early warning before the Pump P-204 failure?"
    rankings = [graph_doc_ranking(graph, query) for _ in range(3)]
    assert rankings[0] == rankings[1] == rankings[2]
    assert rankings[0], "expected a non-empty ranking for the star question"

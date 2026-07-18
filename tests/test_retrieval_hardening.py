"""Fuzzy entity matching + scanned-duplicate dedupe: paraphrased queries
still seed the graph, and clean/scanned twins never both occupy top-k slots."""
import os

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("networkx")

from tests.conftest import ROOT
from ingest.graph_builder import load_or_build_graph
from ingest.retriever import (_canonical_doc, _dedupe_canonical,
                              graph_doc_ranking, match_entities)


@pytest.fixture(scope="module")
def graph():
    return load_or_build_graph(
        path=os.path.join(ROOT, "data", "knowledge_graph.json"),
        extraction_path=os.path.join(ROOT, "data", "extraction_results.json"))


def test_verbatim_match_still_works(graph):
    assert "P-204" in match_entities(graph, "Why did Pump P-204 fail?")


def test_punctuation_insensitive_id_match(graph):
    assert "P-204" in match_entities(graph, "why did pump p204 fail?")
    assert "P-204" in match_entities(graph, "history of P 204 bearing issues")


def test_no_false_seed_on_unrelated_query(graph):
    assert match_entities(graph, "hello there general question") == []


def test_canonical_doc_collapses_scanned_twin():
    assert _canonical_doc("INC-2024-07_incident_report_SCANNED.pdf") == \
        "INC-2024-07_incident_report.pdf"
    assert _canonical_doc("M-118_procedure.pdf") == "M-118_procedure.pdf"


def test_dedupe_keeps_first_variant():
    ranking = ["INC-2024-07_incident_report.pdf",
               "INC-2024-07_incident_report_SCANNED.pdf",
               "M-118_procedure.pdf"]
    assert _dedupe_canonical(ranking) == [
        "INC-2024-07_incident_report.pdf", "M-118_procedure.pdf"]


def test_graph_ranking_has_no_scanned_twins(graph):
    ranking = graph_doc_ranking(graph, "Was there early warning before the P-204 failure?")
    canonical = [_canonical_doc(d) for d in ranking]
    assert len(canonical) == len(set(canonical)), f"duplicate twins in {ranking}"

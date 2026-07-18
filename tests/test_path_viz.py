"""Graph path visualization: paths resolve for the star chain, output is a
self-contained offline HTML document, and degenerate inputs return None."""
import os

import pytest

pytest.importorskip("networkx")
pytest.importorskip("pyvis")

from tests.conftest import ROOT
from ingest.graph_builder import load_or_build_graph
from ingest.path_viz import build_path_html, retrieval_paths

STAR_DOCS = [
    "IR-556_inspection_report.pdf",
    "ML-1183_maintenance_log.pdf",
    "VS-204_vibration_trend_log.pdf",
    "INC-2024-07_incident_report.pdf",
    "M-118_procedure.pdf",
]


@pytest.fixture(scope="module")
def graph():
    return load_or_build_graph(
        path=os.path.join(ROOT, "data", "knowledge_graph.json"),
        extraction_path=os.path.join(ROOT, "data", "extraction_results.json"))


def test_star_chain_paths_all_resolve(graph):
    paths = retrieval_paths(graph, ["P-204"], STAR_DOCS)
    assert set(paths) == set(STAR_DOCS)
    # M-118's own text never mentions P-204 - it must be reached via a hop
    assert len(paths["M-118_procedure.pdf"]) == 2, paths["M-118_procedure.pdf"]


def test_html_is_offline_safe(graph):
    html = build_path_html(graph, ["P-204"], STAR_DOCS)
    assert html and "<html" in html.lower()
    assert "cdn.jsdelivr.net" not in html, "CDN load would break at an offline demo"


def test_no_seeds_returns_none(graph):
    assert build_path_html(graph, [], STAR_DOCS) is None
    assert build_path_html(graph, ["NOT-A-NODE-999"], STAR_DOCS) is None

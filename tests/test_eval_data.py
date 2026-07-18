"""Validates the eval dataset itself: every benchmark question is well-formed
and every 'hop' resolves to either a real corpus document or a real knowledge
graph node - so a recall score can never silently reference a target that
doesn't exist."""
import glob
import json
import os

import pytest

from tests.conftest import ROOT

BENCHMARK = os.path.join(ROOT, "data", "eval", "benchmark_questions.json")
RESULTS = os.path.join(ROOT, "data", "eval", "benchmark_results.json")
RETRIEVAL_TYPES = {"keyword_answerable", "graph_only", "hybrid"}


def _questions():
    with open(BENCHMARK) as f:
        return json.load(f)["questions"]


def _corpus_filenames():
    return {os.path.basename(p)
            for p in glob.glob(os.path.join(ROOT, "data", "corpus", "*", "*.pdf"))}


def test_questions_well_formed():
    questions = _questions()
    assert len(questions) == 8
    ids = [q["id"] for q in questions]
    assert len(set(ids)) == len(ids), "duplicate question ids"
    for q in questions:
        assert q["retrieval_type"] in RETRIEVAL_TYPES, q["id"]
        assert q["question"].strip() and q["ground_truth"].strip(), q["id"]
        assert q["hops"], f"{q['id']} has no hops - recall would always be 1.0"


def test_every_hop_resolves_to_doc_or_graph_node():
    networkx = pytest.importorskip("networkx")  # noqa: F841
    from ingest.graph_builder import load_or_build_graph

    filenames = _corpus_filenames()
    graph = load_or_build_graph(
        path=os.path.join(ROOT, "data", "knowledge_graph.json"),
        extraction_path=os.path.join(ROOT, "data", "extraction_results.json"))

    for q in _questions():
        for hop in q["hops"]:
            is_doc = any(f.lower().startswith(hop.lower()) for f in filenames)
            is_node = graph.has_node(hop.upper()) or graph.has_node(hop)
            assert is_doc or is_node, (
                f"{q['id']} hop '{hop}' is neither a corpus doc prefix nor a graph node")


def test_saved_results_match_question_set():
    if not os.path.exists(RESULTS):
        pytest.skip("no benchmark_results.json")
    with open(RESULTS) as f:
        results = json.load(f)
    assert results["top_k"] == 12
    result_ids = {pq["id"] for pq in results["per_question"]}
    question_ids = {q["id"] for q in _questions()}
    assert result_ids == question_ids
    for pq in results["per_question"]:
        assert 0.0 <= pq["vector_recall"] <= 1.0
        assert 0.0 <= pq["hybrid_recall"] <= 1.0

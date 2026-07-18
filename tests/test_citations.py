"""Citation validation: a hallucinated doc_id must never survive to the UI."""
import pytest

pytest.importorskip("anthropic")
pytest.importorskip("chromadb")

from ingest.synthesis import _validate_citations

RETRIEVED = [
    "IR-556_inspection_report.pdf",
    "ML-1183_maintenance_log.pdf",
    "M-118_procedure.pdf",
]


def test_exact_matches_kept():
    assert _validate_citations(["M-118_procedure.pdf"], RETRIEVED) == ["M-118_procedure.pdf"]


def test_hallucinated_doc_dropped():
    assert _validate_citations(["TOTALLY-FAKE-999.pdf"], RETRIEVED) == []


def test_short_form_resolved_to_full_doc_id():
    assert _validate_citations(["IR-556"], RETRIEVED) == ["IR-556_inspection_report.pdf"]


def test_duplicates_collapsed():
    assert _validate_citations(["IR-556", "IR-556_inspection_report.pdf"], RETRIEVED) == [
        "IR-556_inspection_report.pdf"]


def test_non_string_citation_does_not_crash():
    assert _validate_citations([None, 42, "M-118"], RETRIEVED) == ["M-118_procedure.pdf"]

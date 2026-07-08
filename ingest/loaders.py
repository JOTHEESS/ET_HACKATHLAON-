"""
Document loader for the Hybrid GraphRAG pipeline.

Walks data/corpus/{synthetic,real,scanned}/, extracts text per page,
and returns a uniform list of page-level records ready for chunking.

Scanned PDFs are detected automatically (near-zero extractable text on a
page triggers OCR fallback via pytesseract), so you don't need to tell
the loader which folder a file came from.

Usage:
    from ingest.loaders import load_corpus
    pages = load_corpus("data/corpus")
"""
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

MIN_TEXT_CHARS_PER_PAGE = 20


def _ocr_page(page) -> str:
    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img)
    return text.strip()


def _extract_pdf(path: str) -> list:
    records = []
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        native_text = page.get_text("text").strip()
        is_ocr = False
        if len(native_text) < MIN_TEXT_CHARS_PER_PAGE:
            ocr_text = _ocr_page(page)
            text = ocr_text
            is_ocr = True
        else:
            text = native_text
        records.append({"page_num": i + 1, "text": text, "is_ocr": is_ocr})
    doc.close()
    return records


def _extract_tables(path: str) -> dict:
    import pdfplumber
    tables_by_page = {}
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    tables_by_page[i + 1] = tables
    except Exception:
        pass
    return tables_by_page


def load_corpus(corpus_root: str, subfolders: list = None) -> list:
    if subfolders is None:
        subfolders = ["synthetic", "real", "scanned"]

    all_records = []
    for folder in subfolders:
        folder_path = os.path.join(corpus_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(folder_path, fname)
            try:
                pages = _extract_pdf(full_path)
                tables_by_page = _extract_tables(full_path)
            except Exception as e:
                print(f"  [WARN] Failed to parse {fname}: {e}")
                continue

            for rec in pages:
                all_records.append({
                    "doc_id": fname,
                    "source_path": full_path,
                    "source_folder": folder,
                    "page_num": rec["page_num"],
                    "text": rec["text"],
                    "tables": tables_by_page.get(rec["page_num"], []),
                    "is_ocr": rec["is_ocr"],
                })
    return all_records


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/corpus"
    records = load_corpus(root)
    print(f"Loaded {len(records)} pages from {root}")
    by_doc = {}
    for r in records:
        by_doc.setdefault(r["doc_id"], []).append(r)
    for doc_id, recs in sorted(by_doc.items()):
        n_ocr = sum(1 for r in recs if r["is_ocr"])
        n_tables = sum(len(r["tables"]) for r in recs)
        flag = " [OCR]" if n_ocr else ""
        tflag = f" [{n_tables} tables]" if n_tables else ""
        print(f"  - {doc_id}: {len(recs)} pages{flag}{tflag}")

"""
Scan-degradation tool. Takes clean PDFs and produces realistic
'scanned' image-PDF versions to exercise the OCR (pytesseract) path.

Applies: rasterization, slight rotation, gaussian noise, mild blur,
grayscale, brightness variation. Deliberately MILD so OCR still works.

Usage: python make_scanned.py
"""
import os
import random
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageFilter
import img2pdf

random.seed(42)
np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "data", "corpus", "scanned")
os.makedirs(OUT, exist_ok=True)


def degrade_image(pil_img):
    img = pil_img.convert("L")
    angle = random.uniform(-1.2, 1.2)
    img = img.rotate(angle, expand=False, fillcolor=255)
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 6.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    arr = np.clip(arr * random.uniform(0.94, 1.03), 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img.convert("RGB")


def scan_pdf(input_path):
    name = os.path.splitext(os.path.basename(input_path))[0]
    pages = convert_from_path(input_path, dpi=150)
    tmp_imgs = []
    for i, page in enumerate(pages):
        degraded = degrade_image(page)
        tmp = os.path.join(OUT, f"__{name}_p{i}.jpg")
        degraded.save(tmp, "JPEG", quality=72)
        tmp_imgs.append(tmp)
    out_pdf = os.path.join(OUT, f"{name}_SCANNED.pdf")
    with open(out_pdf, "wb") as fh:
        fh.write(img2pdf.convert(tmp_imgs))
    for t in tmp_imgs:
        os.remove(t)
    return out_pdf


if __name__ == "__main__":
    syn = os.path.join(os.path.dirname(__file__), "data", "corpus", "synthetic")
    targets = [
        os.path.join(syn, "IR-556_inspection_report.pdf"),
        os.path.join(syn, "ML-1183_maintenance_log.pdf"),
        os.path.join(syn, "INC-2024-07_incident_report.pdf"),
    ]
    for t in targets:
        out = scan_pdf(t)
        print("Scanned ->", os.path.basename(out))

"""Regenerates architecture_diagram.png. Run from the repo root:

    python scripts/make_architecture_diagram.py

Kept as a script (like the other dataset builders in scripts/) so the
diagram never drifts silently from the code - update the boxes here
whenever the pipeline changes, re-run, commit both.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BEIGE = ("#f0eee7", "#4a4a44")
LAVENDER = ("#e9e7f9", "#3b3390")
GREEN = ("#e2f3ec", "#167a5b")
ORANGE = ("#fbeade", "#b64914")
PINK = ("#f9e2e8", "#8e1f42")
YELLOW = ("#fdf4d7", "#9c7a10")
GRAY = ("#ececec", "#5a5a5a")

fig, ax = plt.subplots(figsize=(13, 14.5), dpi=150)
ax.set_xlim(0, 130)
ax.set_ylim(0, 158)
ax.axis("off")


def box(x, y, w, h, title, sub, colors, title_size=15, sub_size=10.5):
    face, edge = colors
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.4",
                                facecolor=face, edgecolor=edge, linewidth=1.8))
    if sub:
        ax.text(x, y + h * 0.18, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=edge)
        ax.text(x, y - h * 0.24, sub, ha="center", va="center",
                fontsize=sub_size, color=edge)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=edge)


def arrow(x1, y1, x2, y2, style="-|>", dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, linewidth=1.6,
                                 color="#8a8a8a",
                                 linestyle="--" if dashed else "-",
                                 shrinkA=2, shrinkB=2))


# ---- main pipeline (top to bottom) ----
box(65, 150, 34, 8, "Document corpus", None, BEIGE)
box(65, 136, 42, 10, "Document loader", "PyMuPDF + OCR fallback (portable path)", BEIGE)
box(65, 120, 42, 10, "Entity extractor", "Claude API, cached per document", LAVENDER)
box(30, 101, 40, 10, "Knowledge graph", "NetworkX — canonical entity merge", GREEN)
box(100, 101, 46, 11, "Vector store",
    "ChromaDB + bge-small-en-v1.5\nwindowed chunks, auto-built + staleness check", ORANGE)
box(65, 81, 42, 10, "RRF fusion", "graph + vector ranks, deterministic ties", PINK)
box(65, 60, 46, 12, "Synthesis agent",
    "Claude API — validated citations\nconfidence, 60s timeout + retries", LAVENDER)
box(65, 27, 46, 11, "Streamlit chat UI",
    "answer + citations + graph paths", BEIGE)

# ---- side components (new) ----
box(112, 68, 30, 9, "Answer cache", ".synthesis_cache/ — ⚡ repeats", YELLOW,
    title_size=12.5, sub_size=9)
box(112, 50, 30, 10, "Query traces", "logs/query_traces.jsonl\nlatency · tokens · rankings", GRAY,
    title_size=12.5, sub_size=9)
box(19, 44, 32, 11, "Graph path viz", "pyvis, offline\nwhy each doc was retrieved", GREEN,
    title_size=12.5, sub_size=9)

# ---- arrows ----
arrow(65, 146, 65, 141)
arrow(65, 131, 65, 125)
arrow(56, 115, 36, 106.5)
arrow(74, 115, 94, 107)
arrow(38, 96, 56, 86)
arrow(93, 95.5, 74, 86)
arrow(65, 76, 65, 66)
arrow(65, 54, 65, 32.5)
arrow(88, 63, 97, 66, style="<|-|>")           # synthesis <-> answer cache
arrow(85, 57, 97, 52.5)                        # synthesis -> traces
arrow(24, 96, 19, 49.5)                        # knowledge graph -> path viz
arrow(28, 38.5, 46, 30)                        # path viz -> UI

ax.text(65, 8, "Hybrid GraphRAG pipeline  —  Industrial Knowledge Intelligence",
        ha="center", va="center", fontsize=14, style="italic", color="#8a8a8a")

plt.savefig("architecture_diagram.png", bbox_inches="tight",
            facecolor="white", pad_inches=0.4)
print("Wrote architecture_diagram.png")

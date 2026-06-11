"""Manual inspection of the AI-based parser (Slice 3).

Feeds hand-made DetectedRegions (as a real Docling backend would emit) through AiLayoutParser
and prints the resulting LayoutBlocks, so the label->kind mapping and column-aware reading-order
rules are visible — no Docling install needed (a FakeLayoutDetector stands in for the backend).

Run:  .venv/bin/python scripts/inspect_ai_parser.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ai_parser import AiLayoutParser
from rag.layout_detector import DetectedRegion
from tests.fakes import FakeLayoutDetector


def show(title, regions, pdf="handbook.pdf"):
    blocks = AiLayoutParser(FakeLayoutDetector(regions)).parse(pdf)
    print(f"\n{title}")
    for b in blocks:
        print(f"     page={b.page}  kind={b.kind:<7}  text={b.text[:50]!r}")


# A) labels map onto the heading/body/table vocabulary
show("A) label -> kind mapping:", [
    DetectedRegion(1, "Title", "Expense Policy", (0, 0, 100, 10)),
    DetectedRegion(1, "Text", "Submit within 30 days.", (0, 20, 100, 40)),
    DetectedRegion(1, "List", "- portal\n- email", (0, 50, 100, 70)),
    DetectedRegion(1, "Table", "col_a col_b", (0, 80, 100, 120)),
])

# B) shuffled regions come out top-to-bottom
show("B) reading order recovered from shuffle:", [
    DetectedRegion(1, "Text", "third", (0, 200, 100, 220)),
    DetectedRegion(1, "Title", "first", (0, 0, 100, 10)),
    DetectedRegion(1, "Text", "second", (0, 100, 100, 120)),
])

# C) two columns: left column whole, then right column whole (not interleaved)
show("C) two-column page reads column-by-column:", [
    DetectedRegion(1, "Text", "L1", (0, 0, 100, 20)),
    DetectedRegion(1, "Text", "R1", (300, 5, 400, 25)),
    DetectedRegion(1, "Text", "L2", (0, 100, 100, 120)),
    DetectedRegion(1, "Text", "R2", (300, 105, 400, 125)),
])

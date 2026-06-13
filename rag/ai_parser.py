"""AI-based parsing (the lecture's Figure 6 branch).

``AiLayoutParser`` is the deep, testable module: it turns a layout backend's raw DetectedRegions
into the project's ``LayoutBlock`` vocabulary, recovering reading order and stamping each block
with its source. The heavy backend lives behind the ``LayoutDetector`` seam (see layout_detector).
Output honours the same ``parse(pdf) -> [LayoutBlock]`` contract as the quick-path PdfParser, so
the structure-aware Chunker consumes it unchanged.
"""

from __future__ import annotations

from pathlib import Path

from rag.layout_detector import LayoutDetector
from rag.models import LayoutBlock

# Backend layout classes -> our Chunk kind vocabulary (heading | body | table).
_LABEL_TO_KIND = {
    "Title": "heading",
    "Section-header": "heading",
    "Text": "body",
    "List": "body",
    "List-item": "body",
    "Table": "table",
}


def _reading_order(regions):
    """Order regions the way a human reads: by page, then column left-to-right, then top-down.

    Columns are recovered by clustering each region's left edge (``bbox[0]``); a gap wider than
    a fraction of the page width starts a new column. This keeps each column whole instead of
    interleaving them, which a naive top-to-bottom sort would do on multi-column layouts.
    """
    ordered = []
    pages = sorted({r.page for r in regions})
    for page in pages:
        on_page = [r for r in regions if r.page == page]
        width = max((r.bbox[2] for r in on_page), default=0) - min(r.bbox[0] for r in on_page)
        tol = 0.15 * width  # left edges within this distance belong to the same column
        # Walk left edges left-to-right, starting a new column whenever the gap exceeds `tol`.
        column_anchors: list[float] = []
        for x0 in sorted(r.bbox[0] for r in on_page):
            if not column_anchors or x0 - column_anchors[-1] > tol:
                column_anchors.append(x0)

        def column_index(r):
            for i, anchor in enumerate(column_anchors):
                if r.bbox[0] - anchor <= tol:
                    return i
            return len(column_anchors)

        ordered.extend(sorted(on_page, key=lambda r: (column_index(r), r.bbox[1])))
    return ordered


class AiLayoutParser:
    def __init__(self, detector: LayoutDetector):
        self._detector = detector

    def parse(self, pdf_path) -> list[LayoutBlock]:
        document = Path(pdf_path).name
        regions = self._detector.detect(pdf_path)
        ordered = _reading_order(regions)
        return [
            LayoutBlock(
                document=document,
                page=r.page,
                kind=_LABEL_TO_KIND.get(r.label, "body"),
                text=r.text,
            )
            for r in ordered
        ]

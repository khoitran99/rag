"""Layout-detection seam.

`LayoutDetector` is the interface the AI-based parser depends on; tests inject a deterministic
fake. `DoclingLayoutDetector` is the real, fully-offline adapter (Docling layout+OCR on CPU).
Per the project's testing decisions this adapter is heavy/non-deterministic and left untested —
it is verified manually. See ADR-0002 for why Docling stands in for the lecture's LayoutParser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DetectedRegion:
    """One region the layout backend found on a page: its label, text, and bounding box.

    ``bbox`` is ``(x0, y0, x1, y1)`` in page coordinates (origin top-left), which the parser
    uses to recover reading order. ``label`` is the backend's raw class (e.g. "Title", "Text",
    "List", "Table"); the parser maps it onto the Chunk ``kind`` vocabulary.
    """

    page: int
    label: str
    text: str
    bbox: tuple[float, float, float, float]


class LayoutDetector(Protocol):
    def detect(self, pdf_path) -> list[DetectedRegion]: ...


# Docling's layout classes -> our DetectedRegion labels. Anything unmapped falls through as the
# raw label name and AiLayoutParser treats it as body. See https://github.com/docling-project.
_DOCLING_LABELS = {
    "title": "Title",
    "section_header": "Section-header",
    "text": "Text",
    "paragraph": "Text",
    "list_item": "List-item",
    "table": "Table",
}


class DoclingLayoutDetector:
    """Real, fully-offline layout+OCR backend (Docling, CPU).

    Chosen over the lecture's LayoutParser/Detectron2 because Detectron2 has no Apple-Silicon
    wheels (ADR-0002). Heavy and non-deterministic, so left untested and verified manually:
    ``RAG_PARSER=ai rag ingest`` after ``pip install '.[ai]'``.
    """

    def detect(self, pdf_path) -> list[DetectedRegion]:
        from docling.document_converter import DocumentConverter  # lazy: only for real AI runs

        result = DocumentConverter().convert(str(pdf_path))
        regions: list[DetectedRegion] = []
        for item, _level in result.document.iterate_items():
            text = (getattr(item, "text", "") or "").strip()
            prov = getattr(item, "prov", None)
            if not text or not prov:
                continue
            page = prov[0].page_no
            bbox = prov[0].bbox
            label = _DOCLING_LABELS.get(str(getattr(item, "label", "")), "Text")
            regions.append(
                DetectedRegion(
                    page=page,
                    label=label,
                    text=text,
                    bbox=(bbox.l, bbox.t, bbox.r, bbox.b),
                )
            )
        return regions

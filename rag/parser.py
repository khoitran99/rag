"""Document parsing.

Slice 1 used the rule-based "quick path": PyMuPDF text extraction per page. It now emits
``LayoutBlock``s (one ``body`` block per page) so the structure-aware Chunker has a stable
input shape. Slice 8 replaces this with the AI-based LayoutParser (see ADR-0002), which fills
in real ``heading`` and ``table`` blocks — without changing this ``parse(pdf) -> [LayoutBlock]``
contract.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from rag.models import LayoutBlock


class PdfParser:
    def parse(self, pdf_path: str | Path) -> list[LayoutBlock]:
        document = Path(pdf_path).name
        blocks: list[LayoutBlock] = []
        with fitz.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if text:
                    blocks.append(
                        LayoutBlock(document=document, page=i, kind="body", text=text)
                    )
        return blocks

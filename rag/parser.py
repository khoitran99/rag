"""Document parsing — the rule-based "quick path".

PyMuPDF text extraction per page, emitting ``LayoutBlock``s (one ``body`` block per page) so the
structure-aware Chunker has a stable input shape. This is the default, dependency-light parser.
The AI-based parser (``rag.ai_parser.AiLayoutParser``, see ADR-0002) is a sibling that fills in
real ``heading`` and ``table`` blocks; both honour the same ``parse(pdf) -> [LayoutBlock]``
contract, and ``factory`` picks between them via ``Config.parser`` ("quick" | "ai").
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

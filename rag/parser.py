"""Document parsing.

Slice 1 uses the rule-based "quick path": PyMuPDF text extraction per page. Slice 3
replaces this with the AI-based LayoutParser (see ADR-0002) producing structured blocks;
the `parse(pdf) -> [...]` shape is kept stable so downstream code is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class ParsedPage:
    page: int  # 1-based page number, used for citations
    text: str


class PdfParser:
    def parse(self, pdf_path: str | Path) -> list[ParsedPage]:
        pages: list[ParsedPage] = []
        with fitz.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc, start=1):
                pages.append(ParsedPage(page=i, text=page.get_text().strip()))
        return pages

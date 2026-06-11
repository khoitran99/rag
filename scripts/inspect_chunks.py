"""Manual inspection of the structure-aware Chunker (Slice 2).

Feeds hand-made LayoutBlocks through the Chunker and prints the resulting chunks with their
token counts, so the grouping / page-boundary / table-intact / overlap rules are visible.

Run:  .venv/bin/python scripts/inspect_chunks.py
"""

import tiktoken

from rag.chunker import Chunker
from rag.models import LayoutBlock

enc = tiktoken.get_encoding("cl100k_base")
ch = Chunker(max_tokens=128, overlap_tokens=16)


def show(title, blocks):
    chunks = ch.chunk(blocks)
    print(f"\n{title}")
    print(f"  -> {len(chunks)} chunk(s)")
    for c in chunks:
        ntok = len(enc.encode(c.text))
        print(f"     page={c.page}  tokens={ntok:>3}  text={c.text[:60]!r}")


# A) heading stays with its body  -> ONE chunk containing both
show("A) heading + body grouped:", [
    LayoutBlock("d.pdf", 1, "heading", "Expense Policy"),
    LayoutBlock("d.pdf", 1, "body", "Submit reports within 30 days."),
])

# B) different pages never merge  -> TWO chunks, page 1 then page 2
show("B) different pages NOT merged:", [
    LayoutBlock("d.pdf", 1, "body", "First page content."),
    LayoutBlock("d.pdf", 2, "body", "Second page content."),
])

# C) table kept whole even if huge  -> ONE chunk, tokens >> 128
big_table = " ".join(f"r{i}c{i}" for i in range(300))
show("C) table kept intact (oversized):", [
    LayoutBlock("d.pdf", 9, "table", big_table),
])

# D) oversized body splits with overlap  -> several chunks, each <= 128 tokens
long = " ".join(f"word{i}" for i in range(300))
show("D) oversized body split to budget:", [
    LayoutBlock("d.pdf", 4, "body", long),
])

# tokens != characters
text = "Document parsing, document chunking, and indexing prepare PDF data."
print(f"\nTokens vs characters:\n  characters={len(text)}  tokens={len(enc.encode(text))}")

# overlap loses nothing
chunks = ch.chunk([LayoutBlock("d.pdf", 4, "body", long)])
words_out = [w for c in chunks for w in c.text.split()]
print("\nOverlap check (oversized body):")
print(f"  nothing lost: {set(words_out) == set(long.split())}")
print(f"  overlap caused repeats: {len(words_out) > len(set(long.split()))}")

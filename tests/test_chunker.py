import tiktoken

from rag.models import Chunk, LayoutBlock
from rag.chunker import Chunker

_ENC = tiktoken.get_encoding("cl100k_base")


def test_small_body_block_passes_through_unchanged():
    blocks = [
        LayoutBlock(document="handbook.pdf", page=1, kind="body", text="Submit expense reports in the portal."),
    ]

    chunks = Chunker(max_tokens=512, overlap_tokens=64).chunk(blocks)

    assert len(chunks) == 1
    assert chunks[0].text == "Submit expense reports in the portal."
    assert chunks[0].document == "handbook.pdf"
    assert chunks[0].page == 1


def test_heading_is_grouped_with_following_body_when_it_fits():
    blocks = [
        LayoutBlock(document="handbook.pdf", page=2, kind="heading", text="Expense Policy"),
        LayoutBlock(document="handbook.pdf", page=2, kind="body", text="Submit reports within 30 days."),
    ]

    chunks = Chunker(max_tokens=512, overlap_tokens=64).chunk(blocks)

    assert len(chunks) == 1
    assert "Expense Policy" in chunks[0].text
    assert "Submit reports within 30 days." in chunks[0].text
    assert chunks[0].page == 2


def test_oversized_body_is_split_within_budget_with_overlap_and_no_loss():
    text = " ".join(f"word{i}" for i in range(300))  # well over a small budget
    blocks = [LayoutBlock(document="d.pdf", page=4, kind="body", text=text)]

    chunks = Chunker(max_tokens=50, overlap_tokens=10).chunk(blocks)

    assert len(chunks) > 1
    # No chunk exceeds the token budget.
    assert all(len(_ENC.encode(c.text)) <= 50 for c in chunks)
    # Every chunk still traces back to the source block.
    assert all(c.document == "d.pdf" and c.page == 4 for c in chunks)
    # Nothing is lost, and overlap means some words are repeated across windows.
    original_words = text.split()
    emitted_words = [w for c in chunks for w in c.text.split()]
    assert set(emitted_words) == set(original_words)
    assert len(emitted_words) > len(original_words)


def test_table_is_kept_intact_even_when_oversized():
    big_table = " ".join(f"r{i}c{i}" for i in range(300))  # far over budget
    blocks = [
        LayoutBlock(document="d.pdf", page=9, kind="body", text="See the table below."),
        LayoutBlock(document="d.pdf", page=9, kind="table", text=big_table),
    ]

    chunks = Chunker(max_tokens=50, overlap_tokens=10).chunk(blocks)

    # The table is exactly one chunk, unsplit and not merged into the body chunk.
    table_chunks = [c for c in chunks if c.text == big_table]
    assert len(table_chunks) == 1


def test_blocks_on_different_pages_are_not_merged_so_citations_stay_page_accurate():
    blocks = [
        LayoutBlock(document="d.pdf", page=1, kind="body", text="First page content."),
        LayoutBlock(document="d.pdf", page=2, kind="body", text="Second page content."),
    ]

    chunks = Chunker(max_tokens=512, overlap_tokens=64).chunk(blocks)

    assert len(chunks) == 2
    assert {c.page for c in chunks} == {1, 2}

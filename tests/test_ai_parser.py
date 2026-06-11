from rag.ai_parser import AiLayoutParser
from rag.chunker import Chunker
from rag.layout_detector import DetectedRegion
from tests.fakes import FakeLayoutDetector


def test_detector_labels_map_to_chunk_kind_vocabulary():
    regions = [
        DetectedRegion(page=1, label="Title", text="Expense Policy", bbox=(0, 0, 100, 10)),
        DetectedRegion(page=1, label="Text", text="Submit within 30 days.", bbox=(0, 20, 100, 40)),
        DetectedRegion(page=1, label="List", text="- portal\n- email", bbox=(0, 50, 100, 70)),
        DetectedRegion(page=1, label="Table", text="a b c", bbox=(0, 80, 100, 120)),
    ]

    blocks = AiLayoutParser(FakeLayoutDetector(regions)).parse("handbook.pdf")

    kinds = [b.kind for b in blocks]
    assert kinds == ["heading", "body", "body", "table"]


def test_blocks_come_out_in_reading_order_even_when_detector_returns_them_shuffled():
    # Detector emits regions out of order; correct top-to-bottom reading order is 1,2,3.
    regions = [
        DetectedRegion(page=1, label="Text", text="third", bbox=(0, 200, 100, 220)),
        DetectedRegion(page=1, label="Title", text="first", bbox=(0, 0, 100, 10)),
        DetectedRegion(page=1, label="Text", text="second", bbox=(0, 100, 100, 120)),
    ]

    blocks = AiLayoutParser(FakeLayoutDetector(regions)).parse("d.pdf")

    assert [b.text for b in blocks] == ["first", "second", "third"]


def test_two_column_page_reads_left_column_fully_before_right_column():
    # A naive top-to-bottom y-sort would interleave the columns (L1, R1, L2, R2).
    # Correct reading order keeps each column whole: L1, L2, then R1, R2.
    regions = [
        DetectedRegion(page=1, label="Text", text="L1", bbox=(0, 0, 100, 20)),
        DetectedRegion(page=1, label="Text", text="R1", bbox=(300, 5, 400, 25)),
        DetectedRegion(page=1, label="Text", text="L2", bbox=(0, 100, 100, 120)),
        DetectedRegion(page=1, label="Text", text="R2", bbox=(300, 105, 400, 125)),
    ]

    blocks = AiLayoutParser(FakeLayoutDetector(regions)).parse("d.pdf")

    assert [b.text for b in blocks] == ["L1", "L2", "R1", "R2"]


def test_blocks_are_stamped_with_source_filename_and_page_so_citations_resolve():
    regions = [
        DetectedRegion(page=3, label="Text", text="on page three", bbox=(0, 0, 100, 20)),
        DetectedRegion(page=7, label="Text", text="on page seven", bbox=(0, 0, 100, 20)),
    ]

    blocks = AiLayoutParser(FakeLayoutDetector(regions)).parse("/corpus/handbook.pdf")

    assert all(b.document == "handbook.pdf" for b in blocks)
    assert {b.page for b in blocks} == {3, 7}


def test_ai_parser_output_feeds_the_structure_aware_chunker_unchanged():
    # The AI-based parser honours the same [LayoutBlock] contract as the quick path,
    # so the Slice-2 Chunker groups its heading+body and keeps the table intact, no changes.
    regions = [
        DetectedRegion(page=1, label="Title", text="Expense Policy", bbox=(0, 0, 100, 10)),
        DetectedRegion(page=1, label="Text", text="Submit within 30 days.", bbox=(0, 20, 100, 40)),
        DetectedRegion(page=1, label="Table", text="col_a col_b", bbox=(0, 50, 100, 90)),
    ]

    blocks = AiLayoutParser(FakeLayoutDetector(regions)).parse("handbook.pdf")
    chunks = Chunker(max_tokens=512, overlap_tokens=64).chunk(blocks)

    # Heading grouped with its body in one chunk; the table stands alone.
    assert any("Expense Policy" in c.text and "Submit within 30 days." in c.text for c in chunks)
    assert any(c.text == "col_a col_b" for c in chunks)
    assert all(c.document == "handbook.pdf" and c.page == 1 for c in chunks)

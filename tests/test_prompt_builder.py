from rag.models import Chunk
from rag.prompt_builder import PromptBuilder


def test_build_numbers_context_positionally_and_includes_question_and_citation_instruction():
    chunks = [
        Chunk(chunk_id=42, document="a.pdf", page=1, text="Alpha is the first letter."),
        Chunk(chunk_id=7, document="b.pdf", page=9, text="Beta is the second letter."),
    ]

    prompt = PromptBuilder().build("Which letter is first?", chunks)

    # The user's question is present.
    assert "Which letter is first?" in prompt
    # Context is numbered positionally [1], [2] regardless of chunk_id, in order.
    assert prompt.index("[1]") < prompt.index("Alpha is the first letter.")
    assert prompt.index("[2]") < prompt.index("Beta is the second letter.")
    assert prompt.index("[1]") < prompt.index("[2]")
    # There is an instruction to cite using the bracketed indices.
    assert "cite" in prompt.lower()

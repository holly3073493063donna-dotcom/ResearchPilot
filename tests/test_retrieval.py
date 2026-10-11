from researchpilot.models import PaperChunk
from researchpilot.retrieval import retrieve_chunks, select_analysis_chunks


def test_retrieval_prefers_relevant_page() -> None:
    chunks = [
        PaperChunk(
            chunk_id="p1-c1",
            page_number=1,
            text="This paper studies image classification.",
        ),
        PaperChunk(
            chunk_id="p7-c1",
            page_number=7,
            text="The ablation experiment improves accuracy by three percent.",
        ),
    ]

    result = retrieve_chunks("What did the ablation experiment show?", chunks, limit=1)

    assert result[0].page_number == 7


def test_analysis_selection_covers_long_document() -> None:
    chunks = [
        PaperChunk(chunk_id=f"p{page}-c1", page_number=page, text=f"content page {page}")
        for page in range(1, 41)
    ]

    selected = select_analysis_chunks(chunks, limit=12)

    assert len(selected) == 12
    assert min(chunk.page_number for chunk in selected) == 1
    assert max(chunk.page_number for chunk in selected) > 20


def test_retrieval_limits_repeated_chunks_from_the_same_page() -> None:
    chunks = [
        PaperChunk(chunk_id=f"p1-c{part}", page_number=1, text="dataset introduction")
        for part in range(1, 5)
    ]
    chunks.append(
        PaperChunk(
            chunk_id="p9-c1",
            page_number=9,
            text="The dataset is randomly split into 80% training and 20% testing.",
        )
    )

    result = retrieve_chunks("dataset training testing split", chunks, limit=4)

    assert any(chunk.page_number == 9 for chunk in result)
    assert sum(chunk.page_number == 1 for chunk in result) <= 2

import fitz
import pytest

from researchpilot.pdf import PdfError, chunk_pages, extract_pages


def make_pdf(page_texts: list[str]) -> bytes:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    output = document.tobytes()
    document.close()
    return output


def test_extract_pages_preserves_pdf_page_numbers() -> None:
    pages = extract_pages(make_pdf(["First page", "Second page evidence"]))

    assert [page.page_number for page in pages] == [1, 2]
    assert "Second page evidence" in pages[1].text


def test_empty_input_is_rejected() -> None:
    with pytest.raises(PdfError, match="为空"):
        extract_pages(b"")


def test_chunks_never_cross_page_boundaries() -> None:
    pages = extract_pages(make_pdf(["A " * 2_000, "B " * 2_000]))
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)

    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(not ("A" in chunk.text and "B" in chunk.text) for chunk in chunks)


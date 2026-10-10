import re

import fitz

from researchpilot.models import PageText, PaperChunk

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PAGES = 500


class PdfError(ValueError):
    """Raised when a PDF cannot be safely processed."""


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes) -> list[PageText]:
    if not pdf_bytes:
        raise PdfError("PDF 文件为空。")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise PdfError("PDF 超过 50 MB 限制。")

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PdfError("无法打开 PDF，请确认文件未损坏。") from exc

    with document:
        if document.needs_pass:
            raise PdfError("暂不支持受密码保护的 PDF。")
        if document.page_count > MAX_PAGES:
            raise PdfError(f"PDF 共 {document.page_count} 页，超过 {MAX_PAGES} 页限制。")

        pages = [
            PageText(page_number=index + 1, text=normalize_text(page.get_text("text")))
            for index, page in enumerate(document)
        ]

    if not any(page.text for page in pages):
        raise PdfError("未提取到文本；扫描版 PDF 需要后续 OCR 支持。")
    return pages


def chunk_pages(
    pages: list[PageText], chunk_size: int = 1_600, overlap: int = 240
) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    for page in pages:
        text = page.text
        if not text:
            continue
        start = 0
        part = 1
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            excerpt = text[start:end].strip()
            if excerpt:
                chunks.append(
                    PaperChunk(
                        chunk_id=f"p{page.page_number}-c{part}",
                        page_number=page.page_number,
                        text=excerpt,
                    )
                )
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
            part += 1
    return chunks


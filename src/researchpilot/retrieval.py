import math
import re
from collections import Counter

from researchpilot.models import PaperChunk

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]")
ANALYSIS_TERMS = (
    "abstract introduction method methodology experiment result discussion "
    "conclusion limitation future",
    "摘要 引言 方法 实验 结果 讨论 结论 局限 未来",
)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def retrieve_chunks(
    query: str,
    chunks: list[PaperChunk],
    limit: int = 6,
    max_per_page: int = 2,
) -> list[PaperChunk]:
    if not chunks:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return chunks[:limit]

    document_frequencies: Counter[str] = Counter()
    chunk_tokens: list[list[str]] = []
    for chunk in chunks:
        tokens = tokenize(chunk.text)
        chunk_tokens.append(tokens)
        document_frequencies.update(set(tokens))

    total = len(chunks)
    scored: list[tuple[float, int, PaperChunk]] = []
    for index, (chunk, tokens) in enumerate(zip(chunks, chunk_tokens, strict=True)):
        frequencies = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            if token not in frequencies:
                continue
            inverse_frequency = math.log((total + 1) / (document_frequencies[token] + 1)) + 1
            score += (1 + math.log(frequencies[token])) * inverse_frequency
        scored.append((score, -index, chunk))

    ranked = sorted(scored, reverse=True)
    candidates = [chunk for score, _, chunk in ranked if score > 0] or chunks
    selected: list[PaperChunk] = []
    page_counts: Counter[int] = Counter()
    for chunk in candidates:
        if page_counts[chunk.page_number] >= max_per_page:
            continue
        selected.append(chunk)
        page_counts[chunk.page_number] += 1
        if len(selected) >= limit:
            break
    return selected


def select_analysis_chunks(chunks: list[PaperChunk], limit: int = 18) -> list[PaperChunk]:
    if len(chunks) <= limit:
        return chunks

    selected: dict[str, PaperChunk] = {}
    for chunk in chunks[:3]:
        selected[chunk.chunk_id] = chunk

    keyword_query = " ".join(ANALYSIS_TERMS)
    for chunk in retrieve_chunks(keyword_query, chunks, limit=10):
        selected[chunk.chunk_id] = chunk

    remaining = max(limit - len(selected), 0)
    if remaining:
        step = max(len(chunks) // remaining, 1)
        for index in range(0, len(chunks), step):
            selected[chunks[index].chunk_id] = chunks[index]
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        for chunk in reversed(chunks):
            selected[chunk.chunk_id] = chunk
            if len(selected) >= limit:
                break
    return list(selected.values())[:limit]

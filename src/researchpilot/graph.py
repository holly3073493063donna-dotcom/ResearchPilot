import json
import re
from collections.abc import Callable
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from researchpilot.models import (
    AnalysisItem,
    AnswerResult,
    Evidence,
    PaperAnalysis,
    PaperChunk,
    ResearchMode,
    ResearchRequest,
)
from researchpilot.retrieval import retrieve_chunks, select_analysis_chunks


class ResearchState(TypedDict, total=False):
    request: ResearchRequest
    chunks: list[PaperChunk]
    context: list[PaperChunk]
    answer: AnswerResult
    analysis: PaperAnalysis


def format_context(chunks: list[PaperChunk]) -> str:
    return "\n\n".join(f"[PDF page {chunk.page_number}]\n{chunk.text}" for chunk in chunks)


PAGE_CITATION_PATTERN = re.compile(r"\[p\.\s*(\d+)\]", re.IGNORECASE)


def _cited_pages(answer: str) -> list[int]:
    return list(dict.fromkeys(int(page) for page in PAGE_CITATION_PATTERN.findall(answer)))


def _evidence(
    chunks: list[PaperChunk], answer: str = "", limit: int = 4
) -> list[Evidence]:
    evidence: list[Evidence] = []
    seen_pages: set[int] = set()
    cited_pages = _cited_pages(answer)
    if cited_pages:
        page_order = {page: index for index, page in enumerate(cited_pages)}
        candidates = sorted(
            (chunk for chunk in chunks if chunk.page_number in page_order),
            key=lambda chunk: page_order[chunk.page_number],
        )
    else:
        candidates = chunks

    for chunk in candidates:
        if chunk.page_number in seen_pages:
            continue
        evidence.append(Evidence(page_number=chunk.page_number, excerpt=chunk.text[:700].strip()))
        seen_pages.add(chunk.page_number)
        if len(evidence) >= limit:
            break
    return evidence


def _expand_search_query(model: BaseChatModel, question: str) -> str:
    try:
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "Rewrite the user's research-paper question as one concise line of "
                        "English search keywords. Include likely academic synonyms and section "
                        "terms. Return keywords only, without explanation."
                    )
                ),
                HumanMessage(content=question),
            ]
        )
    except Exception:
        return question
    expansion = str(response.content).strip()
    return f"{question}\n{expansion}" if expansion else question


def _merge_context(
    primary: list[PaperChunk], supplementary: list[PaperChunk], limit: int = 12
) -> list[PaperChunk]:
    merged: dict[str, PaperChunk] = {}
    for chunk in [*primary, *supplementary]:
        merged[chunk.chunk_id] = chunk
        if len(merged) >= limit:
            break
    return list(merged.values())


def _sanitize_analysis(analysis: PaperAnalysis, chunks: list[PaperChunk]) -> PaperAnalysis:
    allowed_pages = {chunk.page_number for chunk in chunks}
    fallback_page = min(allowed_pages) if allowed_pages else 1
    fields = (
        "research_question",
        "approach",
        "key_results",
        "contributions",
        "limitations",
        "future_work",
    )
    updates: dict[str, AnalysisItem] = {}
    for field in fields:
        item = getattr(analysis, field)
        pages = sorted({page for page in item.pages if page in allowed_pages}) or [fallback_page]
        updates[field] = AnalysisItem(finding=item.finding, pages=pages)
    return analysis.model_copy(update=updates)


def _parse_paper_analysis(content: object) -> PaperAnalysis:
    text = str(content).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("模型未返回可解析的 JSON 结构化分析。")

    try:
        return PaperAnalysis.model_validate_json(text[start : end + 1])
    except ValueError as exc:
        raise ValueError("模型返回的结构化分析不符合预期格式。") from exc


def build_research_graph(model: BaseChatModel):
    def retrieve(state: ResearchState) -> ResearchState:
        request = state["request"]
        if request.mode == ResearchMode.AUTOMATIC:
            context = select_analysis_chunks(state["chunks"])
        else:
            expanded_query = _expand_search_query(model, request.question)
            primary = retrieve_chunks(expanded_query, state["chunks"], limit=8)
            supplementary = select_analysis_chunks(state["chunks"], limit=8)
            context = _merge_context(primary, supplementary)
        return {"context": context}

    def answer_question(state: ResearchState) -> ResearchState:
        context = state["context"]
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You answer questions only from the supplied paper excerpts. "
                        "Cite factual claims with the exact PDF page marker [p. N]. "
                        "If the excerpts are insufficient, say so clearly. "
                        "Reply in the user's language."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{state['request'].question}\n\n"
                        f"Paper excerpts:\n{format_context(context)}"
                    )
                ),
            ]
        )
        answer = str(response.content)
        return {"answer": AnswerResult(answer=answer, evidence=_evidence(context, answer))}

    def analyze_paper(state: ResearchState) -> ResearchState:
        context = state["context"]
        schema = json.dumps(PaperAnalysis.model_json_schema(), ensure_ascii=False)
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "Analyze only the supplied paper excerpts. Every finding must name "
                        "one or more supporting PDF page numbers. Distinguish author-stated "
                        "limitations from your careful inferences. Use concise Chinese unless "
                        "the paper is clearly requested otherwise. Return only one valid JSON "
                        "object without Markdown fences or additional commentary. The JSON must "
                        f"follow this schema exactly: {schema}"
                    )
                ),
                HumanMessage(
                    content=f"Create a structured paper analysis.\n\n{format_context(context)}"
                ),
            ]
        )
        analysis = _parse_paper_analysis(response.content)
        return {"analysis": _sanitize_analysis(analysis, context)}

    def route(state: ResearchState) -> str:
        return "analyze" if state["request"].mode == ResearchMode.AUTOMATIC else "answer"

    workflow = StateGraph(ResearchState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("answer", answer_question)
    workflow.add_node("analyze", analyze_paper)
    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges("retrieve", route, {"answer": "answer", "analyze": "analyze"})
    workflow.add_edge("answer", END)
    workflow.add_edge("analyze", END)
    return workflow.compile()


GraphFactory = Callable[[BaseChatModel], object]

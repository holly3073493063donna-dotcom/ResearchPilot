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


def _evidence(chunks: list[PaperChunk], limit: int = 4) -> list[Evidence]:
    return [
        Evidence(page_number=chunk.page_number, excerpt=chunk.text[:700].strip())
        for chunk in chunks[:limit]
    ]


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


def build_research_graph(model: BaseChatModel):
    def retrieve(state: ResearchState) -> ResearchState:
        request = state["request"]
        if request.mode == ResearchMode.AUTOMATIC:
            context = select_analysis_chunks(state["chunks"])
        else:
            context = retrieve_chunks(request.question, state["chunks"])
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
        return {"answer": AnswerResult(answer=str(response.content), evidence=_evidence(context))}

    def analyze_paper(state: ResearchState) -> ResearchState:
        context = state["context"]
        structured_model = model.with_structured_output(PaperAnalysis)
        analysis = structured_model.invoke(
            [
                SystemMessage(
                    content=(
                        "Analyze only the supplied paper excerpts. Every finding must name "
                        "one or more supporting PDF page numbers. Distinguish author-stated "
                        "limitations from your careful inferences. Use concise Chinese unless "
                        "the paper is clearly requested otherwise."
                    )
                ),
                HumanMessage(
                    content=f"Create a structured paper analysis.\n\n{format_context(context)}"
                ),
            ]
        )
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

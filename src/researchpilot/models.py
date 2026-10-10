from enum import StrEnum

from pydantic import BaseModel, Field


class ResearchMode(StrEnum):
    AUTOMATIC = "automatic"
    CUSTOM = "custom"


class PageText(BaseModel):
    page_number: int = Field(ge=1)
    text: str


class PaperChunk(BaseModel):
    chunk_id: str
    page_number: int = Field(ge=1)
    text: str


class Evidence(BaseModel):
    page_number: int = Field(ge=1)
    excerpt: str


class AnswerResult(BaseModel):
    answer: str
    evidence: list[Evidence]


class AnalysisItem(BaseModel):
    finding: str = Field(description="A concise finding grounded in the supplied paper text")
    pages: list[int] = Field(description="PDF page numbers supporting the finding")


class PaperAnalysis(BaseModel):
    title: str
    research_question: AnalysisItem
    approach: AnalysisItem
    key_results: AnalysisItem
    contributions: AnalysisItem
    limitations: AnalysisItem
    future_work: AnalysisItem


class ResearchRequest(BaseModel):
    mode: ResearchMode
    question: str = ""


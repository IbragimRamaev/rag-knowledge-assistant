from pydantic import BaseModel

from app.core.models import HistoryTurn


class AskRequest(BaseModel):
    question: str
    history: list[HistoryTurn] = []


class AskResponse(BaseModel):
    answer: str
    source_documents: list[str]


class IngestRequest(BaseModel):
    documents_dir: str = "data/documents"


class IngestResponse(BaseModel):
    chunks_ingested: int
    documents_dir: str

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class IngestRequest(BaseModel):
    documents_dir: str = "data/documents"


class IngestResponse(BaseModel):
    chunks_ingested: int
    documents_dir: str

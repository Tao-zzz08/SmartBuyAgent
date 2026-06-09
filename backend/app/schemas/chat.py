from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None
    debug: bool = True

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class ProductCardResponse(BaseModel):
    product_id: str
    title: str
    brand: str | None
    price: int
    image_url: str | None
    tags: list[str]
    attributes: dict[str, str]
    source_url: str | None
    compare_url: str | None
    recommend_reason: str


class CitationResponse(BaseModel):
    chunk_id: str
    title: str | None
    section: str | None
    section_path: str | None
    source_file: str | None
    content_preview: str
    score: float


class ChatResponseSchema(BaseModel):
    answer: str
    product_cards: list[ProductCardResponse]
    citations: list[CitationResponse]
    trace: list[dict[str, Any]]
    session_id: str | None = None

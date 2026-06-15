from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    turn_id: int | None = None
    rating: Literal["helpful", "not_helpful"]
    reason: str | None = Field(default=None, max_length=128)
    comment: str | None = Field(default=None, max_length=1000)
    query: str | None = None
    answer_preview: str | None = None

    @field_validator("session_id")
    @classmethod
    def session_id_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("session_id must not be blank")
        return stripped

    @field_validator("reason", "comment", "query", "answer_preview")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class FeedbackResponse(BaseModel):
    id: int
    status: str

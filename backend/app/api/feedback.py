from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import ChatFeedback
from app.schemas.feedback import FeedbackRequest, FeedbackResponse


router = APIRouter(prefix="/api", tags=["feedback"])

ANSWER_PREVIEW_MAX_LENGTH = 500


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    feedback = ChatFeedback(
        session_id=request.session_id,
        turn_id=request.turn_id,
        rating=request.rating,
        reason=request.reason,
        comment=request.comment,
        query=request.query,
        answer_preview=_truncate_answer_preview(request.answer_preview),
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return FeedbackResponse(id=feedback.id, status="saved")


def _truncate_answer_preview(value: str | None) -> str | None:
    if value is None:
        return None
    return value[:ANSWER_PREVIEW_MAX_LENGTH]

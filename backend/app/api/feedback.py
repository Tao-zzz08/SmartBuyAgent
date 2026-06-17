from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.cache.cache_service import CacheService, get_cache_service
from app.core.config import settings
from app.core.db import get_db
from app.models import ChatFeedback
from app.schemas.feedback import FeedbackRequest, FeedbackResponse


router = APIRouter(prefix="/api", tags=["feedback"])

ANSWER_PREVIEW_MAX_LENGTH = 500


def get_feedback_cache_service() -> CacheService:
    return get_cache_service()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
    cache_service: CacheService = Depends(get_feedback_cache_service),
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
    _aggregate_feedback(cache_service, request.session_id, request.rating)
    return FeedbackResponse(id=feedback.id, status="saved")


def _truncate_answer_preview(value: str | None) -> str | None:
    if value is None:
        return None
    return value[:ANSWER_PREVIEW_MAX_LENGTH]


def _aggregate_feedback(
    cache_service: CacheService,
    session_id: str,
    rating: str,
) -> None:
    try:
        cache_service.incr(
            f"smartbuy:feedback:{session_id}:{rating}",
            ttl_seconds=settings.FEEDBACK_CACHE_TTL_SECONDS,
        )
    except Exception:
        return

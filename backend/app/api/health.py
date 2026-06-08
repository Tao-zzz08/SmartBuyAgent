from fastapi import APIRouter

from app.core.config import API_VERSION, settings


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": API_VERSION,
    }

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SQLITE_PREFIX = "sqlite:///"


def _prepare_database_url(database_url: str) -> str:
    if not database_url.startswith(SQLITE_PREFIX):
        return database_url

    db_path_value = database_url.removeprefix(SQLITE_PREFIX)
    if db_path_value == ":memory:":
        return database_url

    db_path = Path(db_path_value)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{SQLITE_PREFIX}{db_path.as_posix()}"


DATABASE_URL = _prepare_database_url(settings.DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=not DATABASE_URL.startswith("sqlite"),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

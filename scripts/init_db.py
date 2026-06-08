from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.db import Base, DATABASE_URL, engine  # noqa: E402
import app.models  # noqa: E402,F401


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized successfully: {DATABASE_URL}")


if __name__ == "__main__":
    init_db()

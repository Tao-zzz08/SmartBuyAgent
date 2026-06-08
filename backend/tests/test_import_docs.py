from pathlib import Path
import json
import sys

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.models import Document, DocumentChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from import_docs import (  # noqa: E402
    import_documents,
    load_markdown_files,
    parse_front_matter,
    strip_front_matter,
)


def _create_test_session(db_name: str):
    db_path = PROJECT_ROOT / "data" / db_name
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, TestingSessionLocal(), db_path


def _count(db, model, *where_clauses) -> int:
    statement = select(func.count()).select_from(model)
    if where_clauses:
        statement = statement.where(*where_clauses)
    return db.scalar(statement) or 0


def test_markdown_docs_can_be_scanned_and_parsed() -> None:
    files = load_markdown_files(PROJECT_ROOT)

    assert len(files) == 17

    phone_camera_path = (
        PROJECT_ROOT
        / "data"
        / "knowledge_docs"
        / "phone"
        / "phone_camera_guide.md"
    )
    markdown_text = phone_camera_path.read_text(encoding="utf-8")
    metadata = parse_front_matter(markdown_text)
    body = strip_front_matter(markdown_text)

    assert metadata["title"] == "手机拍照选购指南"
    assert metadata["doc_type"] == "guide"
    assert metadata["category_id"] == "cat_phone"
    assert metadata["category_path"] == "数码/手机"
    assert not body.startswith("---")
    assert "doc_type:" not in "\n".join(body.splitlines()[:5])


def test_import_docs_is_idempotent() -> None:
    engine, db, db_path = _create_test_session("smartbuy_import_docs_test.db")
    try:
        first_stats = import_documents(db, PROJECT_ROOT)
        first_chunk_count = _count(db, DocumentChunk)

        assert first_stats["documents_inserted"] == 17
        assert _count(db, Document) == 17
        assert first_chunk_count > 17

        document = db.get(Document, "doc_phone_phone_camera_guide")
        assert document is not None
        assert document.source_file == "data/knowledge_docs/phone/phone_camera_guide.md"
        assert document.doc_type == "guide"
        assert document.category_id == "cat_phone"

        chunks = db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        ).scalars().all()
        assert len(chunks) >= 1

        first_chunk = chunks[0]
        metadata = json.loads(first_chunk.metadata_json)
        for key in {
            "title",
            "doc_type",
            "category_id",
            "category_path",
            "source_file",
            "section",
            "chunk_index",
        }:
            assert key in metadata
        assert first_chunk.vector_id is None

        second_stats = import_documents(db, PROJECT_ROOT)

        assert second_stats["documents_inserted"] == 0
        assert second_stats["documents_updated"] == 17
        assert _count(db, Document) == 17
        assert _count(db, DocumentChunk) == first_chunk_count
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)

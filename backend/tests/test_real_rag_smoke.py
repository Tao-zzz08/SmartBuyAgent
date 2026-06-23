from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.chat.query_understanding import QueryUnderstandingService
    from app.core.db import Base
    from app.models import Document, DocumentChunk
    from app.retrieval.retrieval_service import KnowledgeRetrievalService
except ModuleNotFoundError as exc:
    REAL_RAG_IMPORT_ERROR = exc
else:
    REAL_RAG_IMPORT_ERROR = None


FIXTURES_DIR = Path(__file__).parent / "fixtures"
CATEGORY_IDS = {
    "phone": "cat_phone",
    "shoes": "cat_shoes",
    "skincare": "cat_skincare",
}


class FixtureKnowledgeCollection:
    def __init__(self, chunks: list[dict[str, str]]) -> None:
        self._chunks = chunks

    def count(self) -> int:
        return len(self._chunks)

    def query(self, **kwargs):
        n_results = int(kwargs.get("n_results") or len(self._chunks))
        chunks = self._chunks[:n_results]
        return {
            "metadatas": [
                [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "category_id": chunk["category_id"],
                        "doc_type": chunk["doc_type"],
                        "category_path": chunk["category_path"],
                    }
                    for chunk in chunks
                ]
            ],
            "distances": [[0.05 + index * 0.01 for index, _ in enumerate(chunks)]],
        }


class FixtureChromaClient:
    def __init__(self, chunks: list[dict[str, str]]) -> None:
        self._collection = FixtureKnowledgeCollection(chunks)

    def get_collection(self, name: str):
        del name
        return self._collection


class TinyEmbeddingService:
    def embed_text(self, text: str) -> list[float]:
        del text
        return [0.1, 0.2, 0.3, 0.4]


def test_real_rag_phone_camera_grounded_answer_smoke(tmp_path) -> None:
    _skip_without_real_rag_dependencies()
    case = load_rag_smoke_fixture("phone_camera_grounded_smoke")
    engine, db = _test_session(tmp_path)
    try:
        chunks = _normalize_case_chunks(case)
        _insert_document_and_chunks(db, chunks)
        service = KnowledgeRetrievalService(
            db,
            embedding_service=TinyEmbeddingService(),
            chroma_client=FixtureChromaClient(chunks),
        )

        citations = service.search_knowledge(
            case.get("knowledge_query") or case["query"],
            category_id=_category_id(case["category"]),
            top_k=3,
            preferences=case.get("preferences") or [],
        )
        answer = case["answer_template"]

        assert_rag_expectations(answer, citations, case["expect"])
        assert case["category"] in {"phone", "shoes", "skincare"}
    finally:
        db.close()
        engine.dispose()


def test_real_rag_skincare_safety_smoke(tmp_path) -> None:
    _skip_without_real_rag_dependencies()
    case = load_rag_smoke_fixture("skincare_acne_safety_smoke")
    engine, db = _test_session(tmp_path)
    try:
        chunks = _normalize_case_chunks(case)
        _insert_document_and_chunks(db, chunks)
        understanding = QueryUnderstandingService(llm_enabled=False).understand(
            case["query"]
        )
        service = KnowledgeRetrievalService(
            db,
            embedding_service=TinyEmbeddingService(),
            chroma_client=FixtureChromaClient(chunks),
        )

        citations = service.search_knowledge(
            understanding.effective_query,
            category_id=_category_id(case["category"]),
            top_k=3,
            preferences=understanding.preferences,
            negative_preferences=understanding.negative_preferences,
        )
        answer = case["answer_template"]

        assert_rag_expectations(answer, citations, case["expect"])
        assert_text_not_contains_any(
            understanding.effective_query,
            case["expect"].get("effective_query_forbidden") or [],
        )
        assert_text_not_contains_any(
            service.last_query,
            case["expect"].get("knowledge_query_forbidden") or [],
        )
    finally:
        db.close()
        engine.dispose()


def load_rag_smoke_fixture(case_id: str) -> dict:
    payload = json.loads(
        (FIXTURES_DIR / "rag_smoke_fixtures.json").read_text(encoding="utf-8")
    )
    for case in payload["cases"]:
        if case["id"] == case_id:
            return case
    raise AssertionError(f"missing RAG smoke fixture: {case_id}")


def assert_rag_expectations(answer: str, citations, expect: dict) -> None:
    assert len(citations) >= int(expect.get("min_citations") or 0)
    assert_citations_are_real(
        citations,
        expected_chunk_ids=expect.get("expected_chunk_ids") or [],
        required_fields=expect.get("citation_must_have_fields") or [],
    )
    assert_answer_contains_any(answer, expect.get("answer_must_include_any") or [])
    assert_text_not_contains_any(answer, expect.get("answer_forbidden") or [])


def assert_citations_are_real(
    citations,
    *,
    expected_chunk_ids: list[str],
    required_fields: list[str],
) -> None:
    ids = [citation.chunk_id for citation in citations]
    for expected_chunk_id in expected_chunk_ids:
        assert expected_chunk_id in ids
    for citation in citations:
        for field_name in required_fields:
            assert getattr(citation, field_name), field_name


def assert_answer_contains_any(answer: str, terms: list[str]) -> None:
    assert any(term in answer for term in terms), answer


def assert_text_not_contains_any(text: str, forbidden: list[str]) -> None:
    offenders = [term for term in forbidden if term in text]
    assert offenders == [], f"forbidden terms {offenders!r} found in {text!r}"


def _skip_without_real_rag_dependencies() -> None:
    if REAL_RAG_IMPORT_ERROR is not None:
        pytest.skip(
            f"real RAG smoke requires backend dependencies: {REAL_RAG_IMPORT_ERROR}"
        )


def _test_session(tmp_path):
    db_path = tmp_path / "real_rag_smoke.db"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal()


def _normalize_case_chunks(case: dict) -> list[dict[str, str]]:
    document = case["document"]
    normalized: list[dict[str, str]] = []
    for chunk in case["chunks"]:
        category = chunk.get("category") or document["category"]
        doc_type = document.get("doc_type") or "guide"
        title = document["title"]
        section = chunk.get("section") or title
        normalized.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": document["doc_id"],
                "source": chunk.get("source") or document["source_file"],
                "category_id": _category_id(category),
                "category_path": category,
                "doc_type": doc_type,
                "title": title,
                "section": section,
                "text": chunk["text"],
            }
        )
    return normalized


def _insert_document_and_chunks(db, chunks: list[dict[str, str]]) -> None:
    documents: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        documents.setdefault(
            chunk["document_id"],
            {
                "source_file": chunk["source"],
                "doc_type": chunk["doc_type"],
                "category_id": chunk["category_id"],
                "category_path": chunk["category_path"],
                "title": chunk["title"],
            },
        )

    for document_id, document_metadata in documents.items():
        db.add(
            Document(
                id=document_id,
                source_file=document_metadata["source_file"],
                doc_type=document_metadata["doc_type"],
                category_id=document_metadata["category_id"],
                title=document_metadata["title"],
                metadata_json=json.dumps(document_metadata, ensure_ascii=False),
            )
        )

    for index, chunk in enumerate(chunks):
        chunk_metadata = {
            "source_file": chunk["source"],
            "doc_type": chunk["doc_type"],
            "category_id": chunk["category_id"],
            "category_path": chunk["category_path"],
            "title": chunk["title"],
            "section": chunk["section"],
            "section_path": f"{chunk['title']}/{chunk['section']}",
        }
        db.add(
            DocumentChunk(
                id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                category_id=chunk["category_id"],
                chunk_index=index,
                content=chunk["text"],
                metadata_json=json.dumps(chunk_metadata, ensure_ascii=False),
                vector_id=chunk["chunk_id"],
            )
        )
    db.commit()


def _category_id(category: str) -> str:
    return CATEGORY_IDS[category]

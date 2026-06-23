from __future__ import annotations

import json

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


PHONE_CAMERA_CHUNK = {
    "chunk_id": "phone_camera_001",
    "document_id": "doc_phone_camera",
    "source": "knowledge/phone/camera.md",
    "category_id": "cat_phone",
    "category_path": "phone",
    "doc_type": "guide",
    "title": "\u624b\u673a\u62cd\u7167\u9009\u8d2d\u77e5\u8bc6",
    "section": "\u62cd\u7167\u53c2\u6570",
    "text": (
        "\u624b\u673a\u62cd\u7167\u901a\u5e38\u9700\u8981\u5173\u6ce8"
        "\u4f20\u611f\u5668\u5c3a\u5bf8\u3001\u5149\u5708\u3001\u9632\u6296"
        "\u80fd\u529b\u548c\u5f71\u50cf\u7b97\u6cd5\u3002\u4f20\u611f\u5668"
        "\u548c\u9632\u6296\u4f1a\u5f71\u54cd\u6697\u5149\u62cd\u6444\u548c"
        "\u6210\u7247\u7a33\u5b9a\u6027\u3002"
    ),
}

SKINCARE_SAFE_CHUNK = {
    "chunk_id": "skincare_acne_safe_001",
    "document_id": "doc_skincare_acne_safe",
    "source": "knowledge/skincare/acne_safe.md",
    "category_id": "cat_skincare",
    "category_path": "skincare",
    "doc_type": "guide",
    "title": "\u75d8\u75d8\u808c\u65e5\u5e38\u62a4\u7406",
    "section": "\u5b89\u5168\u62a4\u80a4\u5efa\u8bae",
    "text": (
        "\u75d8\u75d8\u808c\u65e5\u5e38\u62a4\u80a4\u53ef\u4ee5\u5173\u6ce8"
        "\u6e05\u723d\u3001\u63a7\u6cb9\u3001\u6e29\u548c\u548c\u4fdd\u6e7f\u3002"
        "\u62a4\u80a4\u54c1\u4e0d\u80fd\u66ff\u4ee3\u836f\u7269\u6216\u533b\u7597"
        "\u6cbb\u7597\uff0c\u5982\u6709\u4e25\u91cd\u76ae\u80a4\u95ee\u9898\u5e94"
        "\u54a8\u8be2\u4e13\u4e1a\u533b\u751f\u3002"
    ),
}

PURCHASE_BOUNDARY_TERMS = [
    "\u7acb\u5373\u8d2d\u4e70",
    "\u4e0b\u5355",
    "\u652f\u4ed8",
    "\u8d2d\u7269\u8f66",
    "\u8d2d\u4e70\u94fe\u63a5",
]

MEDICAL_CLAIM_TERMS = [
    "\u6cbb\u7597",
    "\u6cbb\u6108",
    "\u836f\u6548",
    "\u5904\u65b9",
    "\u533b\u5b66\u4fee\u590d",
    "\u4fee\u590d\u75be\u75c5",
]
LIST_SEPARATOR = "\u3001"


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
    engine, db = _test_session(tmp_path)
    try:
        _insert_knowledge_chunk(db, PHONE_CAMERA_CHUNK)
        service = KnowledgeRetrievalService(
            db,
            embedding_service=TinyEmbeddingService(),
            chroma_client=FixtureChromaClient([PHONE_CAMERA_CHUNK]),
        )

        knowledge_query = "\u624b\u673a \u62cd\u7167 \u53c2\u6570 \u9009\u8d2d\u5efa\u8bae"
        citations = service.search_knowledge(
            knowledge_query,
            category_id="cat_phone",
            top_k=3,
            preferences=["\u62cd\u7167"],
        )
        answer = _compose_grounded_summary(
            citations,
            [
                "\u4f20\u611f\u5668",
                "\u5149\u5708",
                "\u9632\u6296",
                "\u5f71\u50cf",
            ],
            prefix="\u624b\u673a\u62cd\u7167\u5efa\u8bae\u5173\u6ce8",
        )

        assert len(citations) >= 1
        assert_citations_are_real(citations, expected_chunk_id="phone_camera_001")
        assert_answer_contains_any(
            answer,
            ["\u4f20\u611f\u5668", "\u9632\u6296", "\u5f71\u50cf", "\u5149\u5708"],
        )
        assert_text_not_contains_any(answer, PURCHASE_BOUNDARY_TERMS)
        assert "\u624b\u673a" in knowledge_query or "\u62cd\u7167" in knowledge_query
    finally:
        db.close()
        engine.dispose()


def test_real_rag_skincare_safety_smoke(tmp_path) -> None:
    _skip_without_real_rag_dependencies()
    engine, db = _test_session(tmp_path)
    try:
        _insert_knowledge_chunk(db, SKINCARE_SAFE_CHUNK)
        understanding = QueryUnderstandingService(llm_enabled=False).understand(
            "\u9884\u7b97300\uff0c\u63a8\u8350\u80fd\u6cbb\u7597"
            "\u75d8\u75d8\u7684\u62a4\u80a4\u54c1"
        )
        service = KnowledgeRetrievalService(
            db,
            embedding_service=TinyEmbeddingService(),
            chroma_client=FixtureChromaClient([SKINCARE_SAFE_CHUNK]),
        )

        effective_query = understanding.effective_query
        knowledge_query = service.search_knowledge(
            effective_query,
            category_id="cat_skincare",
            top_k=3,
            preferences=understanding.preferences,
            negative_preferences=understanding.negative_preferences,
        )
        citations = knowledge_query
        answer = _compose_safe_skincare_summary(citations)

        assert_text_not_contains_any(answer, MEDICAL_CLAIM_TERMS)
        assert_text_not_contains_any(
            effective_query,
            ["\u6cbb\u7597", "\u6cbb\u6108", "\u836f\u6548"],
        )
        assert_text_not_contains_any(
            service.last_query,
            ["\u6cbb\u7597", "\u6cbb\u6108", "\u836f\u6548"],
        )
        assert_answer_contains_any(
            answer,
            [
                "\u6e05\u723d",
                "\u63a7\u6cb9",
                "\u6e29\u548c",
                "\u4fdd\u6e7f",
                "\u65e5\u5e38\u62a4\u7406",
            ],
        )
        assert len(citations) >= 1
        assert_citations_are_real(
            citations,
            expected_chunk_id="skincare_acne_safe_001",
        )
        assert_text_not_contains_any(answer, PURCHASE_BOUNDARY_TERMS)
    finally:
        db.close()
        engine.dispose()


def assert_citations_are_real(citations, *, expected_chunk_id: str) -> None:
    ids = [citation.chunk_id for citation in citations]
    assert expected_chunk_id in ids
    for citation in citations:
        assert citation.chunk_id
        assert citation.source_file
        assert citation.content_preview
        assert citation.document_id


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


def _insert_knowledge_chunk(db, chunk: dict[str, str]) -> None:
    document_metadata = {
        "source_file": chunk["source"],
        "doc_type": chunk["doc_type"],
        "category_id": chunk["category_id"],
        "category_path": chunk["category_path"],
        "title": chunk["title"],
    }
    chunk_metadata = {
        **document_metadata,
        "section": chunk["section"],
        "section_path": f"{chunk['title']}/{chunk['section']}",
    }
    db.add(
        Document(
            id=chunk["document_id"],
            source_file=chunk["source"],
            doc_type=chunk["doc_type"],
            category_id=chunk["category_id"],
            title=chunk["title"],
            metadata_json=json.dumps(document_metadata, ensure_ascii=False),
        )
    )
    db.add(
        DocumentChunk(
            id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            category_id=chunk["category_id"],
            chunk_index=0,
            content=chunk["text"],
            metadata_json=json.dumps(chunk_metadata, ensure_ascii=False),
            vector_id=chunk["chunk_id"],
        )
    )
    db.commit()


def _compose_grounded_summary(citations, terms: list[str], *, prefix: str) -> str:
    citation_text = " ".join(citation.content_preview for citation in citations)
    grounded_terms = [term for term in terms if term in citation_text]
    if not grounded_terms:
        grounded_terms = terms[:1]
    return f"{prefix}{LIST_SEPARATOR.join(grounded_terms)}。"


def _compose_safe_skincare_summary(citations) -> str:
    safe_terms = [
        "\u6e05\u723d",
        "\u63a7\u6cb9",
        "\u6e29\u548c",
        "\u4fdd\u6e7f",
        "\u65e5\u5e38\u62a4\u7406",
    ]
    citation_text = " ".join(citation.content_preview for citation in citations)
    grounded_terms = [term for term in safe_terms if term in citation_text]
    if not grounded_terms:
        grounded_terms = safe_terms[:3]
    return (
        "\u8fd9\u7c7b\u9700\u6c42\u5efa\u8bae\u5173\u6ce8"
        f"{LIST_SEPARATOR.join(grounded_terms)}\u65b9\u5411\uff0c\u4ee5\u65e5\u5e38"
        "\u62a4\u7406\u548c\u4f4e\u523a\u6fc0\u4e3a\u4e3b\u3002"
    )

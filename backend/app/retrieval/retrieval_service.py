from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DocumentChunk, Product, ProductAttribute, ProductTag
from app.retrieval.chroma_indexer import (
    KNOWLEDGE_COLLECTION,
    PRODUCT_COLLECTION,
    build_product_text,
    get_chroma_client,
)
from app.services.embedding import BaseEmbeddingService


@dataclass(frozen=True)
class ProductSearchFilters:
    category_id: str | None = None
    budget_max: int | None = None
    budget_min: int | None = None
    stock_only: bool = True
    brand_exclude: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProductCandidate:
    product_id: str
    title: str
    brand: str | None
    category_id: str
    price: int
    stock: int
    description: str | None
    image_url: str | None
    tags: list[str]
    attributes: dict[str, str]
    source_url: str | None
    compare_url: str | None
    distance: float | None
    score: float
    product_text: str


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    document_id: str
    title: str | None
    section: str | None
    section_path: str | None
    source_file: str | None
    doc_type: str | None
    category_id: str | None
    category_path: str | None
    content_preview: str
    distance: float | None
    score: float


def distance_to_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + distance)


def parse_json_safe(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _as_product_filters(
    filters: ProductSearchFilters | dict[str, Any] | None,
) -> ProductSearchFilters:
    if filters is None:
        return ProductSearchFilters()
    if isinstance(filters, ProductSearchFilters):
        return filters
    return ProductSearchFilters(**filters)


def _get_chroma_collection(client, name: str):
    try:
        return client.get_collection(name)
    except Exception:
        return None


def _first_result_list(query_result: dict[str, Any], key: str) -> list[Any]:
    values = query_result.get(key) or []
    if not values:
        return []
    return values[0] or []


def _preview(content: str, limit: int = 180) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."


def load_product_detail(
    db: Session,
    product_id: str,
    distance: float | None = None,
    product_text: str | None = None,
) -> ProductCandidate | None:
    product = db.get(Product, product_id)
    if product is None:
        return None

    tags = db.execute(
        select(ProductTag)
        .where(ProductTag.product_id == product.id)
        .order_by(ProductTag.id)
    ).scalars().all()
    attributes = db.execute(
        select(ProductAttribute)
        .where(ProductAttribute.product_id == product.id)
        .order_by(ProductAttribute.id)
    ).scalars().all()
    attribute_map = {
        attribute.attr_name: attribute.attr_value for attribute in attributes
    }

    return ProductCandidate(
        product_id=product.id,
        title=product.title,
        brand=product.brand,
        category_id=product.category_id,
        price=product.price,
        stock=product.stock,
        description=product.description,
        image_url=product.image_url,
        tags=[tag.value for tag in tags],
        attributes=attribute_map,
        source_url=product.source_url,
        compare_url=product.compare_url,
        distance=distance,
        score=distance_to_score(distance),
        product_text=product_text
        or build_product_text(product, tags=tags, attributes=attributes),
    )


class ProductRetrievalService:
    def __init__(
        self,
        db: Session,
        embedding_service: BaseEmbeddingService,
        chroma_client=None,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.chroma_client = chroma_client or get_chroma_client()

    def search_products(
        self,
        query: str,
        filters: ProductSearchFilters | dict[str, Any] | None = None,
        top_k: int = 3,
        semantic_top_k: int = 30,
    ) -> list[ProductCandidate]:
        product_filters = _as_product_filters(filters)
        allowed_products = self._filter_products(product_filters)
        if not allowed_products:
            return []

        allowed_product_ids = {product.id for product in allowed_products}
        candidates: list[ProductCandidate] = []
        seen_product_ids: set[str] = set()

        collection = _get_chroma_collection(self.chroma_client, PRODUCT_COLLECTION)
        collection_count = collection.count() if collection is not None else 0
        if collection_count > 0:
            query_result = collection.query(
                query_embeddings=[self.embedding_service.embed_text(query)],
                n_results=min(semantic_top_k, collection_count),
                include=["metadatas", "documents", "distances"],
            )
            candidates.extend(
                self._candidates_from_chroma_result(
                    query_result,
                    allowed_product_ids=allowed_product_ids,
                    seen_product_ids=seen_product_ids,
                    top_k=top_k,
                )
            )

        if len(candidates) < top_k:
            for product in allowed_products:
                if product.id in seen_product_ids:
                    continue
                candidate = load_product_detail(self.db, product.id)
                if candidate is None:
                    continue
                candidates.append(candidate)
                seen_product_ids.add(product.id)
                if len(candidates) >= top_k:
                    break

        return candidates[:top_k]

    def _filter_products(self, filters: ProductSearchFilters) -> list[Product]:
        statement = select(Product)
        if filters.category_id:
            statement = statement.where(Product.category_id == filters.category_id)
        if filters.budget_max is not None:
            statement = statement.where(Product.price <= filters.budget_max)
        if filters.budget_min is not None:
            statement = statement.where(Product.price >= filters.budget_min)
        if filters.stock_only:
            statement = statement.where(Product.stock > 0)
        if filters.brand_exclude:
            statement = statement.where(Product.brand.not_in(filters.brand_exclude))

        return self.db.execute(statement.order_by(Product.id)).scalars().all()

    def _candidates_from_chroma_result(
        self,
        query_result: dict[str, Any],
        allowed_product_ids: set[str],
        seen_product_ids: set[str],
        top_k: int,
    ) -> list[ProductCandidate]:
        metadatas = _first_result_list(query_result, "metadatas")
        documents = _first_result_list(query_result, "documents")
        distances = _first_result_list(query_result, "distances")
        candidates: list[ProductCandidate] = []

        for index, metadata in enumerate(metadatas):
            metadata = metadata or {}
            product_id = metadata.get("product_id")
            if not product_id or product_id not in allowed_product_ids:
                continue
            if product_id in seen_product_ids:
                continue

            distance = distances[index] if index < len(distances) else None
            product_text = documents[index] if index < len(documents) else None
            candidate = load_product_detail(
                self.db,
                product_id=product_id,
                distance=distance,
                product_text=product_text,
            )
            if candidate is None:
                continue

            candidates.append(candidate)
            seen_product_ids.add(product_id)
            if len(candidates) >= top_k:
                break

        return candidates


class KnowledgeRetrievalService:
    def __init__(
        self,
        db: Session,
        embedding_service: BaseEmbeddingService,
        chroma_client=None,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.chroma_client = chroma_client or get_chroma_client()

    def search_knowledge(
        self,
        query: str,
        category_id: str | None = None,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> list[Citation]:
        collection = _get_chroma_collection(self.chroma_client, KNOWLEDGE_COLLECTION)
        collection_count = collection.count() if collection is not None else 0
        if collection_count == 0:
            return []

        query_result = collection.query(
            query_embeddings=[self.embedding_service.embed_text(query)],
            n_results=min(top_k * 3, collection_count),
            include=["metadatas", "distances"],
        )

        metadatas = _first_result_list(query_result, "metadatas")
        distances = _first_result_list(query_result, "distances")
        citations: list[Citation] = []
        seen_chunk_ids: set[str] = set()

        for index, metadata in enumerate(metadatas):
            metadata = metadata or {}
            chunk_id = metadata.get("chunk_id")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue

            chunk = self.db.get(DocumentChunk, chunk_id)
            if chunk is None:
                continue

            chunk_metadata = parse_json_safe(chunk.metadata_json, {})
            effective_category_id = chunk_metadata.get("category_id") or metadata.get(
                "category_id"
            )
            effective_doc_type = chunk_metadata.get("doc_type") or metadata.get("doc_type")

            if category_id and effective_category_id != category_id:
                continue
            if doc_type and effective_doc_type != doc_type:
                continue

            distance = distances[index] if index < len(distances) else None
            citations.append(
                Citation(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    title=chunk_metadata.get("title"),
                    section=chunk_metadata.get("section"),
                    section_path=chunk_metadata.get("section_path"),
                    source_file=chunk_metadata.get("source_file"),
                    doc_type=effective_doc_type,
                    category_id=effective_category_id,
                    category_path=chunk_metadata.get("category_path")
                    or metadata.get("category_path"),
                    content_preview=_preview(chunk.content),
                    distance=distance,
                    score=distance_to_score(distance),
                )
            )
            seen_chunk_ids.add(chunk.id)
            if len(citations) >= top_k:
                break

        if len(citations) < top_k:
            citations.extend(
                self._fallback_citations(
                    category_id=category_id,
                    doc_type=doc_type,
                    seen_chunk_ids=seen_chunk_ids,
                    limit=top_k - len(citations),
                )
            )

        return citations

    def _fallback_citations(
        self,
        category_id: str | None,
        doc_type: str | None,
        seen_chunk_ids: set[str],
        limit: int,
    ) -> list[Citation]:
        if limit <= 0:
            return []

        fallback: list[Citation] = []
        chunks = self.db.execute(
            select(DocumentChunk).order_by(
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
            )
        ).scalars().all()

        for chunk in chunks:
            if chunk.id in seen_chunk_ids:
                continue
            chunk_metadata = parse_json_safe(chunk.metadata_json, {})
            effective_category_id = chunk_metadata.get("category_id")
            effective_doc_type = chunk_metadata.get("doc_type")
            if category_id and effective_category_id != category_id:
                continue
            if doc_type and effective_doc_type != doc_type:
                continue

            fallback.append(
                Citation(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    title=chunk_metadata.get("title"),
                    section=chunk_metadata.get("section"),
                    section_path=chunk_metadata.get("section_path"),
                    source_file=chunk_metadata.get("source_file"),
                    doc_type=effective_doc_type,
                    category_id=effective_category_id,
                    category_path=chunk_metadata.get("category_path"),
                    content_preview=_preview(chunk.content),
                    distance=None,
                    score=0.0,
                )
            )
            seen_chunk_ids.add(chunk.id)
            if len(fallback) >= limit:
                break

        return fallback

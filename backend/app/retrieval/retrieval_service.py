from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.cache.cache_service import CacheService
from app.core.config import settings
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
    preferences: list[str] = field(default_factory=list)


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


def _cache_key(namespace: str, payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"smartbuy:retrieval:{namespace}:{digest}"


def _candidate_cache_entry(candidate: ProductCandidate) -> dict[str, Any]:
    return {
        "product_id": candidate.product_id,
        "distance": candidate.distance,
        "score": candidate.score,
    }


def _candidate_from_cache_entry(
    db: Session,
    entry: dict[str, Any],
) -> ProductCandidate | None:
    product_id = entry.get("product_id")
    if not isinstance(product_id, str):
        return None
    candidate = load_product_detail(
        db,
        product_id=product_id,
        distance=_float_or_none(entry.get("distance")),
    )
    if candidate is None:
        return None
    return replace(candidate, score=_float_or_none(entry.get("score")) or candidate.score)


def _citation_to_cache(citation: Citation) -> dict[str, Any]:
    return {
        "chunk_id": citation.chunk_id,
        "document_id": citation.document_id,
        "title": citation.title,
        "section": citation.section,
        "section_path": citation.section_path,
        "source_file": citation.source_file,
        "doc_type": citation.doc_type,
        "category_id": citation.category_id,
        "category_path": citation.category_path,
        "content_preview": citation.content_preview,
        "distance": citation.distance,
        "score": citation.score,
    }


def _citation_from_cache(value: dict[str, Any]) -> Citation | None:
    chunk_id = value.get("chunk_id")
    document_id = value.get("document_id")
    if not isinstance(chunk_id, str) or not isinstance(document_id, str):
        return None
    return Citation(
        chunk_id=chunk_id,
        document_id=document_id,
        title=value.get("title") if isinstance(value.get("title"), str) else None,
        section=value.get("section") if isinstance(value.get("section"), str) else None,
        section_path=value.get("section_path")
        if isinstance(value.get("section_path"), str)
        else None,
        source_file=value.get("source_file")
        if isinstance(value.get("source_file"), str)
        else None,
        doc_type=value.get("doc_type") if isinstance(value.get("doc_type"), str) else None,
        category_id=value.get("category_id")
        if isinstance(value.get("category_id"), str)
        else None,
        category_path=value.get("category_path")
        if isinstance(value.get("category_path"), str)
        else None,
        content_preview=str(value.get("content_preview") or ""),
        distance=_float_or_none(value.get("distance")),
        score=_float_or_none(value.get("score")) or 0.0,
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


PRODUCT_PREFERENCE_KEYWORDS: dict[str, list[str]] = {
    "拍照": ["拍照", "拍照好", "影像", "像素", "主摄", "三摄", "人像", "防抖", "夜景", "OIS", "拍摄", "旅行拍摄", "社交分享"],
    "影像": ["影像", "拍照", "主摄", "三摄", "人像", "防抖", "夜景", "OIS"],
    "续航": ["续航", "续航强", "大电池", "电池", "5000mAh", "5500mAh", "6000mAh", "快充"],
    "性能": ["性能", "性能强", "游戏", "处理器", "骁龙", "天玑", "高刷", "运行内存"],
    "游戏": ["游戏", "性能", "性能强", "处理器", "高刷", "散热"],
    "轻薄": ["轻薄", "小屏", "单手", "便携"],
    "通勤": ["通勤", "通勤舒适", "舒适", "步行", "久站", "百搭", "办公室"],
    "防滑": ["防滑", "鞋底", "橡胶", "耐磨", "雨天"],
    "透气": ["透气", "网面", "飞织", "针织", "清爽"],
    "运动": ["运动", "跑步", "健走", "缓震", "支撑"],
    "材质": ["材质", "真皮", "合成革", "织物", "网面", "飞织"],
    "耐磨": ["耐磨", "耐穿", "橡胶", "鞋底"],
    "敏感肌": ["敏感肌", "温和", "低刺激", "保湿", "修护", "屏障", "舒缓"],
    "保湿": ["保湿", "补水", "滋润", "透明质酸", "泛醇", "乳液", "面霜"],
    "修护": ["修护", "屏障", "舒缓", "神经酰胺", "角鲨烷", "温和"],
    "控油": ["控油", "清爽", "油皮", "混油", "洁面", "烟酰胺"],
    "清爽": ["清爽", "控油", "轻薄", "油皮", "乳液"],
    "温和": ["温和", "敏感肌", "低刺激", "成分精简"],
}

NEGATIVE_PREFERENCE_KEYWORDS: dict[str, list[str]] = {
    "苹果": ["苹果", "apple", "iphone"],
    "三星": ["三星", "samsung"],
    "小米": ["小米", "xiaomi", "redmi", "poco"],
    "红米": ["红米", "redmi"],
    "高跟": ["高跟", "高跟鞋"],
    "美白": ["美白"],
    "太贵": ["太贵", "昂贵", "高价"],
    "厚重": ["厚重", "重"],
}

KNOWLEDGE_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "cat_phone": ["手机", "选购建议"],
    "cat_shoes": ["鞋靴", "选购建议"],
    "cat_skincare": ["护肤品", "日常护理"],
}

KNOWLEDGE_SKINCARE_UNSAFE_TERMS = [
    "治疗",
    "治愈",
    "药效",
    "处方",
    "医学修复",
    "修复疾病",
    "祛病",
]

KNOWLEDGE_SKINCARE_SAFE_TERMS = ["清爽", "控油", "温和护理", "保湿", "敏感肌"]


QUERY_KEYWORD_GROUPS: list[tuple[set[str], list[str]]] = [
    (
        {"手机", "拍照", "像素", "影像", "防抖", "夜景"},
        ["手机", "拍照", "像素", "影像", "防抖", "夜景", "主摄", "长焦"],
    ),
    (
        {"续航", "电池", "快充"},
        ["续航", "电池", "快充", "大电池", "重度用户"],
    ),
    (
        {"敏感肌", "护肤", "成分", "修护", "保湿"},
        ["敏感肌", "护肤", "温和", "成分", "修护", "保湿", "屏障"],
    ),
    (
        {"鞋", "鞋靴", "尺码", "脚宽", "脚背", "试穿"},
        ["鞋", "鞋靴", "尺码", "脚宽", "脚背", "试穿", "偏码"],
    ),
    (
        {"通勤", "防滑", "鞋底"},
        ["通勤", "防滑", "鞋底", "舒适", "耐磨", "步行"],
    ),
]


def _normalize_text(text: str | None) -> str:
    return (text or "").lower().replace(" ", "")


def _unique_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


def _keyword_match_count(text: str, keywords: list[str]) -> int:
    normalized_text = _normalize_text(text)
    return sum(
        1
        for keyword in keywords
        if _normalize_text(keyword) and _normalize_text(keyword) in normalized_text
    )


def _expand_preference_keywords(
    preferences: list[str] | None,
    query: str,
) -> list[str]:
    raw_preferences = preferences or []
    combined_text = _normalize_text(" ".join([query, *raw_preferences]))
    keywords = list(raw_preferences)

    for trigger, expanded_keywords in PRODUCT_PREFERENCE_KEYWORDS.items():
        if _normalize_text(trigger) in combined_text:
            keywords.extend(expanded_keywords)

    return _unique_ordered(keywords)


def _expand_negative_keywords(negative_preferences: list[str] | None) -> list[str]:
    keywords: list[str] = []
    for preference in negative_preferences or []:
        keywords.append(preference)
        keywords.extend(NEGATIVE_PREFERENCE_KEYWORDS.get(preference, []))
    return _unique_ordered(keywords)


def _candidate_structured_text(candidate: ProductCandidate) -> str:
    return " ".join(
        [
            candidate.product_id,
            candidate.title,
            candidate.brand or "",
            candidate.description or "",
            " ".join(candidate.tags),
            " ".join(candidate.attributes.keys()),
            " ".join(candidate.attributes.values()),
            candidate.product_text,
        ]
    )


def _matches_negative_preferences(
    candidate: ProductCandidate,
    negative_preferences: list[str] | None,
) -> bool:
    keywords = _expand_negative_keywords(negative_preferences)
    if not keywords:
        return False
    return _keyword_match_count(_candidate_structured_text(candidate), keywords) > 0


def _apply_negative_preference_filter(
    candidates: list[ProductCandidate],
    negative_preferences: list[str] | None,
) -> tuple[list[ProductCandidate], int, bool]:
    if not candidates or not negative_preferences:
        return candidates, 0, False

    matched = [
        candidate
        for candidate in candidates
        if _matches_negative_preferences(candidate, negative_preferences)
    ]
    if not matched:
        return candidates, 0, False

    filtered = [
        candidate
        for candidate in candidates
        if not _matches_negative_preferences(candidate, negative_preferences)
    ]
    if filtered:
        return filtered, len(matched), False

    penalized = [
        replace(candidate, score=round(candidate.score - 2.0, 4))
        if _matches_negative_preferences(candidate, negative_preferences)
        else candidate
        for candidate in candidates
    ]
    return penalized, len(matched), True


def _score_product_rerank(
    candidate: ProductCandidate,
    query: str,
    preferences: list[str] | None,
) -> float:
    keywords = _expand_preference_keywords(preferences, query)
    if not keywords:
        return 0.0

    tag_text = " ".join(candidate.tags)
    attribute_name_text = " ".join(candidate.attributes.keys())
    attribute_value_text = " ".join(candidate.attributes.values())
    title_description_text = " ".join(
        [
            candidate.title,
            candidate.description or "",
        ]
    )

    return (
        _keyword_match_count(tag_text, keywords) * 1.0
        + _keyword_match_count(attribute_name_text, keywords) * 0.6
        + _keyword_match_count(attribute_value_text, keywords) * 0.6
        + _keyword_match_count(title_description_text, keywords) * 0.35
    )


def _rerank_product_candidates(
    candidates: list[ProductCandidate],
    query: str,
    preferences: list[str] | None,
) -> list[ProductCandidate]:
    if not candidates:
        return []

    scored: list[tuple[int, ProductCandidate]] = []
    for index, candidate in enumerate(candidates):
        bonus = _score_product_rerank(candidate, query=query, preferences=preferences)
        scored.append(
            (
                index,
                replace(candidate, score=round(candidate.score + bonus, 4)),
            )
        )

    return [
        candidate
        for index, candidate in sorted(
            scored,
            key=lambda item: (-item[1].score, item[0]),
        )
    ]


def build_knowledge_retrieval_query(
    query: str,
    *,
    category_id: str | None = None,
    preferences: list[str] | None = None,
    negative_preferences: list[str] | None = None,
) -> str:
    parts: list[str] = []
    parts.extend(KNOWLEDGE_CATEGORY_KEYWORDS.get(category_id or "", []))
    parts.extend(preferences or [])
    parts.append(query)
    if negative_preferences:
        parts.append(f"不考虑 {' '.join(negative_preferences)}")

    structured_query = " ".join(part for part in parts if part).strip()
    if category_id == "cat_skincare":
        had_unsafe = any(
            term in structured_query for term in KNOWLEDGE_SKINCARE_UNSAFE_TERMS
        )
        for term in KNOWLEDGE_SKINCARE_UNSAFE_TERMS:
            structured_query = structured_query.replace(term, "")
        if had_unsafe:
            structured_query = " ".join(
                [structured_query, *KNOWLEDGE_SKINCARE_SAFE_TERMS]
            )
    return " ".join(structured_query.split())


def _extract_query_keywords(query: str, category_id: str | None = None) -> list[str]:
    normalized_query = _normalize_text(query)
    keywords: list[str] = []

    if category_id == "cat_phone":
        keywords.append("手机")
    elif category_id == "cat_shoes":
        keywords.extend(["鞋", "鞋靴"])
    elif category_id == "cat_skincare":
        keywords.extend(["护肤", "成分"])

    for triggers, expanded_keywords in QUERY_KEYWORD_GROUPS:
        if any(_normalize_text(trigger) in normalized_query for trigger in triggers):
            keywords.extend(expanded_keywords)

    return _unique_ordered(keywords)


def _score_citation_rerank(
    citation: Citation,
    query_keywords: list[str],
    extra_text: str = "",
) -> float:
    if not query_keywords:
        return 0.0

    citation_text = " ".join(
        [
            citation.title or "",
            citation.section or "",
            citation.section_path or "",
            citation.source_file or "",
            citation.content_preview,
            extra_text,
        ]
    )
    return _keyword_match_count(citation_text, query_keywords) * 0.4


def _with_citation_rerank_score(
    citation: Citation,
    query_keywords: list[str],
    extra_text: str = "",
) -> Citation:
    bonus = _score_citation_rerank(citation, query_keywords, extra_text=extra_text)
    return replace(citation, score=round(citation.score + bonus, 4))


def _rerank_citations(citations: list[Citation]) -> list[Citation]:
    return [
        citation
        for index, citation in sorted(
            enumerate(citations),
            key=lambda item: (-item[1].score, item[0]),
        )
    ]


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
        cache_service: CacheService | None = None,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.chroma_client = chroma_client or get_chroma_client()
        self.cache_service = cache_service
        self.last_cache_status = "disabled" if cache_service is None else "miss"
        self.last_structured_filters: dict[str, Any] = {}
        self.last_filtered_count = 0
        self.last_negative_filtered_count = 0
        self.last_negative_filter_fallback = False

    def search_products(
        self,
        query: str,
        filters: ProductSearchFilters | dict[str, Any] | None = None,
        top_k: int = 3,
        semantic_top_k: int = 30,
    ) -> list[ProductCandidate]:
        product_filters = _as_product_filters(filters)
        self.last_structured_filters = {
            "category_id": product_filters.category_id,
            "budget_min": product_filters.budget_min,
            "budget_max": product_filters.budget_max,
            "preferences": list(product_filters.preferences),
            "negative_preferences": list(product_filters.brand_exclude),
            "stock_only": product_filters.stock_only,
        }
        self.last_filtered_count = 0
        self.last_negative_filtered_count = 0
        self.last_negative_filter_fallback = False
        cache_key = _cache_key(
            "products",
            {
                "query": query,
                "category_id": product_filters.category_id,
                "budget_min": product_filters.budget_min,
                "budget_max": product_filters.budget_max,
                "stock_only": product_filters.stock_only,
                "brand_exclude": product_filters.brand_exclude,
                "preferences": product_filters.preferences,
                "top_k": top_k,
                "semantic_top_k": semantic_top_k,
            },
        )
        cached_candidates = self._get_cached_candidates(cache_key)
        if cached_candidates is not None:
            self.last_filtered_count = len(cached_candidates)
            return cached_candidates

        allowed_products = self._filter_products(product_filters)
        if not allowed_products:
            self._set_cached_candidates(cache_key, [])
            return []

        allowed_product_ids = {product.id for product in allowed_products}
        candidates: list[ProductCandidate] = []
        seen_product_ids: set[str] = set()
        candidate_limit = max(top_k * 3, top_k + 5)

        collection = _get_chroma_collection(self.chroma_client, PRODUCT_COLLECTION)
        collection_count = collection.count() if collection is not None else 0
        if collection_count > 0:
            chroma_limit = min(max(semantic_top_k, candidate_limit), collection_count)
            query_result = collection.query(
                query_embeddings=[self.embedding_service.embed_text(query)],
                n_results=chroma_limit,
                include=["metadatas", "documents", "distances"],
            )
            candidates.extend(
                self._candidates_from_chroma_result(
                    query_result,
                    allowed_product_ids=allowed_product_ids,
                    seen_product_ids=seen_product_ids,
                    top_k=chroma_limit,
                )
            )

        if len(candidates) < candidate_limit:
            for product in allowed_products:
                if product.id in seen_product_ids:
                    continue
                candidate = load_product_detail(self.db, product.id)
                if candidate is None:
                    continue
                candidates.append(candidate)
                seen_product_ids.add(product.id)
                if len(candidates) >= candidate_limit:
                    break

        candidates, negative_filtered_count, negative_filter_fallback = (
            _apply_negative_preference_filter(
                candidates,
                product_filters.brand_exclude,
            )
        )
        self.last_negative_filtered_count = negative_filtered_count
        self.last_negative_filter_fallback = negative_filter_fallback

        results = _rerank_product_candidates(
            candidates,
            query=query,
            preferences=product_filters.preferences,
        )[:top_k]
        self.last_filtered_count = len(results)
        self._set_cached_candidates(cache_key, results)
        return results

    def _get_cached_candidates(
        self,
        cache_key: str,
    ) -> list[ProductCandidate] | None:
        if self.cache_service is None:
            self.last_cache_status = "disabled"
            return None
        try:
            cached = self.cache_service.get_json(cache_key)
        except Exception:
            self.last_cache_status = "failed"
            return None
        if not isinstance(cached, list):
            self.last_cache_status = "miss"
            return None

        candidates: list[ProductCandidate] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            candidate = _candidate_from_cache_entry(self.db, item)
            if candidate is not None:
                candidates.append(candidate)
        self.last_cache_status = "hit"
        return candidates

    def _set_cached_candidates(
        self,
        cache_key: str,
        candidates: list[ProductCandidate],
    ) -> None:
        if self.cache_service is None:
            return
        try:
            self.cache_service.set_json(
                cache_key,
                [_candidate_cache_entry(candidate) for candidate in candidates],
                ttl_seconds=settings.RETRIEVAL_PRODUCT_CACHE_TTL_SECONDS,
            )
        except Exception:
            self.last_cache_status = "failed"

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
            statement = statement.where(
                or_(Product.brand.is_(None), Product.brand.not_in(filters.brand_exclude))
            )

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
        cache_service: CacheService | None = None,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.chroma_client = chroma_client or get_chroma_client()
        self.cache_service = cache_service
        self.last_cache_status = "disabled" if cache_service is None else "miss"
        self.last_query = ""

    def search_knowledge(
        self,
        query: str,
        category_id: str | None = None,
        doc_type: str | None = None,
        top_k: int = 5,
        preferences: list[str] | None = None,
        negative_preferences: list[str] | None = None,
    ) -> list[Citation]:
        structured_query = build_knowledge_retrieval_query(
            query,
            category_id=category_id,
            preferences=preferences,
            negative_preferences=negative_preferences,
        )
        self.last_query = structured_query
        cache_key = _cache_key(
            "knowledge",
            {
                "query": structured_query,
                "category_id": category_id,
                "doc_type": doc_type,
                "preferences": preferences or [],
                "negative_preferences": negative_preferences or [],
                "top_k": top_k,
            },
        )
        cached_citations = self._get_cached_citations(cache_key)
        if cached_citations is not None:
            return cached_citations

        collection = _get_chroma_collection(self.chroma_client, KNOWLEDGE_COLLECTION)
        collection_count = collection.count() if collection is not None else 0
        if collection_count == 0:
            return []

        query_keywords = _extract_query_keywords(structured_query, category_id=category_id)
        internal_n_results = min(max(top_k * 8, top_k + 20), collection_count)
        query_result = collection.query(
            query_embeddings=[self.embedding_service.embed_text(structured_query)],
            n_results=internal_n_results,
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
            citation = Citation(
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
            citations.append(
                _with_citation_rerank_score(
                    citation,
                    query_keywords=query_keywords,
                    extra_text=chunk.content,
                )
            )
            seen_chunk_ids.add(chunk.id)

        if len(citations) < top_k:
            citations.extend(
                self._fallback_citations(
                    category_id=category_id,
                    doc_type=doc_type,
                    query_keywords=query_keywords,
                    seen_chunk_ids=seen_chunk_ids,
                    limit=top_k - len(citations),
                )
            )

        results = _rerank_citations(citations)[:top_k]
        self._set_cached_citations(cache_key, results)
        return results

    def _get_cached_citations(self, cache_key: str) -> list[Citation] | None:
        if self.cache_service is None:
            self.last_cache_status = "disabled"
            return None
        try:
            cached = self.cache_service.get_json(cache_key)
        except Exception:
            self.last_cache_status = "failed"
            return None
        if not isinstance(cached, list):
            self.last_cache_status = "miss"
            return None

        citations = [
            citation
            for item in cached
            if isinstance(item, dict)
            for citation in [_citation_from_cache(item)]
            if citation is not None
        ]
        self.last_cache_status = "hit"
        return citations

    def _set_cached_citations(
        self,
        cache_key: str,
        citations: list[Citation],
    ) -> None:
        if self.cache_service is None:
            return
        try:
            self.cache_service.set_json(
                cache_key,
                [_citation_to_cache(citation) for citation in citations],
                ttl_seconds=settings.RETRIEVAL_KNOWLEDGE_CACHE_TTL_SECONDS,
            )
        except Exception:
            self.last_cache_status = "failed"

    def _fallback_citations(
        self,
        category_id: str | None,
        doc_type: str | None,
        query_keywords: list[str],
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

            citation = Citation(
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
            fallback.append(
                _with_citation_rerank_score(
                    citation,
                    query_keywords=query_keywords,
                    extra_text=chunk.content,
                )
            )
            seen_chunk_ids.add(chunk.id)
            if len(fallback) >= limit:
                break

        return fallback

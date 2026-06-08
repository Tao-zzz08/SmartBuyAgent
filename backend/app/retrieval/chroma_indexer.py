from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import chromadb
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import (
    Category,
    DocumentChunk,
    Product,
    ProductAttribute,
    ProductTag,
)
from app.services.embedding import BaseEmbeddingService


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRODUCT_COLLECTION = "product_text"
KNOWLEDGE_COLLECTION = "knowledge_docs"


def _resolve_chroma_dir(chroma_dir: str | Path | None = None) -> Path:
    raw_path = Path(chroma_dir or settings.CHROMA_DIR)
    if raw_path.is_absolute():
        resolved = raw_path
    else:
        resolved = PROJECT_ROOT / raw_path

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_chroma_client(chroma_dir: str | Path | None = None):
    chroma_path = _resolve_chroma_dir(chroma_dir)
    return chromadb.PersistentClient(path=str(chroma_path))


def get_or_create_collection(client, name: str):
    return client.get_or_create_collection(name=name)


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = json.dumps(value, ensure_ascii=False)
    return cleaned


def _category_paths(db: Session) -> dict[str, str]:
    categories = db.execute(select(Category)).scalars().all()
    by_id = {category.id: category for category in categories}

    def path_for(category_id: str) -> str:
        names: list[str] = []
        current = by_id.get(category_id)
        while current is not None:
            names.append(current.name)
            current = by_id.get(current.parent_id) if current.parent_id else None
        return "/".join(reversed(names))

    return {category_id: path_for(category_id) for category_id in by_id}


def build_product_text(
    product: Product,
    tags: list[ProductTag],
    attributes: list[ProductAttribute],
    category_path: str | None = None,
) -> str:
    tag_text = ", ".join(tag.value for tag in tags) or "无"
    attribute_lines = [
        f"- {attribute.attr_name}：{attribute.attr_value}" for attribute in attributes
    ]
    attributes_text = "\n".join(attribute_lines) if attribute_lines else "- 无"

    return "\n".join(
        [
            f"商品：{product.title}",
            f"品牌：{product.brand or '未知'}",
            f"品类：{category_path or product.category_id}",
            f"价格：{product.price}元",
            f"库存：{product.stock}",
            f"描述：{product.description or ''}",
            f"标签：{tag_text}",
            "属性：",
            attributes_text,
        ]
    ).strip()


def index_products(
    db: Session,
    embedding_service: BaseEmbeddingService,
    client=None,
) -> dict[str, int | str]:
    client = client or get_chroma_client()
    collection = get_or_create_collection(client, PRODUCT_COLLECTION)
    category_paths = _category_paths(db)
    products = db.execute(select(Product).order_by(Product.id)).scalars().all()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for product in products:
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
        category_path = category_paths.get(product.category_id)

        ids.append(f"product_{product.id}")
        documents.append(
            build_product_text(
                product,
                tags=tags,
                attributes=attributes,
                category_path=category_path,
            )
        )
        metadatas.append(
            _clean_metadata(
                {
                    "product_id": product.id,
                    "category_id": product.category_id,
                    "title": product.title,
                    "brand": product.brand,
                    "price": product.price,
                    "source": PRODUCT_COLLECTION,
                }
            )
        )

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embedding_service.embed_texts(documents),
            documents=documents,
            metadatas=metadatas,
        )

    return {"indexed_products": len(ids), "collection": PRODUCT_COLLECTION}


def index_knowledge_docs(
    db: Session,
    embedding_service: BaseEmbeddingService,
    client=None,
) -> dict[str, int | str]:
    client = client or get_chroma_client()
    collection = get_or_create_collection(client, KNOWLEDGE_COLLECTION)
    chunks = db.execute(
        select(DocumentChunk).order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
    ).scalars().all()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for chunk in chunks:
        vector_id = f"vector_{chunk.id}"
        chunk.vector_id = vector_id
        metadata = json.loads(chunk.metadata_json or "{}")
        metadata.update(
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "source": KNOWLEDGE_COLLECTION,
            }
        )

        ids.append(vector_id)
        documents.append(chunk.content)
        metadatas.append(_clean_metadata(metadata))

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embedding_service.embed_texts(documents),
            documents=documents,
            metadatas=metadatas,
        )

    db.commit()
    return {"indexed_chunks": len(ids), "collection": KNOWLEDGE_COLLECTION}


def _delete_collection_if_exists(client, name: str) -> None:
    try:
        client.delete_collection(name=name)
    except Exception as exc:
        if "does not exist" not in str(exc).lower():
            raise


def rebuild_all_indexes(
    db: Session,
    embedding_service: BaseEmbeddingService,
    reset: bool = True,
    client=None,
) -> dict[str, Any]:
    client = client or get_chroma_client()
    if reset:
        _delete_collection_if_exists(client, PRODUCT_COLLECTION)
        _delete_collection_if_exists(client, KNOWLEDGE_COLLECTION)

    product_stats = index_products(db, embedding_service, client=client)
    knowledge_stats = index_knowledge_docs(db, embedding_service, client=client)

    return {
        "product_text": product_stats,
        "knowledge_docs": knowledge_stats,
        "collections": [PRODUCT_COLLECTION, KNOWLEDGE_COLLECTION],
    }

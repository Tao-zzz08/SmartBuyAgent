from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from app.core.db import Base


class CreatedAtMixin:
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class TimestampMixin(CreatedAtMixin):
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class Category(TimestampMixin, Base):
    __tablename__ = "categories"

    id = Column(String(64), primary_key=True, index=True)
    parent_id = Column(String(64), nullable=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    level = Column(Integer, nullable=False, default=1)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)


class CategoryAttributeDef(Base):
    __tablename__ = "category_attribute_defs"

    id = Column(String(64), primary_key=True, index=True)
    category_id = Column(String(64), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    value_type = Column(String(64), nullable=False, default="string")
    is_required = Column(Boolean, nullable=False, default=False)
    is_filterable = Column(Boolean, nullable=False, default=True, index=True)
    is_recommend_factor = Column(Boolean, nullable=False, default=False, index=True)
    display_order = Column(Integer, nullable=False, default=0)


class CategoryProfile(TimestampMixin, Base):
    __tablename__ = "category_profiles"

    id = Column(String(64), primary_key=True, index=True)
    category_id = Column(String(64), nullable=False, index=True)
    profile_json = Column(Text, nullable=False)


class Product(TimestampMixin, Base):
    __tablename__ = "products"

    id = Column(String(64), primary_key=True, index=True)
    category_id = Column(String(64), nullable=False, index=True)
    title = Column(String(255), nullable=False, index=True)
    brand = Column(String(255), nullable=True, index=True)
    price = Column(Integer, nullable=False)
    stock = Column(Integer, nullable=False, default=0)
    description = Column(Text, nullable=True)
    image_url = Column(Text, nullable=True)
    rating = Column(Float, nullable=False, default=0.0)
    sales = Column(Integer, nullable=False, default=0)
    source_platform = Column(String(128), nullable=True, index=True)
    source_url = Column(Text, nullable=True)
    compare_url = Column(Text, nullable=True)
    external_links_json = Column(Text, nullable=False, default="[]")
    status = Column(String(32), nullable=False, default="active", index=True)


class ProductAttribute(CreatedAtMixin, Base):
    __tablename__ = "product_attributes"

    id = Column(String(64), primary_key=True, index=True)
    product_id = Column(String(64), nullable=False, index=True)
    attr_name = Column(String(255), nullable=False, index=True)
    attr_value = Column(Text, nullable=False)
    attr_value_number = Column(Float, nullable=True, index=True)


class ProductTag(Base):
    __tablename__ = "product_tags"

    id = Column(String(64), primary_key=True, index=True)
    product_id = Column(String(64), nullable=False, index=True)
    tag_type = Column(String(64), nullable=False, default="tag", index=True)
    value = Column(String(255), nullable=False, index=True)


class Document(CreatedAtMixin, Base):
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, index=True)
    source_file = Column(Text, nullable=False)
    doc_type = Column(String(64), nullable=False, index=True)
    category_id = Column(String(64), nullable=True, index=True)
    product_id = Column(String(64), nullable=True, index=True)
    title = Column(String(255), nullable=True)
    metadata_json = Column(Text, nullable=False, default="{}")


class DocumentChunk(CreatedAtMixin, Base):
    __tablename__ = "document_chunks"

    id = Column(String(64), primary_key=True, index=True)
    document_id = Column(String(64), nullable=False, index=True)
    category_id = Column(String(64), nullable=True, index=True)
    product_id = Column(String(64), nullable=True, index=True)
    chunk_index = Column(Integer, nullable=False, index=True)
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=False, default="{}")
    vector_id = Column(String(255), nullable=True, index=True)


class Session(TimestampMixin, Base):
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(128), nullable=False, default="demo_user", index=True)
    title = Column(String(255), nullable=True)
    memory_json = Column(Text, nullable=False, default="{}")


class Message(CreatedAtMixin, Base):
    __tablename__ = "messages"

    id = Column(String(64), primary_key=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(32), nullable=False, index=True)
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=False, default="{}")


class RetrievalLog(CreatedAtMixin, Base):
    __tablename__ = "retrieval_logs"

    id = Column(String(64), primary_key=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    query = Column(Text, nullable=False)
    intent = Column(String(128), nullable=True, index=True)
    collection_name = Column(String(255), nullable=True, index=True)
    filters_json = Column(Text, nullable=False, default="{}")
    candidates_json = Column(Text, nullable=False, default="[]")


class RecommendationLog(CreatedAtMixin, Base):
    __tablename__ = "recommendation_logs"

    id = Column(String(64), primary_key=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    message_id = Column(String(64), nullable=True, index=True)
    query = Column(Text, nullable=True)
    products_json = Column(Text, nullable=False, default="[]")
    reason_json = Column(Text, nullable=False, default="{}")


class Feedback(CreatedAtMixin, Base):
    __tablename__ = "feedback"

    id = Column(String(64), primary_key=True, index=True)
    message_id = Column(String(64), nullable=False, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    rating = Column(Integer, nullable=False, index=True)
    reason = Column(Text, nullable=True)

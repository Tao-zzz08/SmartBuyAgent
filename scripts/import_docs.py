from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import delete, func, select  # noqa: E402
from sqlalchemy.orm import Session as DbSession  # noqa: E402

from app.core.db import Base, SessionLocal, engine  # noqa: E402
from app.models import Document, DocumentChunk  # noqa: E402
import app.models  # noqa: E402,F401


REQUIRED_FRONT_MATTER_FIELDS = {"title", "doc_type", "category_id", "category_path"}


class DocumentImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarkdownSection:
    section: str
    section_path: str
    content: str


def load_markdown_files(
    root_dir: str | Path,
    docs_dir: str | Path = "data/knowledge_docs",
) -> list[Path]:
    docs_path = Path(root_dir) / docs_dir
    return sorted(docs_path.rglob("*.md"))


def split_front_matter(markdown_text: str) -> tuple[str, str]:
    normalized = markdown_text.lstrip("\ufeff")
    match = re.match(
        r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)(.*)$",
        normalized,
        re.DOTALL,
    )
    if match is None:
        raise DocumentImportError("Markdown file is missing YAML front matter.")

    front_matter_text = match.group(1).strip()
    body_text = match.group(2).lstrip("\r\n")
    return front_matter_text, body_text


def _parse_scalar(value: str) -> str | None:
    stripped = value.strip().strip('"').strip("'")
    if stripped.lower() in {"null", "none", "~"}:
        return None
    return stripped


def parse_front_matter(markdown_text: str) -> dict[str, Any]:
    front_matter_text, _ = split_front_matter(markdown_text)
    metadata: dict[str, Any] = {}

    for line in front_matter_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise DocumentImportError(f"Invalid front matter line: {line}")

        key, value = stripped.split(":", 1)
        metadata[key.strip()] = _parse_scalar(value)

    missing = [
        field
        for field in sorted(REQUIRED_FRONT_MATTER_FIELDS)
        if field not in metadata
    ]
    if missing:
        raise DocumentImportError(
            "Markdown front matter is missing required fields: "
            + ", ".join(missing)
        )

    return metadata


def strip_front_matter(markdown_text: str) -> str:
    _, body_text = split_front_matter(markdown_text)
    return body_text


def _heading_level(line: str) -> int | None:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    if match is None:
        return None
    return len(match.group(1))


def _heading_title(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line).strip()


def split_markdown_sections(body_text: str) -> list[MarkdownSection]:
    lines = body_text.strip().splitlines()
    root_title = ""
    heading_path: list[str] = []
    current_section = ""
    current_section_path = ""
    current_lines: list[str] = []
    sections: list[MarkdownSection] = []
    preface_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            if root_title and not content.startswith("# "):
                content = f"# {root_title}\n\n{content}"
            sections.append(
                MarkdownSection(
                    section=current_section or root_title or "正文",
                    section_path=current_section_path or root_title or "正文",
                    content=content,
                )
            )
        current_lines = []

    for line in lines:
        level = _heading_level(line)
        if level is None:
            if current_lines:
                current_lines.append(line)
            else:
                preface_lines.append(line)
            continue

        title = _heading_title(line)
        if level == 1 and not root_title:
            root_title = title
            heading_path = [title]
            preface_lines.append(line)
            continue

        if level >= 2:
            flush_section()
            heading_path = heading_path[: level - 1]
            while len(heading_path) < level - 1:
                heading_path.append(root_title or title)
            heading_path.append(title)
            current_section = title
            current_section_path = " / ".join(
                [part for part in heading_path if part]
            )
            current_lines = [line]
            continue

        flush_section()
        root_title = title
        heading_path = [title]
        current_section = title
        current_section_path = title
        current_lines = [line]

    flush_section()

    preface = "\n".join(preface_lines).strip()
    if preface and not sections:
        sections.append(
            MarkdownSection(
                section=root_title or "正文",
                section_path=root_title or "正文",
                content=preface,
            )
        )

    return sections


def chunk_section(
    section_text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    text = section_text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step

    if len(chunks) > 1 and len(chunks[-1]) < 120:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}".strip()
        chunks.pop()

    return chunks


def build_document_id(relative_path: str | Path) -> str:
    path = Path(relative_path)
    parts = list(path.with_suffix("").parts)
    if "knowledge_docs" in parts:
        parts = parts[parts.index("knowledge_docs") + 1 :]
    slug = "_".join(parts)
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", slug).strip("_").lower()
    return f"doc_{slug}"


def _document_record(
    document_id: str,
    source_file: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": document_id,
        "source_file": source_file,
        "doc_type": metadata["doc_type"],
        "category_id": metadata["category_id"],
        "product_id": None,
        "title": metadata["title"],
        "metadata_json": json.dumps(metadata, ensure_ascii=False, indent=2),
    }


def _upsert_document(db: DbSession, record: dict[str, Any]) -> str:
    existing = db.get(Document, record["id"])
    if existing is None:
        db.add(Document(**record))
        return "inserted"

    for key, value in record.items():
        setattr(existing, key, value)
    return "updated"


def import_documents(
    db: DbSession,
    root_dir: str | Path = PROJECT_ROOT,
    docs_dir: str | Path = "data/knowledge_docs",
    chunk_size: int = 800,
    overlap: int = 120,
) -> dict[str, int]:
    root = Path(root_dir)
    stats = {
        "documents_inserted": 0,
        "documents_updated": 0,
        "document_chunks_inserted": 0,
        "total_documents": 0,
        "total_chunks": 0,
    }

    for path in load_markdown_files(root, docs_dir):
        markdown_text = path.read_text(encoding="utf-8")
        metadata = parse_front_matter(markdown_text)
        body_text = strip_front_matter(markdown_text)
        source_file = path.relative_to(root).as_posix()
        document_id = build_document_id(source_file)
        record = _document_record(document_id, source_file, metadata)

        result = _upsert_document(db, record)
        stats[f"documents_{result}"] += 1

        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))

        chunk_index = 1
        for section in split_markdown_sections(body_text):
            for chunk_content in chunk_section(section.content, chunk_size, overlap):
                chunk_metadata = {
                    "title": metadata["title"],
                    "doc_type": metadata["doc_type"],
                    "category_id": metadata["category_id"],
                    "category_path": metadata["category_path"],
                    "source_file": source_file,
                    "section": section.section,
                    "section_path": section.section_path,
                    "chunk_index": chunk_index,
                }
                db.add(
                    DocumentChunk(
                        id=f"{document_id}_chunk_{chunk_index:03d}",
                        document_id=document_id,
                        category_id=metadata["category_id"],
                        product_id=None,
                        chunk_index=chunk_index,
                        content=chunk_content,
                        metadata_json=json.dumps(
                            chunk_metadata,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        vector_id=None,
                    )
                )
                stats["document_chunks_inserted"] += 1
                chunk_index += 1

    db.commit()
    stats["total_documents"] = (
        db.scalar(select(func.count()).select_from(Document)) or 0
    )
    stats["total_chunks"] = (
        db.scalar(select(func.count()).select_from(DocumentChunk)) or 0
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Markdown knowledge documents into SQLite."
    )
    parser.add_argument(
        "--docs-dir",
        default="data/knowledge_docs",
        help="Markdown docs directory relative to project root.",
    )
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        stats = import_documents(db, PROJECT_ROOT, docs_dir=args.docs_dir)
    except DocumentImportError as exc:
        db.rollback()
        raise SystemExit(str(exc)) from exc
    finally:
        db.close()

    print(
        "documents inserted: "
        f"{stats['documents_inserted']}, updated: {stats['documents_updated']}"
    )
    print(f"document_chunks inserted: {stats['document_chunks_inserted']}")
    print(f"total documents: {stats['total_documents']}")
    print(f"total chunks: {stats['total_chunks']}")


if __name__ == "__main__":
    main()

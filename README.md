# SmartBuyAgent

SmartBuyAgent 是一个面向多品类全新商品的电商智能导购 RAG Agent 系统。MVP 后续会支持商品导入、商品知识库 RAG、LangGraph Agent 编排、SSE 流式输出、商品卡片、Web Debug 和 Web Showcase。

当前已完成阶段 1 的数据库模型、基础分类导入和 mini 商品导入。阶段 2 已新增 17 份 Markdown 知识文档，支持将 Markdown 知识文档导入 `documents` / `document_chunks` 表，并支持使用 mock `EmbeddingService` 将商品文本和文档 chunk 写入 Chroma 的 `product_text` / `knowledge_docs` collection。已新增基础 `RetrievalService`，支持商品候选召回和知识文档 citation 召回。阶段 3 已新增规则版 `QueryUnderstandingService`、模板版 `ResponseComposer` 和最小 `ChatService`，用于串联 QueryUnderstanding、RetrievalService 和 ResponseComposer，完成一次最小导购回答闭环。当前还未接入真实 bge-m3 embedding。

## Start Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run Backend Tests

```bash
cd backend
pytest
```

## Initialize Database

```bash
cd backend
python ../scripts/init_db.py
```

## Import Category Seed Data

```bash
cd backend
python ../scripts/init_db.py
python ../scripts/import_categories.py
```

## Import Mini Product Seed Data

```bash
cd backend
python ../scripts/init_db.py
python ../scripts/import_categories.py
python ../scripts/import_products.py --dataset mini
```

## Import Knowledge Documents

```bash
cd backend
python ../scripts/import_docs.py
```

## Rebuild Chroma Index

```bash
cd backend
python ../scripts/init_db.py
python ../scripts/import_categories.py
python ../scripts/import_products.py --dataset mini
python ../scripts/import_docs.py
python ../scripts/rebuild_index.py
```

## Knowledge Documents

Markdown seed documents are stored under `data/knowledge_docs/`:

- `phone/`: phone buying, camera, battery, performance, and FAQ guides.
- `shoes/`: commute, size, material, care, and FAQ guides.
- `skincare/`: sensitive skin, moisturizing, ingredients, usage, and FAQ guides.
- `common/`: after-sales policy and shopping guide tone documents.

## Start Frontend

```bash
cd frontend
npm install
npm run dev
```

## Not Implemented Yet

- 300 条 demo 商品数据
- 真实 embedding
- ChatService
- `/api/chat`
- LLM 回答生成
- LangGraph Agent
- SSE
- Web Debug
- Web Showcase

## Next Stage

阶段 3 后续任务：基础 `/api/chat` 接口，之后再升级 SSE、Web Debug 和 LangGraph Agent。

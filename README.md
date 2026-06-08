# SmartBuyAgent

SmartBuyAgent 是一个面向多品类全新商品的电商智能导购 RAG Agent 系统。MVP 后续会支持商品导入、商品知识库 RAG、LangGraph Agent 编排、SSE 流式输出、商品卡片、Web Debug 和 Web Showcase。

当前已完成阶段 1 的数据库模型、基础分类导入和 mini 商品导入。已新增 17 份 Markdown 知识文档，作为后续 RAG 文档入库的数据源。

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
- 文档导入
- Chroma 索引
- RAG
- Agent
- SSE
- Web Debug
- Web Showcase

## Next Stage

阶段 2 后续任务：文档入库脚本、chunk 切分、embedding 与 Chroma 索引准备。

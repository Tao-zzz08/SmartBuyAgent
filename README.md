# SmartBuyAgent

SmartBuyAgent 是一个面向多品类全新商品的电商智能导购 RAG Agent 系统。MVP 后续会支持商品导入、商品知识库 RAG、LangGraph Agent 编排、SSE 流式输出、商品卡片、Web Debug 和 Web Showcase。

当前已完成阶段 0 工程骨架，并进入阶段 1：数据库模型与种子数据导入。当前代码包含 FastAPI 后端骨架、React 前端骨架、SQLite + SQLAlchemy 数据库模型、数据库初始化脚本，以及基础分类 / 品类配置导入脚本。

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

## Start Frontend

```bash
cd frontend
npm install
npm run dev
```

## Not Implemented Yet

- 商品 CSV 导入
- RAG
- Agent
- SSE
- Web Debug
- Web Showcase

## Next Stage

阶段 1 后续任务：商品种子数据导入、文档入库准备和检索基础设施。

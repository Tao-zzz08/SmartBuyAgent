# SmartBuyAgent

SmartBuyAgent 是一个面向多品类全新商品的电商智能导购 RAG Agent 系统。MVP 后续会支持商品导入、商品知识库 RAG、LangGraph Agent 编排、SSE 流式输出、商品卡片、Web Debug 和 Web Showcase。

当前阶段是阶段 0：工程骨架。本阶段只提供前后端最小可运行结构、配置、SQLite 连接、日志和健康检查接口。

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

## Start Frontend

```bash
cd frontend
npm install
npm run dev
```

## Not Implemented In Stage 0

- Agent
- RAG
- 商品导入
- 文档导入
- SSE
- Web Debug
- Web Showcase

## Next Stage

阶段 1：数据库模型 + 种子数据导入。

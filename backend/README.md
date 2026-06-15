# SmartBuyAgent Backend

FastAPI backend for the SmartBuyAgent RAG + Agent shopping-guide prototype.

## Structure

- `app/main.py`: FastAPI application entry point.
- `app/api/`: API routers for health, chat, chat stream, and feedback.
- `app/core/`: configuration, database connection, and logging.
- `app/models/`: SQLAlchemy models for products, documents, chat memory, and feedback.
- `app/schemas/`: Pydantic request and response schemas.
- `app/services/`: embedding and LLM provider abstractions.
- `app/retrieval/`: Chroma indexer plus product and knowledge retrieval services.
- `app/chat/`: query understanding, response composition, memory, follow-up rewrite, product comparison, and ChatService.
- `app/agent/`: AgentState, AgentRuntimeContext, executable nodes, and LangGraph AgentWorkflow.
- `tests/`: backend tests.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Initialize Data

Run from the `backend` directory:

```bash
python ../scripts/init_db.py
python ../scripts/import_categories.py
python ../scripts/import_products.py --dataset mini
python ../scripts/import_docs.py
python ../scripts/rebuild_index.py
```

## Test

```bash
pytest
```

## Boundaries

The backend provides guidance, retrieval, comparison, trace, memory, and feedback APIs. It does not implement login, cart, orders, payment, fulfillment, or purchase actions.

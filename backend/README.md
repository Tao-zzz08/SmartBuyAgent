# SmartBuyAgent Backend

FastAPI backend scaffold for SmartBuyAgent stage 0.

## Structure

- `app/main.py`: FastAPI application entry point.
- `app/api/`: API routers.
- `app/core/`: configuration, database connection, and logging.
- `app/models/`, `app/schemas/`, `app/services/`, `app/retrieval/`, `app/agents/`: reserved extension points for later stages.
- `tests/`: backend tests.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Initialize Database

```bash
python ../scripts/init_db.py
```

## Test

```bash
pytest
```

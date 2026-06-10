import json
from pathlib import Path
import shutil
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.context import AgentRuntimeContext
from app.agent.workflow import AgentWorkflow
from app.chat.conversation_memory import ConversationMemoryService
from app.chat.followup_rewriter import FollowUpQueryRewriter
from app.chat.product_comparison import ProductComparisonService
from app.chat.query_understanding import QueryUnderstandingService
from app.chat.response_composer import ResponseComposer
from app.core.db import Base
from app.models import ChatSession, ChatTurn
from app.retrieval.chroma_indexer import get_chroma_client, rebuild_all_indexes
from app.retrieval.retrieval_service import (
    KnowledgeRetrievalService,
    ProductRetrievalService,
)
from app.services.embedding import MockEmbeddingService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from import_categories import import_seed_data  # noqa: E402
from import_docs import import_documents  # noqa: E402
from import_products import import_products  # noqa: E402


@pytest.fixture()
def workflow_stack():
    db_path = PROJECT_ROOT / "data" / "smartbuy_agent_workflow_test.db"
    chroma_dir = PROJECT_ROOT / "data" / "chroma_agent_workflow_test"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    shutil.rmtree(chroma_dir, ignore_errors=True)

    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()

    import_seed_data(db, PROJECT_ROOT)
    import_products(db, PROJECT_ROOT, dataset="mini")
    import_documents(db, PROJECT_ROOT)

    embedding_service = MockEmbeddingService()
    chroma_client = get_chroma_client(chroma_dir)
    rebuild_all_indexes(
        db,
        embedding_service=embedding_service,
        reset=True,
        client=chroma_client,
    )

    context = AgentRuntimeContext(
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
        query_understanding_service=QueryUnderstandingService(),
        product_retrieval_service=ProductRetrievalService(
            db=db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        ),
        knowledge_retrieval_service=KnowledgeRetrievalService(
            db=db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        ),
        response_composer=ResponseComposer(),
        conversation_memory_service=ConversationMemoryService(db),
        followup_rewriter=FollowUpQueryRewriter(),
        product_comparison_service=ProductComparisonService(db),
    )

    try:
        yield db, AgentWorkflow(context)
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)


def _steps(state) -> set[str]:
    return {step["step"] for step in state.trace}


def _agent_nodes(state) -> set[str]:
    return {
        step["node"]
        for step in state.trace
        if step.get("step") == "agent_node" and "node" in step
    }


def test_agent_workflow_shopping_guide(workflow_stack) -> None:
    _db, workflow = workflow_stack

    state = workflow.run("预算3000，推荐一款拍照好的手机")

    assert state.intent == "shopping_guide"
    assert state.product_candidates
    assert state.citations
    assert state.product_cards
    assert state.answer
    assert "load_context" in _agent_nodes(state)
    assert "route_by_intent" in _agent_nodes(state)
    assert "save_trace" in _agent_nodes(state)
    assert "follow_up_rewrite" in _steps(state)
    assert "query_understanding" in _steps(state)
    assert "product_retrieval" in _steps(state)
    assert "knowledge_retrieval" in _steps(state)
    assert "response_composer" in _steps(state)


def test_agent_workflow_product_knowledge(workflow_stack) -> None:
    _db, workflow = workflow_stack

    state = workflow.run("为什么手机拍照不能只看像素")

    assert state.intent == "product_knowledge"
    assert state.citations
    assert state.product_candidates == []
    assert state.answer
    assert "knowledge_retrieval" in _steps(state)
    assert "response_composer" in _steps(state)


def test_agent_workflow_clarification(workflow_stack) -> None:
    _db, workflow = workflow_stack

    state = workflow.run("推荐一下")

    assert state.need_clarification is True
    assert state.answer is not None
    assert "你想看哪个品类" in state.answer
    assert state.product_cards == []
    assert "clarification" in _agent_nodes(state)
    assert "save_trace" in _agent_nodes(state)


def test_agent_workflow_follow_up_compare(workflow_stack) -> None:
    db, workflow = workflow_stack
    session_id = "session_agent_workflow"
    db.add(ChatSession(session_id=session_id))
    db.add(
        ChatTurn(
            session_id=session_id,
            turn_index=1,
            user_query="预算3000，推荐一款拍照好的手机",
            assistant_answer="answer",
            intent="shopping_guide",
            category_id="cat_phone",
            category_path="数码/手机",
            budget_min=None,
            budget_max=3000,
            preferences_json=json.dumps(["拍照"], ensure_ascii=False),
            product_ids_json=json.dumps(
                ["phone_001", "phone_002", "phone_003"],
                ensure_ascii=False,
            ),
            citation_chunk_ids_json="[]",
        )
    )
    db.commit()

    state = workflow.run(
        "第一个和第二个有什么区别",
        session_id=session_id,
    )

    returned_ids = {candidate.product_id for candidate in state.product_candidates}
    assert state.compare_context is not None
    assert returned_ids
    assert returned_ids <= {"phone_001", "phone_002"}
    assert "上一轮推荐" in (state.answer or "")
    assert "follow_up_rewrite" in _steps(state)
    product_comparison = next(
        step for step in state.trace if step.get("step") == "product_comparison"
    )
    assert product_comparison["status"] == "compared"
    assert product_comparison["returned_product_ids"] == ["phone_001", "phone_002"]

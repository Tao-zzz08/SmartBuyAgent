import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.context import AgentRuntimeContext
from app.agent.nodes import (
    compare_node,
    follow_up_rewrite_node,
    intent_router_node,
    load_context_node,
    product_knowledge_node,
    response_compose_node,
    shopping_guide_node,
)
from app.agent.state import create_initial_agent_state
from app.chat.conversation_memory import ConversationMemoryService
from app.chat.followup_rewriter import FollowUpQueryRewriter
from app.chat.product_comparison import CompareContext, ProductComparisonService
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
def agent_context():
    db_path = PROJECT_ROOT / "data" / "smartbuy_agent_core_nodes_test.db"
    chroma_dir = PROJECT_ROOT / "data" / "chroma_agent_core_nodes_test"
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
        yield context
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)


def _trace_steps(state) -> set[str]:
    return {step["step"] for step in state.trace}


def test_load_context_node_reads_recent_turns(agent_context) -> None:
    db = agent_context.db
    db.add(ChatSession(session_id="session_agent"))
    db.add(
        ChatTurn(
            session_id="session_agent",
            turn_index=1,
            user_query="预算3000，推荐一款拍照好的手机",
            assistant_answer="answer",
            intent="shopping_guide",
            category_id="cat_phone",
            category_path="数码/手机",
            budget_min=None,
            budget_max=3000,
            preferences_json=json.dumps(["拍照"], ensure_ascii=False),
            product_ids_json=json.dumps(["phone_001", "phone_002"], ensure_ascii=False),
            citation_chunk_ids_json="[]",
        )
    )
    db.commit()
    state = create_initial_agent_state("第一个怎么样", session_id="session_agent")

    load_context_node(state, agent_context)

    assert len(state.recent_turns) == 1
    assert state.trace[-1]["node"] == "load_context"
    assert state.trace[-1]["status"] == "loaded"
    assert state.trace[-1]["turn_count"] == 1


def test_intent_router_node_populates_query_understanding() -> None:
    state = create_initial_agent_state("预算3000，推荐一款拍照好的手机")
    context = AgentRuntimeContext(
        query_understanding_service=QueryUnderstandingService(),
    )

    intent_router_node(state, context)

    assert state.intent == "shopping_guide"
    assert state.category_id == "cat_phone"
    assert state.category_path == "数码/手机"
    assert state.budget_max == 3000
    assert "拍照" in state.preferences
    assert state.query_result is not None
    assert "query_understanding" in _trace_steps(state)


def test_shopping_guide_node_retrieves_products_and_citations(agent_context) -> None:
    state = create_initial_agent_state("预算3000，推荐一款拍照好的手机")

    intent_router_node(state, agent_context)
    shopping_guide_node(state, agent_context)

    assert state.product_candidates
    assert state.citations
    assert "product_retrieval" in _trace_steps(state)
    assert "knowledge_retrieval" in _trace_steps(state)


def test_product_knowledge_node_retrieves_only_citations(agent_context) -> None:
    state = create_initial_agent_state("为什么手机拍照不能只看像素")

    intent_router_node(state, agent_context)
    product_knowledge_node(state, agent_context)

    assert state.citations
    assert state.product_candidates == []
    assert "knowledge_retrieval" in _trace_steps(state)


def test_compare_node_uses_only_compare_context_product_ids(agent_context) -> None:
    state = create_initial_agent_state("第一个和第二个有什么区别")
    state.compare_context = CompareContext(
        product_ids=["phone_001", "phone_002"],
        source="resolved_product_ids",
        focus_preferences=["拍照"],
    )

    compare_node(state, agent_context)

    returned_ids = {candidate.product_id for candidate in state.product_candidates}
    assert returned_ids
    assert returned_ids <= {"phone_001", "phone_002"}
    assert state.answer is not None
    assert "上一轮推荐" in state.answer
    comparison_trace = next(
        step for step in state.trace if step.get("step") == "product_comparison"
    )
    assert comparison_trace["status"] == "compared"


def test_follow_up_rewrite_node_resolves_ordinal_context() -> None:
    previous_turn = SimpleNamespace(
        turn_index=1,
        category_id="cat_phone",
        category_path="数码/手机",
        preferences_json=json.dumps(["拍照"], ensure_ascii=False),
        product_ids_json=json.dumps(
            ["phone_001", "phone_002", "phone_003"],
            ensure_ascii=False,
        ),
    )
    state = create_initial_agent_state(
        "第一个和第二个有什么区别",
        session_id="session_agent",
    )
    state.recent_turns = [previous_turn]
    context = AgentRuntimeContext(followup_rewriter=FollowUpQueryRewriter())

    follow_up_rewrite_node(state, context)

    assert state.original_query == "第一个和第二个有什么区别"
    assert state.effective_query != state.original_query
    assert state.compare_context is not None
    assert state.compare_context.product_ids == ["phone_001", "phone_002"]
    rewrite_trace = next(
        step for step in state.trace if step.get("step") == "follow_up_rewrite"
    )
    assert rewrite_trace["status"] == "rewritten"
    assert rewrite_trace["resolved_product_ids"] == ["phone_001", "phone_002"]


def test_response_compose_node_builds_answer_and_product_cards(agent_context) -> None:
    state = create_initial_agent_state("预算3000，推荐一款拍照好的手机")

    intent_router_node(state, agent_context)
    shopping_guide_node(state, agent_context)
    candidate_ids = {candidate.product_id for candidate in state.product_candidates}

    response_compose_node(state, agent_context)

    assert state.answer
    assert state.product_cards
    assert {card.product_id for card in state.product_cards} <= candidate_ids
    assert state.citations
    assert "response_composer" in _trace_steps(state)

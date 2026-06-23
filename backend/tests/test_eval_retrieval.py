from pathlib import Path
from types import SimpleNamespace
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_retrieval import (  # noqa: E402
    evaluate_knowledge_case,
    evaluate_product_case,
    keyword_hit,
    load_eval_cases,
    run_eval,
)


class FakeProductService:
    def __init__(self, products: list[SimpleNamespace]) -> None:
        self.products = products
        self.last_filters = None
        self.last_top_k = None

    def search_products(self, query, filters, top_k):
        self.last_filters = filters
        self.last_top_k = top_k
        return self.products[:top_k]


class FakeKnowledgeService:
    def __init__(self, citations: list[SimpleNamespace]) -> None:
        self.citations = citations
        self.last_category_id = None
        self.last_top_k = None

    def search_knowledge(self, query, category_id=None, top_k=5):
        self.last_category_id = category_id
        self.last_top_k = top_k
        return self.citations[:top_k]


def test_load_eval_cases_reads_json() -> None:
    cases = load_eval_cases(PROJECT_ROOT / "data" / "eval" / "retrieval_eval_cases.json")

    assert len(cases) >= 15
    assert {case["type"] for case in cases} >= {
        "product_retrieval",
        "knowledge_retrieval",
    }
    assert all("id" in case and "query" in case and "expect" in case for case in cases)


def test_keyword_hit_detects_any_keyword() -> None:
    assert keyword_hit(["手机拍照需要看防抖和夜景"], ["像素", "防抖"])
    assert not keyword_hit(["手机拍照需要看防抖和夜景"], ["尺码", "脚背"])
    assert keyword_hit(["anything"], [])


def test_evaluate_product_case_with_structured_soft_assertions() -> None:
    product_service = FakeProductService(
        [
            SimpleNamespace(
                product_id="phone_ok",
                category_id="cat_phone",
                price=3999,
                title="小米拍照手机",
                brand="Xiaomi",
                tags=["拍照", "影像"],
                description="适合拍照",
            )
        ]
    )
    knowledge_service = FakeKnowledgeService(
        [
            SimpleNamespace(
                chunk_id="chunk_phone_camera",
                source_file="data/knowledge_docs/phone/camera.md",
                content_preview="手机拍照需要看影像、防抖和传感器。",
            )
        ]
    )
    case = {
        "id": "phone_camera",
        "query": "预算5000，推荐拍照好的手机，不考虑苹果",
        "type": "product_retrieval",
        "structured_filters": {
            "category_id": "cat_phone",
            "budget_max": 5000,
            "preferences": ["拍照"],
            "negative_preferences": ["苹果"],
        },
        "expect": {
            "min_results": 1,
            "all_category": "phone",
            "max_price": 5000,
            "must_match_any": ["拍照", "影像"],
            "forbidden_terms": ["苹果", "Apple"],
            "top_k": 5,
        },
        "knowledge_expect": {"must_contain_any": ["防抖"]},
    }

    result = evaluate_product_case(case, product_service, knowledge_service)

    assert result["passed"] is True
    assert result["actual_product_ids"] == ["phone_ok"]
    assert result["category_ok"] is True
    assert result["budget_ok"] is True
    assert result["negative_preference_violations"] == 0
    assert result["failure_reasons"] == []
    assert product_service.last_filters.category_id == "cat_phone"
    assert product_service.last_filters.budget_max == 5000
    assert product_service.last_filters.preferences == ["拍照"]
    assert product_service.last_filters.brand_exclude == ["苹果"]
    assert product_service.last_top_k == 5


def test_evaluate_knowledge_case_with_soft_assertions() -> None:
    knowledge_service = FakeKnowledgeService(
        [
            SimpleNamespace(
                chunk_id="chunk_skincare_sensitive",
                source_file="data/knowledge_docs/skincare/sensitive.md",
                content_preview="敏感肌应关注温和、保湿和低刺激。",
            )
        ]
    )
    case = {
        "id": "sensitive_skin",
        "query": "敏感肌护肤应该注意什么？",
        "type": "knowledge_retrieval",
        "category_id": "cat_skincare",
        "expect": {
            "min_chunks": 1,
            "must_contain_any": ["敏感肌", "温和"],
            "forbidden_terms": ["治疗"],
            "top_k": 5,
        },
    }

    result = evaluate_knowledge_case(case, knowledge_service)

    assert result["passed"] is True
    assert result["citation_keyword_hit"] is True
    assert result["citation_count"] == 1
    assert result["failure_reasons"] == []
    assert result["actual_citation_sources"] == [
        "data/knowledge_docs/skincare/sensitive.md"
    ]
    assert knowledge_service.last_category_id == "cat_skincare"
    assert knowledge_service.last_top_k == 5


def test_evaluate_product_case_reports_soft_failures() -> None:
    product_service = FakeProductService(
        [
            SimpleNamespace(
                product_id="shoe_wrong",
                category_id="cat_shoes",
                price=5999,
                title="Apple 高跟鞋",
                brand="Apple",
                tags=["高跟"],
                description="wrong category",
            )
        ]
    )
    knowledge_service = FakeKnowledgeService([])
    case = {
        "id": "failing_product",
        "query": "预算5000，推荐拍照好的手机，不考虑苹果",
        "type": "product_retrieval",
        "structured_filters": {"category_id": "cat_phone", "budget_max": 5000},
        "expect": {
            "min_results": 1,
            "all_category": "phone",
            "max_price": 5000,
            "must_match_any": ["拍照"],
            "forbidden_terms": ["Apple"],
        },
    }

    result = evaluate_product_case(case, product_service, knowledge_service)

    assert result["passed"] is False
    assert "category mismatch" in result["failure_reasons"]
    assert "budget constraint violated" in result["failure_reasons"]
    assert "product preference keywords not found" in result["failure_reasons"]
    assert "forbidden product terms found" in result["failure_reasons"]


def test_run_eval_returns_structured_summary_metrics() -> None:
    product_service = FakeProductService(
        [
            SimpleNamespace(
                product_id="phone_ok",
                category_id="cat_phone",
                price=1999,
                title="拍照手机",
                brand="Xiaomi",
                tags=["拍照"],
            )
        ]
    )
    knowledge_service = FakeKnowledgeService(
        [
            SimpleNamespace(
                chunk_id="chunk_camera",
                source_file="data/knowledge_docs/phone/camera.md",
                content_preview="拍照需要看防抖。",
            )
        ]
    )
    cases = [
        {
            "id": "passing_product",
            "query": "预算3000，推荐拍照手机",
            "type": "product_retrieval",
            "structured_filters": {"category_id": "cat_phone", "budget_max": 3000},
            "expect": {
                "min_results": 1,
                "all_category": "phone",
                "max_price": 3000,
                "must_match_any": ["拍照"],
            },
        },
        {
            "id": "passing_knowledge",
            "query": "手机拍照主要看什么？",
            "type": "knowledge_retrieval",
            "category_id": "cat_phone",
            "expect": {"min_chunks": 1, "must_contain_any": ["防抖"]},
        },
    ]

    eval_output = run_eval(cases, product_service, knowledge_service)

    assert eval_output["summary"]["total_cases"] == 2
    assert eval_output["summary"]["passed_cases"] == 2
    assert eval_output["summary"]["failed_cases"] == 0
    assert eval_output["summary"]["product_category_compliance"] == 1.0
    assert eval_output["summary"]["budget_compliance"] == 1.0
    assert eval_output["summary"]["negative_preference_violation_count"] == 0
    assert eval_output["summary"]["knowledge_chunk_hit_rate"] == 1.0

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

    assert len(cases) >= 8
    assert {case["type"] for case in cases} >= {"product", "knowledge"}
    assert all("id" in case and "query" in case for case in cases)


def test_keyword_hit_detects_any_keyword() -> None:
    assert keyword_hit(["手机拍照需要看防抖和夜景"], ["像素", "防抖"])
    assert not keyword_hit(["手机拍照需要看防抖和夜景"], ["尺码", "脚背"])
    assert keyword_hit(["anything"], [])


def test_evaluate_product_case_with_fake_services() -> None:
    product_service = FakeProductService(
        [
            SimpleNamespace(product_id="phone_001", category_id="cat_phone", price=2599),
            SimpleNamespace(product_id="phone_003", category_id="cat_phone", price=2999),
        ]
    )
    knowledge_service = FakeKnowledgeService(
        [
            SimpleNamespace(
                title="手机拍照选购指南",
                section="为什么不能只看像素",
                section_path="手机拍照选购指南/为什么不能只看像素",
                source_file="data/knowledge_docs/phone/phone_camera_guide.md",
                content_preview="拍照还要看影像、防抖和夜景表现。",
            )
        ]
    )
    case = {
        "id": "phone_camera",
        "query": "预算3000，推荐一款拍照好的手机",
        "type": "product",
        "category_id": "cat_phone",
        "budget_max": 3000,
        "expected_product_ids": ["phone_003"],
        "expected_doc_keywords": ["防抖"],
        "top_k": 3,
    }

    result = evaluate_product_case(case, product_service, knowledge_service)

    assert result["passed"] is True
    assert result["actual_product_ids"] == ["phone_001", "phone_003"]
    assert result["product_hit"] is True
    assert result["category_ok"] is True
    assert result["budget_ok"] is True
    assert result["citation_keyword_hit"] is True
    assert product_service.last_filters.category_id == "cat_phone"
    assert product_service.last_filters.budget_max == 3000
    assert product_service.last_top_k == 3


def test_evaluate_knowledge_case_with_fake_service() -> None:
    knowledge_service = FakeKnowledgeService(
        [
            SimpleNamespace(
                title="敏感肌护肤选购指南",
                section="成分精简",
                section_path="敏感肌护肤选购指南/成分精简",
                source_file="data/knowledge_docs/skincare/skincare_sensitive_skin.md",
                content_preview="敏感肌应关注温和、保湿和修护，不承诺治疗效果。",
            )
        ]
    )
    case = {
        "id": "sensitive_skin",
        "query": "敏感肌护肤要注意什么",
        "type": "knowledge",
        "category_id": "cat_skincare",
        "expected_doc_keywords": ["敏感肌", "温和"],
        "top_k": 5,
    }

    result = evaluate_knowledge_case(case, knowledge_service)

    assert result["passed"] is True
    assert result["citation_keyword_hit"] is True
    assert result["citation_count"] == 1
    assert result["actual_citation_sources"] == [
        "data/knowledge_docs/skincare/skincare_sensitive_skin.md"
    ]
    assert knowledge_service.last_category_id == "cat_skincare"
    assert knowledge_service.last_top_k == 5


def test_run_eval_returns_summary_counts() -> None:
    product_service = FakeProductService(
        [SimpleNamespace(product_id="phone_001", category_id="cat_phone", price=1999)]
    )
    knowledge_service = FakeKnowledgeService(
        [
            SimpleNamespace(
                title="手机拍照选购指南",
                section="防抖",
                section_path="手机拍照选购指南/防抖",
                source_file="data/knowledge_docs/phone/phone_camera_guide.md",
                content_preview="拍照要看防抖。",
            )
        ]
    )
    cases = [
        {
            "id": "passing_product",
            "query": "预算3000，推荐手机",
            "type": "product",
            "category_id": "cat_phone",
            "budget_max": 3000,
            "expected_product_ids": ["phone_001"],
            "expected_doc_keywords": ["防抖"],
            "top_k": 3,
        },
        {
            "id": "failing_knowledge",
            "query": "鞋靴尺码怎么选",
            "type": "knowledge",
            "category_id": "cat_shoes",
            "expected_doc_keywords": ["脚背"],
            "top_k": 5,
        },
    ]

    eval_output = run_eval(cases, product_service, knowledge_service)

    assert eval_output["summary"]["total_cases"] == 2
    assert eval_output["summary"]["passed_cases"] == 1
    assert eval_output["summary"]["failed_cases"] == 1
    assert eval_output["summary"]["product_hit_rate"] == 1.0
    assert eval_output["summary"]["citation_keyword_hit_rate"] == 0.5

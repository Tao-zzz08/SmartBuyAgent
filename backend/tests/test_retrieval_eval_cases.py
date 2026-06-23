from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_retrieval import load_eval_cases, run_eval  # noqa: E402


class RoutingFakeProductService:
    def search_products(self, query, filters, top_k):
        if filters.category_id == "cat_skincare":
            products = [
                SimpleNamespace(
                    product_id="skincare_sensitive_001",
                    category_id="cat_skincare",
                    price=199,
                    title="敏感肌保湿乳",
                    brand="Demo",
                    tags=["敏感肌", "保湿", "温和"],
                    description="适合日常保湿护理",
                )
            ]
        else:
            products = [
                SimpleNamespace(
                    product_id="phone_camera_001",
                    category_id="cat_phone",
                    price=3999,
                    title="小米拍照手机",
                    brand="Xiaomi",
                    tags=["拍照", "影像"],
                    description="适合拍照和日常使用",
                )
            ]
        return products[:top_k]


class RoutingFakeKnowledgeService:
    def search_knowledge(self, query, category_id=None, top_k=5):
        text_by_category = {
            "cat_phone": "手机拍照主要看传感器、光圈、防抖和影像算法。",
            "cat_skincare": "敏感肌保湿应关注温和、低刺激和日常护理。",
        }
        return [
            SimpleNamespace(
                chunk_id=f"chunk_{category_id}",
                source_file=f"data/knowledge_docs/{category_id}/guide.md",
                content_preview=text_by_category.get(category_id, "通用知识"),
            )
        ][:top_k]


def test_core_retrieval_eval_cases_pass_with_fake_services() -> None:
    wanted = {
        "phone_camera_under_5000",
        "phone_without_apple",
        "skincare_sensitive_moisturizing_under_300",
    }
    cases = [
        case
        for case in load_eval_cases(PROJECT_ROOT / "data" / "eval" / "retrieval_eval_cases.json")
        if case["id"] in wanted
    ]

    output = run_eval(cases, RoutingFakeProductService(), RoutingFakeKnowledgeService())

    assert output["summary"]["failed_cases"] == 0
    assert output["summary"]["passed_cases"] == len(cases)

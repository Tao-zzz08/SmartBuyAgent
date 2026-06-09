from app.chat.query_understanding import QueryUnderstandingResult
from app.chat.response_composer import ResponseComposer
from app.retrieval.retrieval_service import Citation, ProductCandidate


def _query_result(
    intent: str,
    need_clarification: bool = False,
) -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        raw_query="预算3000，推荐一款拍照好的手机",
        intent=intent,
        category_id="cat_phone",
        category_path="数码/手机",
        budget_min=None,
        budget_max=3000,
        preferences=["拍照"],
        need_clarification=need_clarification,
        clarification_question="你想看哪个品类的商品？目前我可以帮你选手机、鞋靴和护肤品。"
        if need_clarification
        else None,
    )


def _product_candidate(product_id: str, title: str) -> ProductCandidate:
    return ProductCandidate(
        product_id=product_id,
        title=title,
        brand="星曜",
        category_id="cat_phone",
        price=2599,
        stock=10,
        description="适合日常拍照和学生使用",
        image_url=f"https://example.com/images/{product_id}.jpg",
        tags=["拍照", "性价比"],
        attributes={"存储容量": "256GB", "电池容量": "5000mAh"},
        source_url=f"https://example.com/products/{product_id}",
        compare_url=f"https://example.com/compare/{product_id}",
        distance=0.2,
        score=0.83,
        product_text="商品：测试手机\n标签：拍照, 性价比",
    )


def _citation() -> Citation:
    return Citation(
        chunk_id="chunk_001",
        document_id="doc_phone_camera",
        title="手机拍照选购指南",
        section="为什么不能只看像素",
        section_path="手机拍照选购指南 / 为什么不能只看像素",
        source_file="data/knowledge_docs/phone/phone_camera_guide.md",
        doc_type="guide",
        category_id="cat_phone",
        category_path="数码/手机",
        content_preview="高像素不自动等于好照片，还要看传感器、防抖和算法。",
        distance=0.3,
        score=0.76,
    )


def test_compose_shopping_guide_with_products_and_citations() -> None:
    composer = ResponseComposer()
    query_result = _query_result("shopping_guide")
    products = [
        _product_candidate("phone_001", "星曜 X1 5G 全新手机"),
        _product_candidate("phone_002", "星曜 X2 Pro 全新手机"),
    ]

    response = composer.compose(
        query_result,
        product_candidates=products,
        citations=[_citation()],
    )

    assert response.answer
    assert len(response.product_cards) == 2
    assert response.product_cards[0].product_id == "phone_001"
    assert response.product_cards[0].recommend_reason
    assert len(response.citations) == 1
    assert response.trace[0]["step"] == "response_composer"


def test_compose_shopping_guide_without_products() -> None:
    composer = ResponseComposer()

    response = composer.compose(_query_result("shopping_guide"))

    assert "没有找到满足条件的商品" in response.answer
    assert response.product_cards == []


def test_compose_clarification() -> None:
    composer = ResponseComposer()
    query_result = _query_result("clarification", need_clarification=True)

    response = composer.compose(query_result)

    assert response.answer == query_result.clarification_question
    assert response.product_cards == []
    assert response.citations == []
    assert response.trace[0]["need_clarification"] is True


def test_compose_chitchat() -> None:
    composer = ResponseComposer()

    response = composer.compose(_query_result("chitchat"))

    assert response.answer
    assert response.product_cards == []
    assert response.citations == []


def test_compose_product_knowledge_with_citation() -> None:
    composer = ResponseComposer()

    response = composer.compose(
        _query_result("product_knowledge"),
        citations=[_citation()],
    )

    assert response.answer
    assert len(response.citations) == 1
    assert response.product_cards == []


def test_compose_compare_not_implemented() -> None:
    composer = ResponseComposer()

    response = composer.compose(_query_result("compare"))

    assert "当前阶段还没有实现对比服务" in response.answer
    assert response.product_cards == []


def test_product_card_does_not_invent_attributes() -> None:
    composer = ResponseComposer()
    candidate = _product_candidate("phone_001", "星曜 X1 5G 全新手机")

    response = composer.compose(
        _query_result("shopping_guide"),
        product_candidates=[candidate],
    )

    assert response.product_cards[0].attributes == candidate.attributes
    assert set(response.product_cards[0].attributes) == {"存储容量", "电池容量"}

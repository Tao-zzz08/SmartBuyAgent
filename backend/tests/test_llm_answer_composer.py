from __future__ import annotations

from dataclasses import replace

from app.chat.llm_answer_composer import LLMAnswerComposer
from app.chat.query_understanding import QueryUnderstandingResult
from app.retrieval.retrieval_service import Citation, ProductCandidate
from app.services.llm import BaseLLMService, LLMMessage, LLMResponse


class FakeLLMService(BaseLLMService):
    def __init__(self, content: str = "LLM answer", should_raise: bool = False) -> None:
        self.content = content
        self.should_raise = should_raise
        self.messages: list[LLMMessage] | None = None
        self.calls = 0

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls += 1
        self.messages = messages
        if self.should_raise:
            raise RuntimeError("fake error")
        return LLMResponse(content=self.content, model="fake-model", provider="fake")


def _query_result() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        raw_query="预算3000，推荐一款拍照好的手机",
        intent="shopping_guide",
        category_id="cat_phone",
        category_path="数码/手机",
        budget_min=None,
        budget_max=3000,
        preferences=["拍照"],
        need_clarification=False,
        clarification_question=None,
    )


def _product(product_id: str = "phone_001", title: str = "星曜 X1 5G 全新手机") -> ProductCandidate:
    return ProductCandidate(
        product_id=product_id,
        title=title,
        brand="星曜",
        category_id="cat_phone",
        price=2599,
        stock=20,
        description="适合日常拍照和学生使用，影像表现稳定。",
        image_url="https://example.com/images/phone.jpg",
        tags=["拍照好", "性价比"],
        attributes={
            "存储容量": "256GB",
            "运行内存": "12GB",
            "处理器": "骁龙 7 Gen 3",
            "屏幕尺寸": "6.7英寸",
            "刷新率": "120Hz",
            "电池容量": "5100mAh",
            "快充功率": "67W",
            "拍照能力": "OIS 防抖主摄",
            "保修": "官方一年保修",
        },
        source_url="https://example.com/products/phone_001",
        compare_url="https://example.com/compare/phone_001",
        distance=0.2,
        score=0.83,
        product_text="商品：星曜 X1 5G 全新手机\n标签：拍照好，性价比",
    )


def _citation(
    chunk_id: str = "chunk_001",
    title: str = "手机拍照选购指南",
    preview: str = "手机拍照不能只看像素，还要看主摄、防抖、夜景和影像算法。",
) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        document_id="doc_phone_camera",
        title=title,
        section="为什么不能只看像素",
        section_path="手机拍照选购指南/为什么不能只看像素",
        source_file="data/knowledge_docs/phone/phone_camera_guide.md",
        doc_type="guide",
        category_id="cat_phone",
        category_path="数码/手机",
        content_preview=preview,
        distance=0.1,
        score=0.9,
    )


def test_compose_calls_llm_with_products_and_citations() -> None:
    fake_llm = FakeLLMService(content="这是最终导购回答")
    composer = LLMAnswerComposer(fake_llm)

    answer = composer.compose(
        query="预算3000，推荐一款拍照好的手机",
        query_result=_query_result(),
        product_candidates=[_product()],
        citations=[_citation()],
    )

    assert answer == "这是最终导购回答"
    assert fake_llm.calls == 1
    assert fake_llm.messages is not None
    assert [message.role for message in fake_llm.messages] == ["system", "user"]

    system_prompt = fake_llm.messages[0].content
    user_prompt = fake_llm.messages[1].content
    assert "不能编造商品" in system_prompt
    assert "候选商品" in system_prompt
    assert "预算3000，推荐一款拍照好的手机" in user_prompt
    assert "星曜 X1 5G 全新手机" in user_prompt
    assert "price: 2599" in user_prompt
    assert "手机拍照不能只看像素" in user_prompt


def test_compose_returns_fallback_without_products_or_citations() -> None:
    fake_llm = FakeLLMService()
    composer = LLMAnswerComposer(fake_llm)

    answer = composer.compose(
        query="推荐一下",
        query_result=_query_result(),
        product_candidates=[],
        citations=[],
    )

    assert fake_llm.calls == 0
    assert "当前没有找到足够匹配的商品或知识依据" in answer


def test_compose_returns_fallback_when_llm_raises() -> None:
    composer = LLMAnswerComposer(FakeLLMService(should_raise=True))

    answer = composer.compose(
        query="预算3000，推荐一款拍照好的手机",
        query_result=_query_result(),
        product_candidates=[_product()],
        citations=[],
    )

    assert "当前没有找到足够匹配的商品或知识依据" in answer


def test_compose_returns_fallback_when_llm_content_empty() -> None:
    composer = LLMAnswerComposer(FakeLLMService(content="   "))

    answer = composer.compose(
        query="预算3000，推荐一款拍照好的手机",
        query_result=_query_result(),
        product_candidates=[_product()],
        citations=[],
    )

    assert "当前没有找到足够匹配的商品或知识依据" in answer


def test_prompt_limits_product_and_citation_count() -> None:
    fake_llm = FakeLLMService()
    composer = LLMAnswerComposer(fake_llm)
    products = [
        _product(product_id=f"phone_00{index}", title=f"候选手机 {index}")
        for index in range(1, 6)
    ]
    citations = [
        _citation(chunk_id=f"chunk_00{index}", title=f"引用文档 {index}")
        for index in range(1, 8)
    ]

    composer.compose(
        query="预算3000，推荐一款拍照好的手机",
        query_result=_query_result(),
        product_candidates=products,
        citations=citations,
    )

    assert fake_llm.messages is not None
    user_prompt = fake_llm.messages[1].content
    assert "候选手机 1" in user_prompt
    assert "候选手机 2" in user_prompt
    assert "候选手机 3" in user_prompt
    assert "候选手机 4" not in user_prompt
    assert "引用文档 5" in user_prompt
    assert "引用文档 6" not in user_prompt


def test_prompt_does_not_include_purchase_action_instruction() -> None:
    fake_llm = FakeLLMService()
    composer = LLMAnswerComposer(fake_llm)

    composer.compose(
        query="预算3000，推荐一款拍照好的手机",
        query_result=_query_result(),
        product_candidates=[_product()],
        citations=[_citation()],
    )

    assert fake_llm.messages is not None
    prompt_text = "\n".join(message.content for message in fake_llm.messages)
    assert "不能声称已经下单" in prompt_text
    assert "加入购物车" in prompt_text
    assert "支付" in prompt_text
    assert "不要输出购买、下单、支付、加入购物车等动作指令" in prompt_text


def test_prompt_truncates_long_text() -> None:
    fake_llm = FakeLLMService()
    composer = LLMAnswerComposer(fake_llm)
    long_description = "长描述" * 200
    long_preview = "长引用" * 250

    composer.compose(
        query="预算3000，推荐一款拍照好的手机",
        query_result=_query_result(),
        product_candidates=[
            replace(_product(title="长文本手机"), description=long_description)
        ],
        citations=[_citation(preview=long_preview)],
    )

    assert fake_llm.messages is not None
    user_prompt = fake_llm.messages[1].content
    assert len(user_prompt) < len(long_description) + len(long_preview)
    assert "..." in user_prompt

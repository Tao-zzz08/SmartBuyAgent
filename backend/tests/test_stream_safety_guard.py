from app.streaming.safety import StreamSafetyGuard


def test_stream_safety_guard_blocks_single_delta_purchase_action() -> None:
    guard = StreamSafetyGuard()

    decision = guard.check_delta(
        "我已经帮你下单这款手机。",
        category="cat_phone",
        intent="shopping_guide",
    )

    assert decision.safe is False
    assert decision.reason == "purchase_action"
    assert decision.matched_phrase == "我已经帮你下单"


def test_stream_safety_guard_blocks_cross_token_medical_claim() -> None:
    guard = StreamSafetyGuard()

    first_decision = guard.check_delta("这款可以治", category="cat_skincare")
    second_decision = guard.check_delta("疗湿疹。", category="cat_skincare")

    assert first_decision.safe is True
    assert second_decision.safe is False
    assert second_decision.reason == "medical_claim"
    assert second_decision.matched_phrase == "可以治疗湿疹"


def test_stream_safety_guard_is_stricter_for_skincare_category() -> None:
    guard = StreamSafetyGuard()

    decision = guard.check_delta(
        "这款产品可以治疗痘痘。",
        category="cat_skincare",
        intent="shopping_guide",
    )

    assert decision.safe is False
    assert decision.reason in {"medical_claim", "skincare_medical_claim"}


def test_stream_safety_guard_does_not_block_normal_purchase_advice() -> None:
    guard = StreamSafetyGuard()

    first_decision = guard.check_delta("这款是否值得购买？", intent="shopping_guide")
    second_decision = guard.check_delta("购买前建议比较参数和售后。")

    assert first_decision.safe is True
    assert second_decision.safe is True

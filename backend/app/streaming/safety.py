from __future__ import annotations

from dataclasses import dataclass


DEFAULT_ROLLING_CHARS = 120
STREAM_SAFE_RELEASE_HOLD_CHARS = 80
STREAM_GUARDED_FALLBACK_ANSWER = (
    "抱歉，刚才的回答可能包含不适合直接展示的内容。"
    "我可以基于已检索到的商品信息，重新用更安全的方式为你说明。"
)


@dataclass(frozen=True)
class StreamSafetyDecision:
    safe: bool
    reason: str | None = None
    matched_phrase: str | None = None
    severity: str = "none"


class StreamSafetyViolation(Exception):
    def __init__(
        self,
        reason: str,
        matched_phrase: str | None = None,
        severity: str = "high",
    ) -> None:
        self.reason = reason
        self.matched_phrase = matched_phrase
        self.severity = severity
        super().__init__("Streaming output blocked by safety guard")


class StreamSafetyGuard:
    def __init__(self, rolling_chars: int = DEFAULT_ROLLING_CHARS) -> None:
        self.rolling_chars = rolling_chars
        self._buffer = ""

    def check_delta(
        self,
        delta: str,
        *,
        category: str | None = None,
        intent: str | None = None,
    ) -> StreamSafetyDecision:
        self._buffer = (self._buffer + (delta or ""))[-self.rolling_chars :]
        return self.check_buffer(self._buffer, category=category, intent=intent)

    def check_buffer(
        self,
        text: str,
        *,
        category: str | None = None,
        intent: str | None = None,
    ) -> StreamSafetyDecision:
        normalized = _normalize(text)
        category_text = _normalize(category)
        is_skincare = "skincare" in category_text or "护肤" in category_text

        decision = _match_phrases(normalized, PURCHASE_ACTION_PHRASES, "purchase_action")
        if decision is not None:
            return decision

        decision = _match_phrases(normalized, PURCHASE_LINK_PHRASES, "purchase_link")
        if decision is not None:
            return decision

        decision = _match_phrases(normalized, FABRICATED_SOURCE_PHRASES, "fabricated_source")
        if decision is not None:
            return decision

        decision = _match_phrases(
            normalized,
            MEDICAL_CLAIM_ALWAYS_PHRASES,
            "medical_claim",
        )
        if decision is not None:
            return decision

        if is_skincare:
            decision = _match_phrases(
                normalized,
                MEDICAL_CLAIM_SKINCARE_PHRASES,
                "skincare_medical_claim",
            )
            if decision is not None:
                return decision

        return StreamSafetyDecision(safe=True)


PURCHASE_ACTION_PHRASES = [
    "我已经帮你下单",
    "我已为你购买",
    "已为你购买",
    "已经加入购物车",
    "已加入购物车",
    "点击购买",
    "立即付款",
    "支付成功",
    "订单已创建",
    "已帮你下单",
    "立即下单",
    "现在下单",
    "已完成购买",
    "buy now",
    "checkout now",
    "payment successful",
    "order created",
]

PURCHASE_LINK_PHRASES = [
    "购买链接",
    "下单链接",
    "支付链接",
    "点击这里购买",
    "我帮你生成了购买链接",
    "buy link",
    "checkout link",
    "payment link",
]

MEDICAL_CLAIM_ALWAYS_PHRASES = [
    "可以治疗湿疹",
    "可以治疗痤疮",
    "可以治疗皮炎",
    "治疗湿疹",
    "治疗痤疮",
    "治疗皮炎",
    "修复皮肤病",
]

MEDICAL_CLAIM_SKINCARE_PHRASES = [
    "治疗",
    "治愈",
    "药效",
    "药用",
    "处方",
    "祛病",
    "医学修复",
    "cure",
    "treat",
    "medical effect",
    "prescription",
]

FABRICATED_SOURCE_PHRASES = [
    "根据我检索到的真实购买页面",
    "来自京东官方",
    "来自淘宝官方",
    "真实电商链接",
    "jd official",
    "taobao official",
    "real ecommerce link",
]


def _match_phrases(
    normalized_text: str,
    phrases: list[str],
    reason: str,
) -> StreamSafetyDecision | None:
    for phrase in phrases:
        normalized_phrase = _normalize(phrase)
        if normalized_phrase and normalized_phrase in normalized_text:
            return StreamSafetyDecision(
                safe=False,
                reason=reason,
                matched_phrase=phrase,
                severity="high",
            )
    return None


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())

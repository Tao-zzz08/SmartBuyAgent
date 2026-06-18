from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any


VALID_CATEGORIES = {"phone", "shoes", "skincare"}
BUDGET_MIN_VALUE = 1
BUDGET_MAX_VALUE = 100000

CATEGORY_TO_ID = {
    "phone": "cat_phone",
    "shoes": "cat_shoes",
    "skincare": "cat_skincare",
}
CATEGORY_TO_PATH = {
    "phone": "数码/手机",
    "shoes": "服装/鞋靴",
    "skincare": "美妆/护肤",
}
CATEGORY_TO_DISPLAY = {
    "phone": "手机",
    "shoes": "鞋靴",
    "skincare": "护肤品",
}
CATEGORY_ID_TO_CATEGORY = {value: key for key, value in CATEGORY_TO_ID.items()}

CATEGORY_ALIASES = {
    "phone": ["苹果手机", "iphone", "影像手机", "拍照手机", "手机", "安卓"],
    "shoes": ["运动鞋", "通勤鞋", "鞋靴", "鞋子", "靴子", "跑鞋", "鞋"],
    "skincare": ["护肤品", "护肤", "面霜", "精华", "水乳", "洁面", "防晒"],
}

CATEGORY_COMPATIBLE_PREFERENCES = {
    "phone": {"拍照", "续航", "性能", "屏幕", "轻薄", "快充", "游戏", "影像", "存储", "高刷屏", "NFC"},
    "shoes": {"轻便", "防滑", "透气", "通勤", "运动", "跑步", "保暖", "舒适", "耐磨", "休闲"},
    "skincare": {"保湿", "清爽", "温和", "敏感肌", "控油", "补水", "修护", "抗氧", "舒缓", "日常护理"},
}

PREFERENCE_KEYWORDS = [
    "敏感肌",
    "性价比",
    "高刷屏",
    "拍照",
    "续航",
    "性能",
    "屏幕",
    "游戏",
    "轻薄",
    "快充",
    "学生",
    "影像",
    "存储",
    "通勤",
    "防滑",
    "透气",
    "舒适",
    "跑步",
    "运动",
    "尺码",
    "材质",
    "耐磨",
    "轻便",
    "保暖",
    "休闲",
    "保湿",
    "补水",
    "修护",
    "舒缓",
    "成分",
    "清爽",
    "防晒",
    "控油",
    "温和",
    "抗氧",
]

NEGATIVE_TARGETS = [
    "苹果",
    "三星",
    "小米",
    "红米",
    "华为",
    "oppo",
    "vivo",
    "高跟",
    "美白",
    "太贵",
    "厚重",
]
NEGATIVE_PATTERNS = [
    r"不考虑\s*(?P<target>[\w\u4e00-\u9fff]+)",
    r"不要\s*(?P<target>[\w\u4e00-\u9fff]+)",
    r"不想要\s*(?P<target>[\w\u4e00-\u9fff]+)",
    r"不喜欢\s*(?P<target>[\w\u4e00-\u9fff]+)",
]

SKINCARE_UNSAFE_TERMS = ["治疗", "治愈", "药效", "药用", "处方", "医学修复", "修复疾病", "祛病"]
SKINCARE_SAFE_REWRITES = {
    "治疗痘痘": ["清爽", "控油", "温和"],
    "治疗湿疹": ["温和", "保湿"],
    "治疗皮炎": ["温和", "保湿"],
    "修复皮肤病": ["温和", "保湿"],
    "药效": [],
    "药用": [],
    "处方": [],
    "医学修复": [],
}

BUDGET_UPDATE_PATTERNS = [
    "预算",
    "以内",
    "以下",
    "提高到",
    "降到",
    "增加到",
    "加到",
    "涨到",
    "升到",
    "改到",
    "改成",
    "变成",
    "到",
    "如果",
]


@dataclass(frozen=True)
class Budget:
    min: int | None = None
    max: int | None = None
    currency: str = "CNY"


@dataclass(frozen=True)
class ShoppingMemory:
    category: str | None = None
    budget: Budget = field(default_factory=Budget)
    preferences: list[str] = field(default_factory=list)
    negative_preferences: list[str] = field(default_factory=list)
    last_product_ids: list[str] = field(default_factory=list)
    last_intent: str | None = None

    def has_shopping_context(self) -> bool:
        return bool(
            self.category
            or self.preferences
            or self.negative_preferences
            or self.last_product_ids
            or self.last_intent == "shopping_guide"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def empty_shopping_memory() -> ShoppingMemory:
    return ShoppingMemory()


def category_to_id(category: str | None) -> str | None:
    return CATEGORY_TO_ID.get(category or "")


def category_to_path(category: str | None) -> str | None:
    return CATEGORY_TO_PATH.get(category or "")


def category_from_id_or_path(
    category_id: str | None,
    category_path: str | None = None,
) -> str | None:
    if category_id in CATEGORY_ID_TO_CATEGORY:
        return CATEGORY_ID_TO_CATEGORY[category_id]
    text = category_path or ""
    if "手机" in text:
        return "phone"
    if "鞋" in text:
        return "shoes"
    if "护肤" in text or "美妆" in text:
        return "skincare"
    return None


def extract_memory_from_query(
    query: str,
    *,
    intent: str | None = None,
) -> ShoppingMemory:
    category = extract_category(query)
    preferences = extract_preferences(query, category=category)
    negative_preferences = extract_negative_preferences(query)
    preferences = [item for item in preferences if item not in negative_preferences]
    budget_max = parse_budget_max(query)
    return ShoppingMemory(
        category=category,
        budget=Budget(max=budget_max),
        preferences=preferences,
        negative_preferences=negative_preferences,
        last_intent=intent,
    )


def memory_from_turn(turn: Any) -> ShoppingMemory:
    category = category_from_id_or_path(
        getattr(turn, "category_id", None),
        getattr(turn, "category_path", None),
    )
    positive, negative, embedded_memory = parse_preferences_payload(
        getattr(turn, "preferences_json", None)
    )
    product_ids = _json_list(getattr(turn, "product_ids_json", None))
    if embedded_memory:
        category = embedded_memory.category or category
        positive = embedded_memory.preferences or positive
        negative = embedded_memory.negative_preferences or negative
    return ShoppingMemory(
        category=category,
        budget=Budget(
            min=_int_or_none(getattr(turn, "budget_min", None)),
            max=_int_or_none(getattr(turn, "budget_max", None)),
            currency="CNY",
        ),
        preferences=positive,
        negative_preferences=negative,
        last_product_ids=product_ids,
        last_intent=getattr(turn, "intent", None),
    )


def merge_turns_to_memory(turns: list[Any]) -> ShoppingMemory:
    memory = empty_shopping_memory()
    for turn in turns:
        memory = merge_shopping_memory(memory, memory_from_turn(turn))
    return memory


def merge_shopping_memory(
    previous: ShoppingMemory,
    current: ShoppingMemory,
) -> ShoppingMemory:
    category = current.category or previous.category
    category_switched = bool(
        previous.category and current.category and previous.category != current.category
    )
    budget = Budget(
        min=current.budget.min if current.budget.min is not None else previous.budget.min,
        max=current.budget.max if current.budget.max is not None else previous.budget.max,
        currency=current.budget.currency or previous.budget.currency or "CNY",
    )

    base_preferences = (
        _filter_preferences_for_category(previous.preferences, category)
        if category_switched
        else list(previous.preferences)
    )
    preferences = _dedupe([*base_preferences, *current.preferences])
    if category:
        preferences = _filter_preferences_for_category(preferences, category)

    negative_preferences = _dedupe(
        [*previous.negative_preferences, *current.negative_preferences]
    )
    preferences = [item for item in preferences if item not in negative_preferences]

    return ShoppingMemory(
        category=category,
        budget=budget,
        preferences=preferences,
        negative_preferences=negative_preferences,
        last_product_ids=current.last_product_ids or previous.last_product_ids,
        last_intent=_merge_last_intent(previous.last_intent, current.last_intent),
    )


def build_effective_query(memory: ShoppingMemory) -> str:
    parts: list[str] = []
    if memory.budget.max is not None:
        parts.append(f"预算{memory.budget.max}元以内")

    category = memory.category or "商品"
    category_text = CATEGORY_TO_DISPLAY.get(category, "商品")
    preference_text = _preference_text(memory.category, memory.preferences)
    if preference_text:
        parts.append(f"推荐{preference_text}的{category_text}")
    else:
        parts.append(f"推荐{category_text}")

    if memory.negative_preferences:
        parts.append(f"不考虑{'、'.join(memory.negative_preferences)}")

    return sanitize_skincare_query("，".join(parts), category=memory.category)


def looks_like_budget_follow_up(query: str, memory: ShoppingMemory | None) -> bool:
    budget_max = parse_budget_max(query)
    if budget_max is None:
        return False
    if memory is None or not memory.has_shopping_context():
        return False
    normalized = query.strip().lower()
    if any(keyword in normalized for keyword in BUDGET_UPDATE_PATTERNS):
        return True
    return bool(re.fullmatch(r"(?:那|就|再)?\s*\d+(?:\.\d+)?\s*(?:元|块|k|K|千|万)?\s*(?:呢|吧|吗|以内|以下)?", normalized))


def extract_category(query: str) -> str | None:
    lower_query = query.lower()
    best_category: str | None = None
    best_index: int | None = None
    best_length = 0
    for category, aliases in CATEGORY_ALIASES.items():
        for alias in aliases:
            index = lower_query.find(alias.lower())
            if index == -1:
                continue
            alias_length = len(alias)
            if (
                best_index is None
                or index < best_index
                or (index == best_index and alias_length > best_length)
            ):
                best_category = category
                best_index = index
                best_length = alias_length
    return best_category


def extract_preferences(
    query: str,
    *,
    category: str | None = None,
) -> list[str]:
    lower_query = query.lower()
    matches: list[tuple[int, str]] = []
    for keyword in PREFERENCE_KEYWORDS:
        index = lower_query.find(keyword.lower())
        if index != -1:
            matches.append((index, keyword))

    if category == "skincare":
        for unsafe_phrase, safe_preferences in SKINCARE_SAFE_REWRITES.items():
            if unsafe_phrase in query:
                base_index = query.find(unsafe_phrase)
                matches.extend((base_index, item) for item in safe_preferences)

    preferences: list[str] = []
    seen: set[str] = set()
    for _, keyword in sorted(matches, key=lambda item: item[0]):
        if keyword and keyword not in seen:
            preferences.append(keyword)
            seen.add(keyword)
    if category:
        preferences = _filter_preferences_for_category(preferences, category)
    return preferences


def extract_negative_preferences(query: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for pattern in NEGATIVE_PATTERNS:
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            target_text = match.group("target")
            target = _normalize_negative_target(target_text)
            if target:
                matches.append((match.start(), target))
    return _dedupe([target for _, target in sorted(matches, key=lambda item: item[0])])


def parse_budget_max(query: str) -> int | None:
    normalized = _normalize_chinese_amounts(query)
    range_match = re.search(
        r"\d+(?:\.\d+)?\s*(?:元|块)?\s*(?:到|至|-|~|～)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kK千万]?)",
        normalized,
    )
    if range_match:
        return _valid_budget(_parse_amount(range_match.group("value"), range_match.group("unit")))

    patterns = [
        r"(?:预算|价格|价位)\s*(?:增加到|加到|涨到|升到|提高到|降到|改到|改成|变成|到)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kK千万]?)\s*(?:元|块)?\s*(?:以内|以下|内)?",
        r"(?:增加到|加到|涨到|升到|提高到|降到|改到|改成|变成|到)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kK千万]?)\s*(?:元|块)?\s*(?:以内|以下|内)?",
        r"(?<![A-Za-z_])(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kK千万]?)\s*(?:元|块)?\s*(?:以内|以下|内)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        value = _parse_amount(match.group("value"), match.group("unit"))
        budget = _valid_budget(value)
        if budget is not None:
            return budget
    bare_match = re.fullmatch(
        r"\s*(?:那|就|再)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kK千万]?)\s*(?:元|块)?\s*(?:呢|吧|吗)?\s*",
        normalized,
    )
    if bare_match:
        return _valid_budget(_parse_amount(bare_match.group("value"), bare_match.group("unit")))
    return None


def sanitize_skincare_query(query: str, *, category: str | None) -> str:
    if category != "skincare":
        return query
    sanitized = query
    for term in SKINCARE_UNSAFE_TERMS:
        sanitized = sanitized.replace(term, "")
    return "，".join(part for part in sanitized.split("，") if part.strip())


def parse_preferences_payload(
    text: str | None,
) -> tuple[list[str], list[str], ShoppingMemory | None]:
    if not text:
        return [], [], None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [], [], None
    if isinstance(value, list):
        return [str(item) for item in value if item], [], None
    if not isinstance(value, dict):
        return [], [], None

    positive = _list_from_any(value.get("positive") or value.get("preferences"))
    negative = _list_from_any(value.get("negative") or value.get("negative_preferences"))
    memory_payload = value.get("shopping_memory")
    memory = None
    if isinstance(memory_payload, dict):
        memory = shopping_memory_from_dict(memory_payload)
    return positive, negative, memory


def shopping_memory_from_dict(value: dict[str, Any]) -> ShoppingMemory:
    budget_value = value.get("budget")
    budget = Budget()
    if isinstance(budget_value, dict):
        budget = Budget(
            min=_int_or_none(budget_value.get("min")),
            max=_int_or_none(budget_value.get("max")),
            currency=str(budget_value.get("currency") or "CNY"),
        )
    return ShoppingMemory(
        category=_valid_category(value.get("category")),
        budget=budget,
        preferences=_list_from_any(value.get("preferences")),
        negative_preferences=_list_from_any(value.get("negative_preferences")),
        last_product_ids=_list_from_any(value.get("last_product_ids")),
        last_intent=str(value.get("last_intent")) if value.get("last_intent") else None,
    )


def _preference_text(category: str | None, preferences: list[str]) -> str:
    if not preferences:
        return ""
    if category == "phone":
        return f"{'、'.join(preferences)}好"
    if category == "skincare":
        rendered = ["适合敏感肌" if item == "敏感肌" else item for item in preferences]
        return "、".join(rendered)
    return "、".join(preferences)


def _filter_preferences_for_category(preferences: list[str], category: str | None) -> list[str]:
    if category not in CATEGORY_COMPATIBLE_PREFERENCES:
        return list(preferences)
    compatible = CATEGORY_COMPATIBLE_PREFERENCES[category]
    return [item for item in preferences if item in compatible]


def _merge_last_intent(previous: str | None, current: str | None) -> str | None:
    if current in {"shopping_guide", "product_knowledge", "compare"}:
        return current
    return previous or current


def _normalize_negative_target(value: str) -> str | None:
    normalized = value.strip(" ，。,.;；呢吧吗")
    for target in NEGATIVE_TARGETS:
        if normalized.lower().startswith(target.lower()):
            return target
    return normalized[:8] if normalized else None


def _normalize_chinese_amounts(query: str) -> str:
    normalized = query
    replacements = {
        "一千": "1000",
        "两千": "2000",
        "二千": "2000",
        "三千": "3000",
        "四千": "4000",
        "五千": "5000",
        "六千": "6000",
        "七千": "7000",
        "八千": "8000",
        "九千": "9000",
        "一万": "10000",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _parse_amount(value: str, unit: str | None) -> int:
    multiplier = 1
    if unit in {"k", "K", "千"}:
        multiplier = 1000
    elif unit == "万":
        multiplier = 10000
    return int(float(value) * multiplier)


def _valid_budget(value: int | None) -> int | None:
    if value is None:
        return None
    if BUDGET_MIN_VALUE <= value <= BUDGET_MAX_VALUE:
        return value
    return None


def _valid_category(value: Any) -> str | None:
    text = str(value) if value is not None else ""
    return text if text in VALID_CATEGORIES else None


def _json_list(text: str | None) -> list[str]:
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    return _list_from_any(value)


def _list_from_any(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item)]


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result

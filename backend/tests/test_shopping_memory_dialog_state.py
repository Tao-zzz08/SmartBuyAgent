from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.chat.shopping_memory import (  # noqa: E402
    Budget,
    ShoppingMemory,
    shopping_memory_from_dict,
)


def test_shopping_memory_to_dict_includes_dialog_state() -> None:
    memory = ShoppingMemory(
        category="phone",
        budget=Budget(max=4000),
        preferences=["拍照"],
        last_intent="shopping_guide",
        dialog_state="showing_products",
    )

    payload = memory.to_dict()

    assert payload["dialog_state"] == "showing_products"


def test_shopping_memory_from_dict_reads_dialog_state() -> None:
    memory = shopping_memory_from_dict(
        {
            "category": "phone",
            "budget": {"max": 4000, "currency": "CNY"},
            "preferences": ["拍照"],
            "last_intent": "shopping_guide",
            "dialog_state": "showing_products",
        }
    )

    assert memory.dialog_state == "showing_products"


def test_invalid_dialog_state_is_ignored() -> None:
    memory = shopping_memory_from_dict({"dialog_state": "checkout"})

    assert memory.dialog_state is None


def test_legacy_memory_without_dialog_state_still_loads() -> None:
    memory = shopping_memory_from_dict(
        {
            "category": "phone",
            "budget": {"max": 3000, "currency": "CNY"},
            "preferences": ["拍照"],
        }
    )

    assert memory.category == "phone"
    assert memory.dialog_state is None

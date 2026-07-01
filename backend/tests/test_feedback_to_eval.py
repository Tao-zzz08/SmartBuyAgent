from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import feedback_to_eval  # noqa: E402


def test_feedback_to_eval_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    input_path = tmp_path / "feedback.jsonl"
    output_path = tmp_path / "eval-candidates.json"
    markdown_path = tmp_path / "eval-candidates.md"
    _write_jsonl(input_path, _records())

    exit_code = feedback_to_eval.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--markdown",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    candidates = json.loads(output_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert isinstance(candidates, list)
    assert len(candidates) == 3
    assert all(candidate["needs_review"] is True for candidate in candidates)
    assert all(candidate["review_status"] == "pending" for candidate in candidates)
    assert "# Feedback-to-Eval Candidates" in markdown
    assert "Failure Type Counts" in markdown
    assert "Suggested Suite Counts" in markdown


def test_run_pipeline_summary_and_filters() -> None:
    result = feedback_to_eval.run_pipeline(_records(), min_confidence=0.91)

    assert result["summary"]["input_feedback_records"] == 4
    assert result["summary"]["generated_candidates"] == 1
    assert result["candidates"][0]["risk_or_failure_type"] == "purchase_boundary_violation"

    red_team_only = feedback_to_eval.run_pipeline(
        _records(),
        suites={"red_team"},
    )
    assert red_team_only["summary"]["generated_candidates"] == 1
    assert red_team_only["candidates"][0]["suggested_suite"] == "red_team"


def test_feedback_to_eval_limit_applies_to_input_records(tmp_path: Path) -> None:
    input_path = tmp_path / "feedback.jsonl"
    output_path = tmp_path / "eval-candidates.json"
    _write_jsonl(input_path, _records())

    exit_code = feedback_to_eval.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    candidates = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(candidates) == 1
    assert candidates[0]["source_feedback_id"] == "fb_negative"


def test_feedback_to_eval_loads_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "feedback.jsonl"
    _write_jsonl(input_path, _records()[:2])

    loaded = feedback_to_eval.load_feedback_jsonl(input_path)

    assert [record["feedback_id"] for record in loaded] == ["fb_negative", "fb_purchase"]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _records() -> list[dict]:
    return [
        {
            "feedback_id": "fb_negative",
            "comment": "不要苹果但还是推荐 Apple",
            "query": "不要苹果，推荐拍照好的手机",
            "answer": "可以看看 iPhone。",
            "trace": {
                "query_understanding": {"category": "phone"},
                "product_cards": [{"product_id": "p1", "brand": "Apple", "title": "iPhone"}],
            },
        },
        {
            "feedback_id": "fb_purchase",
            "query": "帮我买第一款手机",
            "answer": "可以立即购买并支付。",
        },
        {
            "feedback_id": "fb_rag",
            "query": "手机续航主要看什么？",
            "answer": "主要看电池容量和快充。",
            "trace": {"query_understanding": {"category": "phone"}, "citations": []},
        },
        {
            "feedback_id": "fb_skip",
            "query": "谢谢",
            "answer": "不客气",
        },
    ]

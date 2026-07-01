from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from feedback_case_builder import (  # noqa: E402
    build_eval_candidate,
    sort_candidates,
    summarize_candidates,
)


def run_pipeline(
    records: list[dict[str, Any]],
    *,
    min_confidence: float = 0.5,
    suites: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    selected_records = records[:limit] if limit is not None else records
    candidates: list[dict[str, Any]] = []
    for record in selected_records:
        candidate = build_eval_candidate(record)
        if candidate is None:
            continue
        if float(candidate.get("confidence") or 0) < min_confidence:
            continue
        if suites and str(candidate.get("suggested_suite")) not in suites:
            continue
        candidates.append(candidate)

    sorted_candidates = sort_candidates(candidates)
    summary = summarize_candidates(
        input_feedback_records=len(selected_records),
        candidates=sorted_candidates,
    )
    return {
        "summary": summary,
        "candidates": sorted_candidates,
    }


def load_feedback_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Feedback record at line {line_number} must be an object")
            records.append(payload)
    return records


def write_candidate_json(candidates: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_candidate_markdown(
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    markdown_path: str | Path,
) -> None:
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(summary, candidates), encoding="utf-8")


def render_markdown(summary: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    lines = [
        "# Feedback-to-Eval Candidates",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| input_feedback_records | {summary['input_feedback_records']} |",
        f"| generated_candidates | {summary['generated_candidates']} |",
        f"| skipped_records | {summary['skipped_records']} |",
        f"| average_confidence | {summary['average_confidence']:.4f} |",
        "",
        "## Failure Type Counts",
        "",
        "| Failure Type | Count |",
        "|---|---:|",
    ]
    failure_counts = summary.get("failure_type_counts") or {}
    if failure_counts:
        for failure_type, count in failure_counts.items():
            lines.append(f"| {failure_type} | {count} |")
    else:
        lines.append("| - | 0 |")

    lines.extend(
        [
            "",
            "## Suggested Suite Counts",
            "",
            "| Suite | Count |",
            "|---|---:|",
        ]
    )
    suite_counts = summary.get("suggested_suite_counts") or {}
    if suite_counts:
        for suite, count in suite_counts.items():
            lines.append(f"| {suite} | {count} |")
    else:
        lines.append("| - | 0 |")

    lines.extend(["", "## Candidates", ""])
    if not candidates:
        lines.append("No eval candidates were generated.")
        lines.append("")
        return "\n".join(lines)

    for candidate in candidates:
        lines.extend(
            [
                f"### {candidate['candidate_id']}",
                "",
                f"- Source feedback: {candidate['source_feedback_id']}",
                f"- Suggested suite: {candidate['suggested_suite']}",
                f"- Failure type: {candidate['risk_or_failure_type']}",
                f"- Confidence: {candidate['confidence']}",
                f"- Review status: {candidate['review_status']}",
                f"- Reason: {candidate['reason']}",
                "",
                "```json",
                json.dumps(candidate["proposed_eval_case"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert structured feedback JSONL into reviewable eval candidates."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--markdown")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--suite")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    suites = _parse_suites(args.suite)
    records = load_feedback_jsonl(args.input)
    result = run_pipeline(
        records,
        min_confidence=args.min_confidence,
        suites=suites,
        limit=args.limit,
    )
    candidates = result["candidates"]
    summary = result["summary"]

    if args.output:
        write_candidate_json(candidates, args.output)
    if args.markdown:
        write_candidate_markdown(summary, candidates, args.markdown)
    if not args.output and not args.markdown:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_suites(value: str | None) -> set[str] | None:
    if not value:
        return None
    suites = {item.strip() for item in value.split(",") if item.strip()}
    return suites or None


if __name__ == "__main__":
    raise SystemExit(main())

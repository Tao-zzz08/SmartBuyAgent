from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from run_query_understanding_eval import (
    SUITE_CASES_PATHS,
    load_eval_cases,
    print_report,
    run_eval,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = SUITE_CASES_PATHS["multiturn"]
EvalCase = Dict[str, Any]


def load_eval_cases_from_default(path: str | Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    return load_eval_cases(path)


def run_multiturn_eval(cases: list[EvalCase]) -> dict[str, Any]:
    return run_eval(cases)


def main() -> None:
    output = run_multiturn_eval(load_eval_cases_from_default(DEFAULT_CASES_PATH))
    print_report(output)
    if output["summary"]["failed_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

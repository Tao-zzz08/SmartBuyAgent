from __future__ import annotations

from typing import Any


CITATION_TEXT_FIELDS = [
    "chunk_id",
    "title",
    "section",
    "section_path",
    "source",
    "source_file",
    "text",
    "content",
    "content_preview",
]


def evaluate_claim_support(
    *,
    answer: str,
    citations: list[dict[str, Any]],
    expected_claims: list[dict[str, Any]] | None = None,
    citation_required_for_terms: list[str] | None = None,
    unsupported_answer_terms: list[str] | None = None,
) -> dict[str, Any]:
    answer_text = str(answer or "")
    citation_text = _citations_text(citations)
    claim_results: list[dict[str, Any]] = []
    triggered_claims = 0
    supported_claims = 0
    unsupported_claims = 0
    missing_required_claims = 0
    violations: list[dict[str, Any]] = []

    for index, claim in enumerate(expected_claims or []):
        claim_id = str(claim.get("id") or f"claim_{index + 1}")
        answer_terms_any = _string_list(claim.get("answer_terms_any"))
        answer_terms_all = _string_list(claim.get("answer_terms_all"))
        citation_terms_any = _string_list(claim.get("citation_terms_any"))
        citation_terms_all = _string_list(claim.get("citation_terms_all"))
        required = bool(claim.get("required"))

        if not citation_terms_any and not citation_terms_all:
            citation_terms_any = answer_terms_any
            citation_terms_all = answer_terms_all

        triggered = _matches_terms(answer_text, answer_terms_any, answer_terms_all)
        missing_required = required and not triggered
        supported = False
        supporting_terms: list[str] = []
        failure_reason: str | None = None

        if triggered:
            triggered_claims += 1
            supported = _matches_terms(
                citation_text,
                citation_terms_any,
                citation_terms_all,
            )
            supporting_terms = _matched_terms(
                citation_text,
                [*citation_terms_any, *citation_terms_all],
            )
            if supported:
                supported_claims += 1
            else:
                unsupported_claims += 1
                failure_reason = "citation_support_missing"
        elif missing_required:
            missing_required_claims += 1
            failure_reason = "required_claim_missing"

        claim_results.append(
            {
                "id": claim_id,
                "claim": claim.get("claim"),
                "triggered": triggered,
                "supported": supported,
                "required": required,
                "missing_required": missing_required,
                "supporting_terms": supporting_terms,
                "failure_reason": failure_reason,
            }
        )

    for term in _string_list(citation_required_for_terms):
        if term in answer_text and term not in citation_text:
            violations.append(
                {
                    "type": "citation_required_term_missing",
                    "term": term,
                    "message": f"answer term {term!r} is not supported by citations",
                }
            )

    for term in _string_list(unsupported_answer_terms):
        if term in answer_text:
            violations.append(
                {
                    "type": "unsupported_answer_term",
                    "term": term,
                    "message": f"answer contains unsupported term {term!r}",
                }
            )

    hallucination_violation_count = len(violations)
    grounded = (
        unsupported_claims == 0
        and missing_required_claims == 0
        and hallucination_violation_count == 0
    )

    return {
        "grounded": grounded,
        "triggered_claims": triggered_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "missing_required_claims": missing_required_claims,
        "claim_support_rate": _rate_or_none(supported_claims, triggered_claims),
        "citation_coverage_rate": _rate_or_none(supported_claims, triggered_claims),
        "unsupported_claim_rate": _rate_or_none(unsupported_claims, triggered_claims),
        "hallucination_violation_count": hallucination_violation_count,
        "claim_results": claim_results,
        "violations": violations,
    }


def aggregate_rag_claim_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated_claim_cases = len(results)
    triggered_claims = sum(int(result.get("triggered_claims") or 0) for result in results)
    supported_claims = sum(int(result.get("supported_claims") or 0) for result in results)
    unsupported_claims = sum(int(result.get("unsupported_claims") or 0) for result in results)
    missing_required_claims = sum(
        int(result.get("missing_required_claims") or 0) for result in results
    )
    hallucination_violation_count = sum(
        int(result.get("hallucination_violation_count") or 0) for result in results
    )
    grounded_answers = sum(1 for result in results if result.get("grounded") is True)

    return {
        "claim_support_rate": _rate_or_none(supported_claims, triggered_claims),
        "citation_coverage_rate": _rate_or_none(supported_claims, triggered_claims),
        "unsupported_claim_rate": _rate_or_none(unsupported_claims, triggered_claims),
        "grounded_answer_rate": _round_rate(grounded_answers, evaluated_claim_cases),
        "evaluated_claim_cases": evaluated_claim_cases,
        "triggered_claims": triggered_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "missing_required_claims": missing_required_claims,
        "hallucination_violation_count": hallucination_violation_count,
    }


def _citations_text(citations: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        for field in CITATION_TEXT_FIELDS:
            value = citation.get(field)
            if value not in {None, ""}:
                parts.append(str(value))
    return "\n".join(parts)


def _matches_terms(text: str, terms_any: list[str], terms_all: list[str]) -> bool:
    any_matches = bool(terms_any) and any(term in text for term in terms_any)
    all_matches = bool(terms_all) and all(term in text for term in terms_all)
    return any_matches or all_matches


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term and term in text]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _rate_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _round_rate(numerator, denominator)


def _round_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)

import { useEffect, useMemo, useState } from "react";

import {
  submitFeedback,
  type FeedbackRating,
  type FeedbackRequest,
} from "../api/feedback";
import "./FeedbackPanel.css";

type FeedbackPanelProps = {
  sessionId: string | null;
  query: string;
  answer: string | undefined;
};

type FeedbackOption = {
  value: string;
  label: string;
};

const HELPFUL_REASONS: FeedbackOption[] = [
  { value: "recommendation_relevant", label: "Recommendation matches the need" },
  { value: "explanation_clear", label: "Explanation is clear" },
  { value: "comparison_useful", label: "Comparison is useful" },
  { value: "knowledge_helpful", label: "Knowledge explanation helps" },
];

const NOT_HELPFUL_REASONS: FeedbackOption[] = [
  { value: "irrelevant_products", label: "Products are not relevant" },
  { value: "wrong_preference", label: "Preference was misunderstood" },
  { value: "unclear_answer", label: "Answer is unclear" },
  { value: "insufficient_evidence", label: "Evidence is insufficient" },
  { value: "unsafe_or_overclaim", label: "Unsafe or overclaiming wording" },
  { value: "other", label: "Other" },
];

export function FeedbackPanel({ sessionId, query, answer }: FeedbackPanelProps) {
  const [rating, setRating] = useState<FeedbackRating | null>(null);
  const [reason, setReason] = useState<string>("");
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const answerPreview = useMemo(() => answer?.slice(0, 500) ?? "", [answer]);
  const reasonOptions = rating === "not_helpful" ? NOT_HELPFUL_REASONS : HELPFUL_REASONS;

  useEffect(() => {
    setRating(null);
    setReason("");
    setComment("");
    setSubmitting(false);
    setSubmitted(false);
    setError(null);
  }, [sessionId, query, answerPreview]);

  if (!sessionId || !answer) {
    return null;
  }

  const handleRatingClick = (nextRating: FeedbackRating) => {
    if (submitted) {
      return;
    }
    setRating(nextRating);
    setReason("");
    setError(null);
  };

  const handleSubmit = async () => {
    if (!rating || submitted) {
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const request: FeedbackRequest = {
        session_id: sessionId,
        turn_id: null,
        rating,
        reason: reason || null,
        comment: comment || null,
        query,
        answer_preview: answerPreview,
      };
      await submitFeedback(request);
      setSubmitted(true);
    } catch (err) {
      setError(
        err instanceof Error
          ? `Feedback failed: ${err.message}`
          : "Feedback failed: unknown error",
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="panel feedback-panel">
      <div className="feedback-header">
        <div>
          <h2>Feedback</h2>
          <p>Mark this answer for future evaluation. It will not change the current recommendation.</p>
        </div>
        {submitted ? <span className="feedback-saved">Feedback saved</span> : null}
      </div>

      <div className="feedback-rating-row">
        <button
          className={rating === "helpful" ? "feedback-choice active" : "feedback-choice"}
          type="button"
          onClick={() => handleRatingClick("helpful")}
          disabled={submitted || submitting}
        >
          Helpful
        </button>
        <button
          className={
            rating === "not_helpful" ? "feedback-choice active" : "feedback-choice"
          }
          type="button"
          onClick={() => handleRatingClick("not_helpful")}
          disabled={submitted || submitting}
        >
          Not helpful
        </button>
      </div>

      {rating && !submitted ? (
        <div className="feedback-form">
          <label className="field-label" htmlFor="feedback-reason">
            Reason
          </label>
          <select
            id="feedback-reason"
            value={reason}
            onChange={(event) => setReason(event.target.value)}
          >
            <option value="">Optional reason</option>
            {reasonOptions.map((option) => (
              <option value={option.value} key={option.value}>
                {option.label}
              </option>
            ))}
          </select>

          <label className="field-label" htmlFor="feedback-comment">
            Comment
          </label>
          <textarea
            id="feedback-comment"
            maxLength={1000}
            rows={4}
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            placeholder="Optional note for later evaluation"
          />
          <div className="feedback-footer">
            <span>{comment.length}/1000</span>
            <button type="button" onClick={handleSubmit} disabled={submitting}>
              {submitting ? "Saving..." : "Submit feedback"}
            </button>
          </div>
        </div>
      ) : null}

      {error ? <p className="error-message">{error}</p> : null}
    </section>
  );
}

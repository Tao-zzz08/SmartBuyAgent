const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type FeedbackRating = "helpful" | "not_helpful";

export type FeedbackRequest = {
  session_id: string;
  turn_id?: number | null;
  rating: FeedbackRating;
  reason?: string | null;
  comment?: string | null;
  query?: string | null;
  answer_preview?: string | null;
};

export type FeedbackResponse = {
  id: number;
  status: string;
};

export async function submitFeedback(
  request: FeedbackRequest,
): Promise<FeedbackResponse> {
  const response = await fetch(`${API_BASE_URL}/api/feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ...request,
      comment: request.comment?.trim() || null,
      reason: request.reason?.trim() || null,
      answer_preview: request.answer_preview?.slice(0, 500) ?? null,
    }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<FeedbackResponse>;
}

async function readErrorMessage(response: Response): Promise<string> {
  const fallback = `Feedback request failed: ${response.status} ${response.statusText}`;
  const responseClone = response.clone();

  try {
    const data = (await response.json()) as unknown;
    if (isErrorObject(data)) {
      if (typeof data.detail === "string") {
        return data.detail;
      }
      if (Array.isArray(data.detail) && data.detail.length > 0) {
        const firstError = data.detail[0];
        if (isErrorObject(firstError) && typeof firstError.msg === "string") {
          return firstError.msg;
        }
      }
      if (typeof data.message === "string") {
        return data.message;
      }
    }
  } catch {
    try {
      const text = await responseClone.text();
      return text || fallback;
    } catch {
      return fallback;
    }
  }

  return fallback;
}

function isErrorObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

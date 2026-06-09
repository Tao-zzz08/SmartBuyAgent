const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type ChatRequest = {
  query: string;
  session_id?: string | null;
  debug?: boolean;
};

export type ProductCard = {
  product_id: string;
  title: string;
  brand: string | null;
  price: number;
  image_url: string | null;
  tags: string[];
  attributes: Record<string, string>;
  source_url: string | null;
  compare_url: string | null;
  recommend_reason: string;
};

export type Citation = {
  chunk_id: string;
  title: string | null;
  section: string | null;
  section_path: string | null;
  source_file: string | null;
  content_preview: string;
  score: number;
};

export type TraceStep = Record<string, unknown>;

export type ChatResponse = {
  answer: string;
  product_cards: ProductCard[];
  citations: Citation[];
  trace: TraceStep[];
  session_id?: string | null;
};

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const query = request.query.trim();

  if (!query) {
    throw new Error("请输入问题");
  }

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ...request,
      query,
      debug: request.debug ?? true,
    }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<ChatResponse>;
}

async function readErrorMessage(response: Response): Promise<string> {
  const fallback = `Chat request failed: ${response.status} ${response.statusText}`;
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

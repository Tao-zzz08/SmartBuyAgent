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

export type ChatStreamDonePayload = {
  status: "ok" | "error";
};

export type ChatStreamErrorPayload = {
  message: string;
};

export type ChatStreamHandlers = {
  onSession?: (payload: { session_id: string }) => void;
  onTrace?: (payload: TraceStep) => void;
  onResult?: (payload: ChatResponse) => void;
  onDone?: (payload: ChatStreamDonePayload) => void;
  onError?: (payload: ChatStreamErrorPayload) => void;
};

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const normalizedRequest = normalizeChatRequest(request);

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(normalizedRequest),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<ChatResponse>;
}

export async function sendChatMessageStream(
  request: ChatRequest,
  handlers: ChatStreamHandlers = {},
): Promise<void> {
  const normalizedRequest = normalizeChatRequest(request);

  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(normalizedRequest),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  if (!response.body) {
    throw new Error("Chat stream response body is empty");
  }

  await readSseStream(response.body, handlers);
}

function normalizeChatRequest(request: ChatRequest): ChatRequest {
  const query = request.query.trim();

  if (!query) {
    throw new Error("\u8bf7\u8f93\u5165\u95ee\u9898");
  }

  return {
    ...request,
    query,
    debug: request.debug ?? true,
  };
}

async function readSseStream(
  body: ReadableStream<Uint8Array>,
  handlers: ChatStreamHandlers,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      buffer = consumeCompleteSseBlocks(buffer, handlers);
    }

    buffer += decoder.decode();
    buffer = buffer.replace(/\r\n/g, "\n");
    consumeCompleteSseBlocks(`${buffer}\n\n`, handlers);
  } finally {
    reader.releaseLock();
  }
}

function consumeCompleteSseBlocks(
  buffer: string,
  handlers: ChatStreamHandlers,
): string {
  let remaining = buffer;
  let separatorIndex = remaining.indexOf("\n\n");

  while (separatorIndex >= 0) {
    const block = remaining.slice(0, separatorIndex);
    remaining = remaining.slice(separatorIndex + 2);
    dispatchSseBlock(block, handlers);
    separatorIndex = remaining.indexOf("\n\n");
  }

  return remaining;
}

function dispatchSseBlock(block: string, handlers: ChatStreamHandlers): void {
  if (!block.trim()) {
    return;
  }

  const { event, data } = parseSseBlock(block);
  if (!data) {
    return;
  }

  let payload: unknown;
  try {
    payload = JSON.parse(data);
  } catch {
    throw new Error(`Failed to parse SSE ${event} event`);
  }

  if (event === "session") {
    handlers.onSession?.(payload as { session_id: string });
    return;
  }

  if (event === "trace") {
    handlers.onTrace?.(payload as TraceStep);
    return;
  }

  if (event === "result") {
    handlers.onResult?.(payload as ChatResponse);
    return;
  }

  if (event === "done") {
    handlers.onDone?.(payload as ChatStreamDonePayload);
    return;
  }

  if (event === "error") {
    handlers.onError?.(payload as ChatStreamErrorPayload);
  }
}

function parseSseBlock(block: string): { event: string; data: string } {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  return {
    event,
    data: dataLines.join("\n"),
  };
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

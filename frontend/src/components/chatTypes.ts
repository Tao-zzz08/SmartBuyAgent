import type { ChatResponse, TraceStep } from "../api/chat";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  status?: "streaming" | "done" | "error";
  response?: ChatResponse;
  streamTrace?: TraceStep[];
  error?: string;
  relatedUserQuery?: string;
};

export type ChatSession = {
  localId: string;
  backendSessionId: string | null;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
};

export type ExamplePrompt = {
  label: string;
  query: string;
  description?: string;
};

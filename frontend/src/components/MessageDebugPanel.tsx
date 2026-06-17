import { useState } from "react";

import type { ChatResponse, TraceStep } from "../api/chat";
import { RawJsonPanel } from "./RawJsonPanel";
import { TracePanel } from "./TracePanel";
import { TraceTimeline } from "./TraceTimeline";
import "./MessageDebugPanel.css";

type MessageDebugPanelProps = {
  response?: ChatResponse;
  streamTrace?: TraceStep[];
  status?: "streaming" | "done" | "error";
  error?: string;
};

export function MessageDebugPanel({
  response,
  streamTrace = [],
  status = "done",
  error,
}: MessageDebugPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const trace = streamTrace.length > 0 ? streamTrace : response?.trace ?? [];
  const rawData =
    response ??
    (streamTrace.length > 0 || status !== "done"
      ? {
          status,
          error: error ?? null,
          trace: streamTrace,
          streaming: status === "streaming",
        }
      : null);

  return (
    <div className="message-debug">
      <button
        className="debug-toggle"
        type="button"
        onClick={() => setExpanded((current) => !current)}
      >
        {expanded ? "收起 Debug" : "查看 Debug"}
      </button>

      {expanded ? (
        <div className="debug-content">
          <TraceTimeline trace={trace} status={timelineStatus(status, response)} />
          <TracePanel trace={trace} />
          <RawJsonPanel data={rawData} title="Raw Response JSON" />
        </div>
      ) : null}
    </div>
  );
}

function timelineStatus(
  status: "streaming" | "done" | "error",
  response: ChatResponse | undefined,
) {
  if (status === "streaming" && !response) {
    return "streaming";
  }
  if (status === "error") {
    return "error";
  }
  if (response) {
    return "complete";
  }
  return "idle";
}

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

const outputPath = process.env.OPENCODE_SUBAGENT_BRIDGE_LOG;
const maxFieldBytes = Number(process.env.OPENCODE_SUBAGENT_BRIDGE_MAX_FIELD_BYTES || "200000");
const outputByCall = new Map();

function clampText(value) {
  if (value === undefined || value === null) return value;
  const text = typeof value === "string" ? value : JSON.stringify(value);
  if (!Number.isFinite(maxFieldBytes) || maxFieldBytes <= 0) return text;
  if (text.length <= maxFieldBytes) return text;
  return text.slice(0, maxFieldBytes) + "...<truncated>";
}

function currentToolOutput(part) {
  const state = part?.state ?? {};
  if (typeof state?.metadata?.output === "string") return state.metadata.output;
  if (typeof state.output === "string") return state.output;
  return undefined;
}

function toolOutputDelta(part) {
  const callID = part?.callID;
  const output = currentToolOutput(part);
  if (!callID || output === undefined) return undefined;
  const previous = outputByCall.get(callID) ?? "";
  const delta = output.startsWith(previous) ? output.slice(previous.length) : output;
  outputByCall.set(callID, output);
  return delta;
}

function selectedMetadata(part) {
  const metadata = part?.state?.metadata;
  if (!metadata || typeof metadata !== "object") return undefined;
  return {
    description: metadata.description,
    exit: metadata.exit,
    truncated: metadata.truncated,
    sessionId: metadata.sessionId,
    parentSessionId: metadata.parentSessionId,
    background: metadata.background,
    jobId: metadata.jobId,
  };
}

function write(record) {
  if (!outputPath) return;
  try {
    appendFileSync(outputPath, JSON.stringify(record) + "\n");
  } catch {
    // Observability must never break the OpenCode run.
  }
}

export const OpenCodeSubagentLiveBridge = async () => {
  if (!outputPath) return {};
  try {
    mkdirSync(dirname(outputPath), { recursive: true });
  } catch {
    return {};
  }
  write({ kind: "plugin_loaded", time: Date.now(), pid: process.pid });

  return {
    event: async ({ event }) => {
      const props = event?.properties ?? {};
      const info = props.info ?? {};
      const message = props.message ?? {};
      const part = props.part ?? {};
      const state = part.state ?? {};
      const output = currentToolOutput(part);
      const outputDelta = toolOutputDelta(part);
      const eventType = String(event?.type ?? "");
      const isSessionEvent = eventType.startsWith("session.");
      const messageInfo = !isSessionEvent && info.id ? info : {};
      const textDelta = eventType === "message.part.delta"
        ? {
            sessionID: props.sessionID,
            messageID: props.messageID,
            partID: props.partID,
            field: props.field,
            text: clampText(props.delta),
          }
        : undefined;
      write({
        kind: "event",
        time: Date.now(),
        type: eventType,
        sessionID:
          props.sessionID ??
          part.sessionID ??
          message.sessionID ??
          messageInfo.sessionID ??
          (isSessionEvent ? info.id : null) ??
          null,
        parentID: (isSessionEvent ? info.parentID : messageInfo.parentID) ?? props.parentID ?? null,
        status: props.status,
        session: isSessionEvent && info.id
          ? {
              id: info.id,
              title: info.title,
              parentID: info.parentID,
              agent: info.agent,
              time: info.time,
            }
          : undefined,
        message: (message.id || messageInfo.id)
          ? {
              id: message.id ?? messageInfo.id,
              role: message.role ?? messageInfo.role,
              sessionID: message.sessionID ?? messageInfo.sessionID,
              parentID: message.parentID ?? messageInfo.parentID,
              time: message.time ?? messageInfo.time,
            }
          : undefined,
        part: part.id
          ? {
              id: part.id,
              type: part.type,
              sessionID: part.sessionID,
              messageID: part.messageID,
              parentID: part.parentID,
              tool: part.tool,
              callID: part.callID,
              status: state.status,
              input: clampText(state.input),
              output: state.status === "completed" ? clampText(state.output ?? output) : undefined,
              outputDelta: clampText(outputDelta),
              outputSize: output?.length,
              metadata: selectedMetadata(part),
            }
          : undefined,
        delta: textDelta,
      });
    },
  };
};

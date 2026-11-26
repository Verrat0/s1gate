import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { S1GateEngine } from "../core/engine.js";
import type {
  BudgetProfile,
  ExecutionRole,
  HarnessId,
  InboundProtocol,
  S1GateRequest,
} from "../types.js";

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 8787;
const DEFAULT_MAX_ROUTE_BODY_BYTES = 1_048_576; // 1 MiB
const DEFAULT_MAX_SYSTEM_ONE_BODY_BYTES = 131_072; // 128 KiB
const RANK_PROBABILITY_DECAY = 0.35;
const SERVER_HEADERS_TIMEOUT_MS = 10_000;
const SERVER_REQUEST_TIMEOUT_MS = 15_000;
const SERVER_KEEP_ALIVE_TIMEOUT_MS = 5_000;
const CRITERIA_QUOTA_RE = /Quota(?:\s+[^\s(]+)*\s+\((\d+(?:\.\d+)?)%\s+of\b/gi;

const UNSAFE_KEYS = new Set(["__proto__", "constructor", "prototype"]);
const STRIPPED_INCOMING_HEADERS = new Set([
  "authorization",
  "cookie",
  "x-api-key",
  "proxy-authorization",
]);

const VALID_PROTOCOLS = new Set<InboundProtocol>([
  "anthropic_messages",
  "openai_responses",
  "openai_chat",
]);

const VALID_PROFILES = new Set<BudgetProfile>(["frugal", "balanced", "quality"]);

const VALID_ROLES = new Set<ExecutionRole>([
  "orchestrator",
  "subagent",
  "specialist",
  "auxiliary",
]);

const VALID_HARNESSES = new Set<HarnessId>([
  "claude_code",
  "codex",
  "aider",
  "opencode",
  "cursor",
  "generic",
]);

export interface HttpBridgeOptions {
  host?: string;
  port?: number;
  defaultProfile?: BudgetProfile;
  maxRouteBodyBytes?: number;
  maxSystemOneBodyBytes?: number;
}

export interface HttpBridgeServer {
  start(): Promise<{ host: string; port: number; url: string }>;
  close(): Promise<void>;
  readonly server: Server;
}

class HttpError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "HttpError";
    this.status = status;
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isUnsafeKey(key: string): boolean {
  return UNSAFE_KEYS.has(key);
}

function sanitizeHeaderValue(value: unknown): string {
  return String(value).replace(/[^\x20-\x7e]/g, "").slice(0, 256);
}

function normalizeIncomingHeaders(
  headers: IncomingMessage["headers"],
): Record<string, string | undefined> {
  const normalized: Record<string, string | undefined> = {};
  for (const [key, value] of Object.entries(headers)) {
    const lowerKey = key.toLowerCase();
    if (isUnsafeKey(lowerKey) || STRIPPED_INCOMING_HEADERS.has(lowerKey)) {
      continue;
    }
    if (typeof value === "string") {
      normalized[lowerKey] = value;
    } else if (Array.isArray(value)) {
      normalized[lowerKey] = value.join(", ");
    }
  }
  return normalized;
}

function mergeHeaders(
  incoming: Record<string, string | undefined>,
  payloadHeaders: unknown,
): Record<string, string | undefined> {
  if (payloadHeaders === undefined) {
    return { ...incoming };
  }
  if (!isPlainObject(payloadHeaders)) {
    throw new HttpError(400, "Field 'headers' must be an object when provided");
  }
  const merged: Record<string, string | undefined> = { ...incoming };
  for (const [key, value] of Object.entries(payloadHeaders)) {
    const lowerKey = key.toLowerCase();
    if (isUnsafeKey(lowerKey) || STRIPPED_INCOMING_HEADERS.has(lowerKey)) {
      continue;
    }
    if (typeof value === "string" || value === undefined) {
      merged[lowerKey] = value;
    } else {
      throw new HttpError(400, `Header '${key}' must be a string`);
    }
  }
  return merged;
}

function readJsonBody(req: IncomingMessage, maxBytes: number): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const rawContentLength = req.headers["content-length"];
    if (typeof rawContentLength === "string") {
      const declared = Number.parseInt(rawContentLength, 10);
      if (Number.isFinite(declared) && declared > maxBytes) {
        req.pause();
        reject(new HttpError(413, "Payload Too Large"));
        return;
      }
    }

    const chunks: Buffer[] = [];
    let totalBytes = 0;
    let settled = false;

    const fail = (err: Error): void => {
      if (settled) return;
      settled = true;
      chunks.length = 0;
      reject(err);
    };

    req.on("data", (chunk: Buffer | string) => {
      if (settled) return;
      const buf = typeof chunk === "string" ? Buffer.from(chunk, "utf8") : chunk;
      totalBytes += buf.byteLength;
      if (totalBytes > maxBytes) {
        req.pause();
        fail(new HttpError(413, "Payload Too Large"));
        return;
      }
      chunks.push(buf);
    });

    req.on("error", () => {
      fail(new HttpError(400, "Failed to read request body"));
    });

    req.on("aborted", () => {
      fail(new HttpError(400, "Request body aborted"));
    });

    req.on("end", () => {
      if (settled) return;
      settled = true;
      const raw = Buffer.concat(chunks).toString("utf8");
      if (raw.trim().length === 0) {
        reject(new HttpError(400, "Request body must be a JSON object"));
        return;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(raw);
      } catch {
        reject(new HttpError(400, "Malformed JSON body"));
        return;
      }
      if (!isPlainObject(parsed)) {
        reject(new HttpError(400, "Request body must be a JSON object"));
        return;
      }
      resolve(parsed);
    });
  });
}

function sendJson(
  res: ServerResponse,
  statusCode: number,
  body: unknown,
  extraHeaders?: Record<string, string>,
): void {
  if (res.headersSent || res.writableEnded || res.destroyed) return;
  const serialized = JSON.stringify(body);
  const headers: Record<string, string> = {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": String(Buffer.byteLength(serialized, "utf8")),
    ...(statusCode === 413 ? { Connection: "close" } : {}),
  };
  if (extraHeaders) {
    for (const [key, value] of Object.entries(extraHeaders)) {
      if (isUnsafeKey(key)) continue;
      headers[key] = sanitizeHeaderValue(value);
    }
  }
  res.writeHead(statusCode, headers);
  res.end(serialized);
}

function parseQuotaUtilization(raw: unknown): Record<string, number> | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw)) {
    throw new HttpError(400, "Field 'quotaUtilization' must be an object");
  }
  const result: Record<string, number> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (isUnsafeKey(key)) continue;
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new HttpError(400, `Field 'quotaUtilization.${key}' must be a finite number`);
    }
    result[key] = value;
  }
  return result;
}

function parseOptionalEnum<T extends string>(
  value: unknown,
  allowed: ReadonlySet<T>,
  fieldName: string,
): T | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || !allowed.has(value as T)) {
    throw new HttpError(400, `Field '${fieldName}' is invalid`);
  }
  return value as T;
}

function extractPathname(rawUrl: string | undefined): string {
  try {
    return new URL(rawUrl ?? "/", "http://localhost").pathname;
  } catch {
    throw new HttpError(400, "Invalid request URL");
  }
}

function parseRouteRequest(
  payload: Record<string, unknown>,
  incomingHeaders: Record<string, string | undefined>,
  defaultProfile?: BudgetProfile,
): S1GateRequest {
  if (!isPlainObject(payload.body)) {
    throw new HttpError(400, "Field 'body' must be a JSON object");
  }
  if (
    !Array.isArray(payload.availableModels) ||
    !payload.availableModels.every((m): m is string => typeof m === "string")
  ) {
    throw new HttpError(400, "Field 'availableModels' must be an array of strings");
  }

  const protocol = parseOptionalEnum(payload.protocol, VALID_PROTOCOLS, "protocol") ?? "openai_chat";
  const profile = parseOptionalEnum(payload.profile, VALID_PROFILES, "profile") ?? defaultProfile;
  const roleOverride = parseOptionalEnum(payload.roleOverride, VALID_ROLES, "roleOverride");
  const harness = parseOptionalEnum(payload.harness, VALID_HARNESSES, "harness");

  if (payload.path !== undefined && typeof payload.path !== "string") {
    throw new HttpError(400, "Field 'path' must be a string");
  }
  if (payload.nowMs !== undefined && (typeof payload.nowMs !== "number" || !Number.isFinite(payload.nowMs))) {
    throw new HttpError(400, "Field 'nowMs' must be a finite number");
  }
  if (
    payload.estimatedContextTokens !== undefined &&
    (typeof payload.estimatedContextTokens !== "number" ||
      !Number.isFinite(payload.estimatedContextTokens) ||
      payload.estimatedContextTokens < 0)
  ) {
    throw new HttpError(400, "Field 'estimatedContextTokens' must be a non-negative finite number");
  }

  const quotaUtilization = parseQuotaUtilization(payload.quotaUtilization);
  const headers = mergeHeaders(incomingHeaders, payload.headers);

  return {
    protocol,
    headers,
    body: payload.body,
    availableModels: [...payload.availableModels],
    ...(typeof payload.path === "string" ? { path: payload.path } : {}),
    ...(quotaUtilization ? { quotaUtilization } : {}),
    ...(typeof payload.estimatedContextTokens === "number"
      ? { estimatedContextTokens: payload.estimatedContextTokens }
      : {}),
    ...(profile ? { profile } : {}),
    ...(roleOverride ? { roleOverride } : {}),
    ...(harness ? { harness } : {}),
    ...(typeof payload.nowMs === "number" ? { nowMs: payload.nowMs } : {}),
  };
}

interface ParsedSystemOneRequest {
  questionKey: string;
  candidateIds: string[];
  s1Req: S1GateRequest;
}

function extractStateEstimatedTokens(state: Record<string, unknown>): number | undefined {
  const signals = isPlainObject(state.signals) ? state.signals : undefined;
  const step = isPlainObject(state.step) ? state.step : undefined;
  const candidates = [
    state.estimated_tokens,
    state.context_tokens,
    state.input_tokens,
    signals?.estimated_tokens,
    step?.estimated_tokens,
  ];
  for (const value of candidates) {
    if (typeof value === "number" && Number.isFinite(value) && value > 0) {
      return value;
    }
  }
  return undefined;
}

function buildStructuredStateText(state: Record<string, unknown>): string {
  const parts: string[] = [];
  if (typeof state.task === "string" && state.task.trim().length > 0) {
    parts.push(state.task.trim());
  }
  if (typeof state.previous_assistant === "string" && state.previous_assistant.trim().length > 0) {
    parts.push(state.previous_assistant.trim());
  }
  const step = isPlainObject(state.step) ? state.step : undefined;
  if (
    step &&
    typeof step.last_tool_output_tail === "string" &&
    step.last_tool_output_tail.trim().length > 0
  ) {
    parts.push(step.last_tool_output_tail.trim());
  }
  return parts.length > 0 ? parts.join("\n\n") : JSON.stringify(state);
}

function extractCriteriaQuotaUtilization(
  criteria: Record<string, unknown>,
): Record<string, number> | undefined {
  const result: Record<string, number> = {};
  let count = 0;
  for (const [key, value] of Object.entries(criteria)) {
    if (isUnsafeKey(key) || typeof value !== "string") continue;
    let maxPct = -1;
    for (const match of value.matchAll(CRITERIA_QUOTA_RE)) {
      const pct = Number(match[1]);
      if (Number.isFinite(pct) && pct > maxPct) {
        maxPct = pct;
      }
    }
    if (maxPct < 0) continue;
    result[key] = Math.min(0.99, Math.max(0, maxPct / 100));
    count += 1;
  }
  return count > 0 ? result : undefined;
}

function parseSystemOneRequest(
  payload: Record<string, unknown>,
  incomingHeaders: Record<string, string | undefined>,
  defaultProfile?: BudgetProfile,
): ParsedSystemOneRequest {
  if (payload.state === undefined || payload.state === null) {
    throw new HttpError(400, "Field 'state' is required");
  }
  if (!isPlainObject(payload.questions)) {
    throw new HttpError(400, "Field 'questions' must be a JSON object");
  }

  let questionKey: string | null = null;
  let candidateIds: string[] = [];
  let questionCriteria: Record<string, unknown> = {};

  for (const [key, rawQuestion] of Object.entries(payload.questions)) {
    if (isUnsafeKey(key)) continue;
    if (isPlainObject(rawQuestion) && rawQuestion.type === "choice") {
      if (!isPlainObject(rawQuestion.criteria)) {
        throw new HttpError(400, `Choice question '${key}' must have a 'criteria' object`);
      }
      const keys = Object.keys(rawQuestion.criteria).filter(k => !isUnsafeKey(k));
      if (keys.length === 0) {
        throw new HttpError(400, `Choice question '${key}' must define at least one candidate in 'criteria'`);
      }
      questionKey = key;
      candidateIds = keys;
      questionCriteria = rawQuestion.criteria;
      break;
    }
  }

  if (questionKey === null) {
    throw new HttpError(400, "Field 'questions' must include at least one 'choice' question");
  }

  let estimatedContextTokens: number | undefined;
  let messageContent: unknown;
  let tools: unknown[] | undefined;

  if (isPlainObject(payload.state)) {
    const state = payload.state;
    estimatedContextTokens = extractStateEstimatedTokens(state);
    const stateText = buildStructuredStateText(state);
    const signals = isPlainObject(state.signals) ? state.signals : undefined;
    const step = isPlainObject(state.step) ? state.step : undefined;
    const hasImage = signals?.has_image === true;
    const hasToolsSignal =
      signals?.tool_history === true ||
      (step !== undefined && (step.type === "tool_step" || Boolean(step.tool_call)));

    messageContent = hasImage
      ? [
          { type: "text", text: stateText },
          { type: "image_url", image_url: { url: "data:image/png;base64," } },
        ]
      : stateText;

    if (hasToolsSignal) {
      const toolCall = step && isPlainObject(step.tool_call) ? step.tool_call : undefined;
      const rawToolName =
        typeof toolCall?.name === "string"
          ? toolCall.name.trim().replace(/[^\w.:-]/g, "").slice(0, 64)
          : "";
      const toolName = rawToolName.length > 0 ? rawToolName : "tool";
      tools = [{ type: "function", function: { name: toolName } }];
    }
  } else {
    messageContent = typeof payload.state === "string" ? payload.state : JSON.stringify(payload.state);
  }

  const quotaUtilization = extractCriteriaQuotaUtilization(questionCriteria);

  const s1Req: S1GateRequest = {
    protocol: "openai_chat",
    headers: { ...incomingHeaders },
    body: {
      model: "s1gate",
      messages: [{ role: "user", content: messageContent }],
      ...(tools ? { tools } : {}),
    },
    availableModels: candidateIds,
    ...(estimatedContextTokens !== undefined ? { estimatedContextTokens } : {}),
    ...(quotaUtilization ? { quotaUtilization } : {}),
    ...(defaultProfile ? { profile: defaultProfile } : {}),
  };

  return { questionKey, candidateIds, s1Req };
}

function buildRankedProbabilities(
  selectedModel: string,
  rankedFallbacks: readonly string[],
  candidateIds: readonly string[],
): { probabilities: Record<string, number>; confidence: number } {
  const ordered: string[] = [];
  const seen = new Set<string>();

  const pushUnique = (id: string): void => {
    if (id.length > 0 && !isUnsafeKey(id) && !seen.has(id)) {
      seen.add(id);
      ordered.push(id);
    }
  };

  pushUnique(selectedModel);
  for (const fb of rankedFallbacks) pushUnique(fb);
  for (const candidate of candidateIds) pushUnique(candidate);

  if (ordered.length === 0) {
    return { probabilities: {}, confidence: 0 };
  }
  if (ordered.length === 1) {
    const probabilities: Record<string, number> = {};
    probabilities[ordered[0]] = 1;
    return { probabilities, confidence: 1 };
  }

  const weights = ordered.map((_, idx) => Math.pow(RANK_PROBABILITY_DECAY, idx));
  const totalWeight = weights.reduce((acc, w) => acc + w, 0);

  const probabilities: Record<string, number> = {};
  for (let i = 0; i < ordered.length; i += 1) {
    probabilities[ordered[i]] = weights[i] / totalWeight;
  }

  return {
    probabilities,
    confidence: probabilities[ordered[0]],
  };
}

function formatBaseUrl(host: string, port: number): string {
  const formattedHost = host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
  return `http://${formattedHost}:${port}`;
}

export function createHttpBridge(
  engine: S1GateEngine,
  opts: HttpBridgeOptions = {},
): HttpBridgeServer {
  const host = opts.host ?? DEFAULT_HOST;
  const port = opts.port ?? DEFAULT_PORT;
  const defaultProfile = opts.defaultProfile;
  const maxRouteBodyBytes = opts.maxRouteBodyBytes ?? DEFAULT_MAX_ROUTE_BODY_BYTES;
  const maxSystemOneBodyBytes = opts.maxSystemOneBodyBytes ?? DEFAULT_MAX_SYSTEM_ONE_BODY_BYTES;

  async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const method = (req.method ?? "GET").toUpperCase();
    const pathname = extractPathname(req.url);

    if (method === "GET" && pathname === "/health") {
      sendJson(res, 200, {
        status: "ok",
        engine: "s1gate",
        benchmarkSource: "llm-stats.com",
        snapshotModels: Object.keys(engine.snapshot.models).length,
        domains: engine.domains.length,
        stats: engine.getStats(),
      });
      return;
    }

    if (method === "POST" && pathname === "/v1/route") {
      const payload = await readJsonBody(req, maxRouteBodyBytes);
      const incomingHeaders = normalizeIncomingHeaders(req.headers);
      const s1Req = parseRouteRequest(payload, incomingHeaders, defaultProfile);
      const decision = await engine.route(s1Req);
      sendJson(res, 200, decision, decision.headers);
      return;
    }

    if (method === "POST" && pathname === "/v1/systemone") {
      const payload = await readJsonBody(req, maxSystemOneBodyBytes);
      const incomingHeaders = normalizeIncomingHeaders(req.headers);
      const { questionKey, candidateIds, s1Req } = parseSystemOneRequest(
        payload,
        incomingHeaders,
        defaultProfile,
      );
      const decision = await engine.route(s1Req);
      const candidateSet = new Set(candidateIds);
      const selectedChoice = candidateSet.has(decision.selectedModel)
        ? decision.selectedModel
        : candidateIds[0];
      const validFallbacks = decision.rankedFallbacks.filter(fb => candidateSet.has(fb));
      const { probabilities, confidence } = buildRankedProbabilities(
        selectedChoice,
        validFallbacks,
        candidateIds,
      );
      const responseHeaders =
        selectedChoice !== decision.selectedModel
          ? { ...decision.headers, "X-Router-Model": selectedChoice }
          : decision.headers;

      sendJson(
        res,
        200,
        {
          model: "s1gate",
          answers: {
            [questionKey]: {
              type: "choice",
              choice: selectedChoice,
              probabilities,
              confidence,
            },
          },
          usage: { input_tokens: 0, output_tokens: 0 },
          s1gate: {
            action: decision.action,
            domain: decision.signals?.domain ?? null,
            role: decision.role,
            cachePinned: decision.cachePinned,
          },
        },
        responseHeaders,
      );
      return;
    }

    sendJson(res, 404, { error: "Not Found" });
  }

  const server: Server = createServer((req, res) => {
    void handleRequest(req, res).catch((err: unknown) => {
      if (err instanceof HttpError) {
        if (err.status === 413) {
          res.once("finish", () => req.socket?.destroy());
        }
        sendJson(res, err.status, { error: err.message });
        return;
      }
      sendJson(res, 500, { error: "Internal server error" });
    });
  });
  server.headersTimeout = SERVER_HEADERS_TIMEOUT_MS;
  server.requestTimeout = SERVER_REQUEST_TIMEOUT_MS;
  server.keepAliveTimeout = SERVER_KEEP_ALIVE_TIMEOUT_MS;

  function getBoundAddress(): { host: string; port: number; url: string } {
    const addr = server.address();
    const resolvedPort = typeof addr === "object" && addr !== null ? addr.port : port;
    const resolvedHost =
      host || (typeof addr === "object" && addr !== null ? addr.address : DEFAULT_HOST);
    return {
      host: resolvedHost,
      port: resolvedPort,
      url: formatBaseUrl(resolvedHost, resolvedPort),
    };
  }

  return {
    server,
    start(): Promise<{ host: string; port: number; url: string }> {
      if (server.listening) {
        return Promise.resolve(getBoundAddress());
      }
      return new Promise((resolve, reject) => {
        const onError = (err: Error): void => {
          server.off("listening", onListening);
          reject(err);
        };
        const onListening = (): void => {
          server.off("error", onError);
          resolve(getBoundAddress());
        };
        server.once("error", onError);
        server.once("listening", onListening);
        server.listen(port, host);
      });
    },
    close(): Promise<void> {
      if (!server.listening) {
        return Promise.resolve();
      }
      return new Promise((resolve, reject) => {
        server.close((err) => {
          if (err) {
            reject(err);
            return;
          }
          resolve();
        });
        server.closeIdleConnections?.();
      });
    },
  };
}

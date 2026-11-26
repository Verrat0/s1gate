import type {
  BudgetProfile,
  HarnessId,
  InboundProtocol,
  S1GateDecision,
} from "../types.js";
import type { S1GateEngine } from "./engine.js";

const UNSAFE_KEYS = new Set(["__proto__", "constructor", "prototype"]);
const STRIPPED_HEADERS = new Set([
  "authorization",
  "cookie",
  "x-api-key",
  "proxy-authorization",
]);

export interface RoutingFetchOptions {
  availableModels: readonly string[] | (() => readonly string[]);
  protocol?: InboundProtocol;
  harness?: HarnessId;
  profile?: BudgetProfile;
  baseFetch?: typeof fetch;
  applyEffort?: boolean;
  onDecision?: (
    decision: S1GateDecision,
    requestInfo: { url: string; originalModel: string },
  ) => void;
}

type FetchInput = Parameters<typeof fetch>[0];
type FetchInit = Parameters<typeof fetch>[1];

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function resolveRequestUrl(input: FetchInput): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  return input.url;
}

function extractPathname(rawUrl: string): string {
  try {
    return new URL(rawUrl, "http://s1gate.local").pathname;
  } catch {
    return rawUrl.split("?")[0] ?? "/";
  }
}

function detectProtocolFromPath(pathname: string): InboundProtocol {
  const normalized = pathname.toLowerCase().replace(/\/+$/, "");
  if (
    normalized === "/messages" ||
    normalized.endsWith("/messages") ||
    normalized.includes("/messages/")
  ) {
    return "anthropic_messages";
  }
  if (
    normalized === "/responses" ||
    normalized.endsWith("/responses") ||
    normalized.includes("/responses/")
  ) {
    return "openai_responses";
  }
  return "openai_chat";
}

function resolveMethod(input: FetchInput, init?: FetchInit): string {
  if (typeof init?.method === "string" && init.method.trim().length > 0) {
    return init.method.trim().toUpperCase();
  }
  if (input instanceof Request) {
    return input.method.toUpperCase();
  }
  return "GET";
}

function resolveEffectiveHeaders(input: FetchInput, init?: FetchInit): Headers {
  if (init?.headers !== undefined) {
    return new Headers(init.headers);
  }
  if (input instanceof Request) {
    return new Headers(input.headers);
  }
  return new Headers();
}

function sanitizeHeadersForEngine(headers: Headers): Record<string, string> {
  const sanitized: Record<string, string> = {};
  headers.forEach((value, key) => {
    const lower = key.toLowerCase();
    if (UNSAFE_KEYS.has(lower) || STRIPPED_HEADERS.has(lower)) {
      return;
    }
    sanitized[lower] = value;
  });
  return sanitized;
}

function resolveAvailableModels(
  source: RoutingFetchOptions["availableModels"],
): string[] {
  const list = typeof source === "function" ? source() : source;
  if (!Array.isArray(list)) return [];
  return list.filter(
    (model): model is string => typeof model === "string" && model.trim().length > 0,
  );
}

async function readJsonBodyText(
  input: FetchInput,
  init: FetchInit | undefined,
  contentType: string,
): Promise<string | null> {
  const isJsonContentType = contentType.includes("application/json");
  const hasExplicitNonJsonType = contentType.length > 0 && !isJsonContentType;

  if (init?.body !== undefined) {
    const body = init.body;
    if (body === null) return null;
    if (typeof body === "string") {
      return hasExplicitNonJsonType ? null : body;
    }
    if (isJsonContentType) {
      if (body instanceof ArrayBuffer) {
        return new TextDecoder().decode(new Uint8Array(body));
      }
      if (ArrayBuffer.isView(body)) {
        return new TextDecoder().decode(
          new Uint8Array(body.buffer, body.byteOffset, body.byteLength),
        );
      }
    }
    return null;
  }

  if (input instanceof Request) {
    if (input.body === null || !isJsonContentType) {
      return null;
    }
    try {
      return await input.clone().text();
    } catch {
      return null;
    }
  }

  return null;
}

function parseJsonObject(raw: string): Record<string, unknown> | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  try {
    const parsed: unknown = JSON.parse(trimmed);
    return isPlainObject(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function hasCallerEffortControl(body: Record<string, unknown>): boolean {
  return (
    body.reasoning_effort !== undefined ||
    body.reasoning !== undefined ||
    body.thinking !== undefined
  );
}

function buildRewrittenBody(
  parsedBody: Record<string, unknown>,
  decision: S1GateDecision,
  protocol: InboundProtocol,
  applyEffort: boolean,
): Record<string, unknown> {
  const rewritten: Record<string, unknown> = {
    ...parsedBody,
    model: decision.selectedModel,
  };

  if (
    applyEffort &&
    decision.recommendedEffort !== undefined &&
    !hasCallerEffortControl(parsedBody)
  ) {
    if (protocol === "openai_chat") {
      rewritten.reasoning_effort = decision.recommendedEffort;
    } else if (protocol === "openai_responses") {
      rewritten.reasoning = { effort: decision.recommendedEffort };
    }
  }

  return rewritten;
}

function stripStaleContentLength(headersInit: HeadersInit | undefined): HeadersInit | undefined {
  if (headersInit === undefined) return undefined;
  const headers = new Headers(headersInit);
  if (!headers.has("content-length")) {
    return headersInit;
  }
  headers.delete("content-length");
  return headers;
}

export function createRoutingFetch(
  engine: S1GateEngine,
  options: RoutingFetchOptions,
): typeof fetch {
  const invokeBaseFetch: typeof fetch = options.baseFetch
    ? (input, init) => options.baseFetch!(input, init)
    : (input, init) => globalThis.fetch(input, init);
  const harness: HarnessId = options.harness ?? "generic";
  const applyEffort = options.applyEffort === true;

  const routingFetch: typeof fetch = async (input, init) => {
    if (resolveMethod(input, init) !== "POST") {
      return invokeBaseFetch(input, init);
    }

    const availableModels = resolveAvailableModels(options.availableModels);
    if (availableModels.length === 0) {
      return invokeBaseFetch(input, init);
    }

    const effectiveHeaders = resolveEffectiveHeaders(input, init);
    const contentType = (effectiveHeaders.get("content-type") ?? "").toLowerCase();
    const rawBodyText = await readJsonBodyText(input, init, contentType);
    if (rawBodyText === null) {
      return invokeBaseFetch(input, init);
    }

    const parsedBody = parseJsonObject(rawBodyText);
    if (parsedBody === null) {
      return invokeBaseFetch(input, init);
    }

    const rawUrl = resolveRequestUrl(input);
    const path = extractPathname(rawUrl);
    const protocol = options.protocol ?? detectProtocolFromPath(path);
    const originalModel = typeof parsedBody.model === "string" ? parsedBody.model : "";
    const sanitizedHeaders = sanitizeHeadersForEngine(effectiveHeaders);

    const decision = await engine.route({
      protocol,
      path,
      headers: sanitizedHeaders,
      body: parsedBody,
      availableModels,
      harness,
      ...(options.profile !== undefined ? { profile: options.profile } : {}),
    });

    if (decision.action === "passthrough" || !decision.selectedModel) {
      return invokeBaseFetch(input, init);
    }

    const rewrittenBody = buildRewrittenBody(parsedBody, decision, protocol, applyEffort);
    const serializedBody = JSON.stringify(rewrittenBody);
    options.onDecision?.(decision, { url: rawUrl, originalModel });

    if (input instanceof Request) {
      const nextHeaders = new Headers(init?.headers ?? input.headers);
      nextHeaders.delete("content-length");
      const nextRequest = new Request(input, {
        ...init,
        headers: nextHeaders,
        body: serializedBody,
      });
      return invokeBaseFetch(nextRequest);
    }

    const nextInit: RequestInit = {
      ...init,
      body: serializedBody,
      ...(init?.headers !== undefined
        ? { headers: stripStaleContentLength(init.headers) }
        : {}),
    };
    return invokeBaseFetch(input, nextInit);
  };

  return routingFetch;
}

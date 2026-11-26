import { describe, expect, it, vi } from "vitest";
import {
  createRoutingFetch,
  createS1Gate,
  type ClassifierAdapter,
  type S1GateDecision,
  type S1GateEngine,
  type S1GateRequest,
  type SystemOneSignals,
} from "../src/core/index.js";

const TEST_MODELS = [
  "anthropic/claude-opus-5-5",
  "anthropic/claude-sonnet-5-5",
  "anthropic/claude-haiku-5-5",
  "google-antigravity/gemini-4-hydrogen",
  "gpt-6.2-sol",
];

function syntheticToken(prefix: string): string {
  return [prefix, "synthetic", "test", "value"].join("-");
}

function createMockClassifier(
  overrides: Partial<SystemOneSignals> = {},
): ClassifierAdapter {
  return {
    name: "mock-fetch-classifier",
    async classify(): Promise<SystemOneSignals> {
      return {
        domain: "terminal_devops",
        domainProbabilities: { terminal_devops: 0.9, software_engineering: 0.1 },
        domainConfidence: 0.9,
        complexity: 0.5,
        needsFlagshipOrchestrator: 0.2,
        shouldEscalate: 0.1,
        backend: "mock-fetch-classifier",
        latencyMs: 5,
        fromFallback: false,
        ...overrides,
      };
    },
  };
}

function createSpyEngine(
  decisionOverrides: Partial<S1GateDecision> = {},
): { engine: S1GateEngine; routeSpy: ReturnType<typeof vi.fn> } {
  const routeSpy = vi.fn(async (_req: S1GateRequest): Promise<S1GateDecision> => ({
    selectedModel: "anthropic/claude-sonnet-5-5",
    rankedFallbacks: ["google-antigravity/gemini-4-hydrogen"],
    action: "route",
    role: "specialist",
    profile: "balanced",
    recommendedEffort: "medium",
    sessionKey: "sess-1",
    cachePinned: true,
    latencyMs: { total: 2, fastPath: 1, classifier: 0.5, scoring: 0.5 },
    headers: { "X-Router-Engine": "s1gate" },
    ...decisionOverrides,
  }));

  const realEngine = createS1Gate({ classifier: createMockClassifier() });
  const engine: S1GateEngine = {
    route: routeSpy as S1GateEngine["route"],
    getStats: () => realEngine.getStats(),
    clearSessions: () => realEngine.clearSessions(),
    snapshot: realEngine.snapshot,
    domains: realEngine.domains,
  };
  return { engine, routeSpy };
}

describe("createRoutingFetch middleware", () => {
  it("auto-detects anthropic_messages, openai_responses, and openai_chat from URL path", async () => {
    const { engine, routeSpy } = createSpyEngine();
    const baseFetch = vi.fn(async () => new Response("{}", { status: 200 }));
    const routedFetch = createRoutingFetch(engine, {
      availableModels: TEST_MODELS,
      baseFetch,
    });

    await routedFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "s1gate", messages: [{ role: "user", content: "hi" }] }),
    });

    await routedFetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "s1gate", input: "hello" }),
    });

    await routedFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "s1gate", messages: [{ role: "user", content: "hi" }] }),
    });

    expect(routeSpy).toHaveBeenCalledTimes(3);
    expect((routeSpy.mock.calls[0][0] as S1GateRequest).protocol).toBe("anthropic_messages");
    expect((routeSpy.mock.calls[0][0] as S1GateRequest).path).toBe("/v1/messages");
    expect((routeSpy.mock.calls[1][0] as S1GateRequest).protocol).toBe("openai_responses");
    expect((routeSpy.mock.calls[1][0] as S1GateRequest).path).toBe("/v1/responses");
    expect((routeSpy.mock.calls[2][0] as S1GateRequest).protocol).toBe("openai_chat");
    expect((routeSpy.mock.calls[2][0] as S1GateRequest).path).toBe("/v1/chat/completions");
  });

  it("supports explicit protocol, harness, and profile overrides", async () => {
    const { engine, routeSpy } = createSpyEngine();
    const baseFetch = vi.fn(async () => new Response("{}", { status: 200 }));
    const routedFetch = createRoutingFetch(engine, {
      availableModels: () => TEST_MODELS,
      protocol: "anthropic_messages",
      harness: "claude_code",
      profile: "frugal",
      baseFetch,
    });

    await routedFetch("https://proxy.example.com/custom/endpoint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "s1gate", messages: [{ role: "user", content: "hi" }] }),
    });

    const captured = routeSpy.mock.calls[0][0] as S1GateRequest;
    expect(captured.protocol).toBe("anthropic_messages");
    expect(captured.harness).toBe("claude_code");
    expect(captured.profile).toBe("frugal");
    expect(captured.availableModels).toEqual(TEST_MODELS);
  });

  it("rewrites body.model while preserving Authorization, x-api-key, stream: true, and streaming Response", async () => {
    const gate = createS1Gate({
      defaultProfile: "balanced",
      classifier: createMockClassifier(),
    });
    const authHeader = `Bearer ${syntheticToken("bearer")}`;
    const apiKeyHeader = syntheticToken("apikey");

    const streamBody = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("data: {\"delta\":\"ok\"}\n\n"));
        controller.close();
      },
    });
    const upstreamResponse = new Response(streamBody, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });

    const baseFetch = vi.fn(async () => upstreamResponse);
    const routedFetch = createRoutingFetch(gate, {
      availableModels: TEST_MODELS,
      baseFetch,
    });

    const callerPayload = {
      model: "s1gate",
      stream: true,
      temperature: 0.2,
      messages: [{ role: "user", content: "Fix docker compose exit code 137" }],
    };

    const res = await routedFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: authHeader,
        "x-api-key": apiKeyHeader,
        "Content-Length": "12",
      },
      body: JSON.stringify(callerPayload),
    });

    // Returns exact streaming Response instance untouched
    expect(res).toBe(upstreamResponse);
    expect(res.bodyUsed).toBe(false);
    expect(await res.text()).toContain("delta");

    // Outbound request preserved auth headers and stream: true, rewrote model, dropped stale Content-Length
    const [, outboundInit] = baseFetch.mock.calls[0] as unknown as [string, RequestInit];
    const outboundHeaders = new Headers(outboundInit.headers);
    expect(outboundHeaders.get("authorization")).toBe(authHeader);
    expect(outboundHeaders.get("x-api-key")).toBe(apiKeyHeader);
    expect(outboundHeaders.has("content-length")).toBe(false);

    const outboundBody = JSON.parse(outboundInit.body as string) as Record<string, unknown>;
    expect(outboundBody.model).not.toBe("s1gate");
    expect(TEST_MODELS).toContain(outboundBody.model);
    expect(outboundBody.stream).toBe(true);
    expect(outboundBody.temperature).toBe(0.2);

    // Caller payload object was never mutated
    expect(callerPayload.model).toBe("s1gate");
  });

  it("strips sensitive auth headers before passing headers to engine.route while preserving safe headers", async () => {
    const { engine, routeSpy } = createSpyEngine();
    const baseFetch = vi.fn(async () => new Response("{}", { status: 200 }));
    const routedFetch = createRoutingFetch(engine, {
      availableModels: TEST_MODELS,
      baseFetch,
    });

    const authHeader = `Bearer ${syntheticToken("auth")}`;
    const apiKeyHeader = syntheticToken("key");
    const cookieHeader = `sid=${syntheticToken("cookie")}`;
    const proxyAuthHeader = `Basic ${syntheticToken("proxy")}`;

    await routedFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: authHeader,
        "X-Api-Key": apiKeyHeader,
        Cookie: cookieHeader,
        "Proxy-Authorization": proxyAuthHeader,
        "X-Claude-Code-Session-Id": "session-safe-123",
      },
      body: JSON.stringify({ model: "s1gate", messages: [{ role: "user", content: "hello" }] }),
    });

    const engineReq = routeSpy.mock.calls[0][0] as S1GateRequest;
    expect(engineReq.headers.authorization).toBeUndefined();
    expect(engineReq.headers["x-api-key"]).toBeUndefined();
    expect(engineReq.headers.cookie).toBeUndefined();
    expect(engineReq.headers["proxy-authorization"]).toBeUndefined();
    expect(engineReq.headers["x-claude-code-session-id"]).toBe("session-safe-123");

    const [, outboundInit] = baseFetch.mock.calls[0] as unknown as [string, RequestInit];
    const outboundHeaders = new Headers(outboundInit.headers);
    expect(outboundHeaders.get("authorization")).toBe(authHeader);
    expect(outboundHeaders.get("x-api-key")).toBe(apiKeyHeader);
    expect(outboundHeaders.get("cookie")).toBe(cookieHeader);
    expect(outboundHeaders.get("proxy-authorization")).toBe(proxyAuthHeader);
  });

  it("handles Request objects, URL objects, and string JSON bodies without Content-Type", async () => {
    const { engine, routeSpy } = createSpyEngine({ selectedModel: "gpt-6.2-sol" });
    const baseFetch = vi.fn(async () => new Response("{}", { status: 200 }));
    const routedFetch = createRoutingFetch(engine, {
      availableModels: TEST_MODELS,
      baseFetch,
    });

    // 1. URL object
    const urlObj = new URL("https://api.openai.com/v1/responses");
    await routedFetch(urlObj, {
      method: "POST",
      body: JSON.stringify({ model: "s1gate", input: "hello via URL" }),
    });
    const [call1Input, call1Init] = baseFetch.mock.calls[0] as unknown as [URL, RequestInit];
    expect(call1Input).toBe(urlObj);
    expect(JSON.parse(call1Init.body as string).model).toBe("gpt-6.2-sol");

    // 2. Request object (OpenAI / Anthropic SDK / Workers pattern)
    const authVal = `Bearer ${syntheticToken("req")}`;
    const reqObj = new Request("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: authVal,
      },
      body: JSON.stringify({
        model: "s1gate",
        stream: true,
        messages: [{ role: "user", content: "hello via Request" }],
      }),
    });
    await routedFetch(reqObj);
    const [call2Input] = baseFetch.mock.calls[1] as unknown as [Request];
    expect(call2Input).toBeInstanceOf(Request);
    expect(call2Input.headers.get("authorization")).toBe(authVal);
    const reqBody = (await call2Input.json()) as Record<string, unknown>;
    expect(reqBody.model).toBe("gpt-6.2-sol");
    expect(reqBody.stream).toBe(true);

    // 3. Uint8Array JSON body with application/json header
    const encoded = new TextEncoder().encode(
      JSON.stringify({ model: "s1gate", messages: [{ role: "user", content: "bytes" }] }),
    );
    await routedFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: encoded,
    });
    const [, call3Init] = baseFetch.mock.calls[2] as unknown as [string, RequestInit];
    expect(JSON.parse(call3Init.body as string).model).toBe("gpt-6.2-sol");
    expect(routeSpy).toHaveBeenCalledTimes(3);
  });

  it("passes through GET, non-JSON, empty, invalid JSON, empty availableModels, and engine passthrough untouched", async () => {
    const { engine, routeSpy } = createSpyEngine();
    const baseFetch = vi.fn(async () => new Response("ok", { status: 200 }));
    const routedFetch = createRoutingFetch(engine, {
      availableModels: TEST_MODELS,
      baseFetch,
    });

    // GET request
    await routedFetch("https://api.openai.com/v1/models");

    // Non-JSON Content-Type
    await routedFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: '{"model":"s1gate"}',
    });

    // FormData body
    const form = new FormData();
    form.set("file", "content");
    await routedFetch("https://api.openai.com/v1/files", {
      method: "POST",
      body: form,
    });

    // Empty body
    await routedFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "",
    });

    // Invalid JSON body
    const invalidInit: RequestInit = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{not-valid-json",
    };
    await routedFetch("https://api.openai.com/v1/chat/completions", invalidInit);

    // Non-object JSON (array)
    await routedFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "[1, 2, 3]",
    });

    expect(routeSpy).not.toHaveBeenCalled();
    expect(baseFetch).toHaveBeenCalledTimes(6);
    expect((baseFetch.mock.calls[4] as unknown as [string, RequestInit])[1]).toBe(invalidInit);

    // Empty availableModels
    const emptyModelsFetch = createRoutingFetch(engine, {
      availableModels: [],
      baseFetch,
    });
    const payload = JSON.stringify({ model: "s1gate", messages: [{ role: "user", content: "hi" }] });
    await emptyModelsFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
    expect(routeSpy).not.toHaveBeenCalled();
    expect((baseFetch.mock.calls[6] as unknown as [string, RequestInit])[1].body).toBe(payload);

    // Engine returns action: "passthrough" -> body untouched, onDecision not called
    const { engine: passEngine } = createSpyEngine({ action: "passthrough" });
    const onDecision = vi.fn();
    const passFetch = createRoutingFetch(passEngine, {
      availableModels: TEST_MODELS,
      baseFetch,
      onDecision,
    });
    const reqInit: RequestInit = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    };
    await passFetch("https://api.openai.com/v1/chat/completions", reqInit);
    expect((baseFetch.mock.calls[7] as unknown as [string, RequestInit])[1]).toBe(reqInit);
    expect(onDecision).not.toHaveBeenCalled();
  });

  it("applies recommendedEffort only when applyEffort is true, respects existing effort, and never touches Anthropic thinking", async () => {
    const { engine } = createSpyEngine({
      selectedModel: "gpt-6.2-sol",
      recommendedEffort: "high",
    });
    const baseFetch = vi.fn(async () => new Response("{}", { status: 200 }));

    // Default applyEffort=false does NOT add effort
    const defaultFetch = createRoutingFetch(engine, {
      availableModels: TEST_MODELS,
      baseFetch,
    });
    await defaultFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "s1gate", messages: [{ role: "user", content: "hi" }] }),
    });
    const defaultOut = JSON.parse(
      (baseFetch.mock.calls[0] as unknown as [string, RequestInit])[1].body as string,
    ) as Record<string, unknown>;
    expect(defaultOut.reasoning_effort).toBeUndefined();

    const effortFetch = createRoutingFetch(engine, {
      availableModels: TEST_MODELS,
      applyEffort: true,
      baseFetch,
    });

    // 1. openai_chat without caller effort -> sets reasoning_effort
    await effortFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "s1gate", messages: [{ role: "user", content: "hi" }] }),
    });
    const chatOut = JSON.parse(
      (baseFetch.mock.calls[1] as unknown as [string, RequestInit])[1].body as string,
    ) as Record<string, unknown>;
    expect(chatOut.reasoning_effort).toBe("high");

    // 2. openai_chat with existing reasoning_effort -> preserves caller value
    await effortFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "s1gate",
        reasoning_effort: "low",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    const chatCallerOut = JSON.parse(
      (baseFetch.mock.calls[2] as unknown as [string, RequestInit])[1].body as string,
    ) as Record<string, unknown>;
    expect(chatCallerOut.reasoning_effort).toBe("low");

    // 3. openai_responses without caller effort -> sets reasoning.effort
    await effortFetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "s1gate", input: "solve this" }),
    });
    const respOut = JSON.parse(
      (baseFetch.mock.calls[3] as unknown as [string, RequestInit])[1].body as string,
    ) as Record<string, unknown>;
    expect(respOut.reasoning).toEqual({ effort: "high" });

    // 4. openai_responses with existing reasoning -> preserves caller object
    await effortFetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "s1gate",
        reasoning: { effort: "low", summary: "auto" },
        input: "solve this",
      }),
    });
    const respCallerOut = JSON.parse(
      (baseFetch.mock.calls[4] as unknown as [string, RequestInit])[1].body as string,
    ) as Record<string, unknown>;
    expect(respCallerOut.reasoning).toEqual({ effort: "low", summary: "auto" });

    // 5. anthropic_messages -> NEVER adds thinking or reasoning fields
    await effortFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "s1gate", messages: [{ role: "user", content: "hi" }] }),
    });
    const anthOut = JSON.parse(
      (baseFetch.mock.calls[5] as unknown as [string, RequestInit])[1].body as string,
    ) as Record<string, unknown>;
    expect(anthOut.thinking).toBeUndefined();
    expect(anthOut.reasoning).toBeUndefined();
    expect(anthOut.reasoning_effort).toBeUndefined();

    // 6. anthropic_messages with existing thinking block -> left completely untouched
    const existingThinking = { type: "enabled", budget_tokens: 2048 };
    await effortFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "s1gate",
        thinking: existingThinking,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    const anthExistingOut = JSON.parse(
      (baseFetch.mock.calls[6] as unknown as [string, RequestInit])[1].body as string,
    ) as Record<string, unknown>;
    expect(anthExistingOut.thinking).toEqual(existingThinking);
    expect(anthExistingOut.reasoning_effort).toBeUndefined();
  });

  it("invokes onDecision with S1GateDecision and requestInfo { url, originalModel }", async () => {
    const { engine } = createSpyEngine({
      selectedModel: "anthropic/claude-sonnet-5-5",
      action: "route",
      recommendedEffort: "medium",
    });
    const baseFetch = vi.fn(async () => new Response("{}", { status: 200 }));
    const onDecision = vi.fn();

    const routedFetch = createRoutingFetch(engine, {
      availableModels: TEST_MODELS,
      baseFetch,
      onDecision,
    });

    await routedFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "s1gate:balanced",
        messages: [{ role: "user", content: "review this diff" }],
      }),
    });

    expect(onDecision).toHaveBeenCalledTimes(1);
    const [decision, reqInfo] = onDecision.mock.calls[0] as [
      S1GateDecision,
      { url: string; originalModel: string },
    ];
    expect(decision.selectedModel).toBe("anthropic/claude-sonnet-5-5");
    expect(decision.action).toBe("route");
    expect(decision.recommendedEffort).toBe("medium");
    expect(reqInfo).toEqual({
      url: "https://api.anthropic.com/v1/messages",
      originalModel: "s1gate:balanced",
    });
  });
});

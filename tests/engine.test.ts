import { describe, expect, it, vi } from "vitest";
import { buildResponseHeaders, createS1Gate } from "../src/core/engine.js";
import type { ClassifierAdapter, S1GateRequest, SystemOneSignals } from "../src/types.js";

const ACTIVE_MODELS = [
  "anthropic/claude-fable-5-2",
  "anthropic/claude-opus-5-5",
  "anthropic/claude-sonnet-5-5",
  "anthropic/claude-haiku-5-5",
  "google-antigravity/gemini-4-hydrogen",
  "ollama-cloud/hy4",
  "ollama-cloud/mimo-v2.6-flash",
  "gpt-6.2-sol",
];

function mockClassifier(overrides: Partial<SystemOneSignals> = {}): ClassifierAdapter & { calls: number } {
  const clf = {
    name: "mock-systemone",
    calls: 0,
    async classify(): Promise<SystemOneSignals> {
      clf.calls += 1;
      return {
        domain: "terminal_devops",
        domainProbabilities: { terminal_devops: 0.92, software_engineering: 0.08 },
        domainConfidence: 0.9,
        complexity: 0.45,
        needsFlagshipOrchestrator: 0.15,
        shouldEscalate: 0.2,
        backend: "mock-systemone",
        latencyMs: 12,
        fromFallback: false,
        ...overrides,
      };
    },
  };
  return clf;
}

describe("S1GateEngine end-to-end orchestration", () => {
  it("routes turn 1 via System-1, pins sticky session for turn 2+ (0 classifier calls), and emits llm-stats attribution header", async () => {
    const clf = mockClassifier();
    const gate = createS1Gate({ defaultProfile: "balanced", classifier: clf });

    const req1: S1GateRequest = {
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: { "x-claude-code-session-id": "sess-main-1" },
      body: {
        model: "s1gate",
        tools: [{ name: "bash" }, { name: "edit_file" }],
        messages: [{ role: "user", content: "Fix docker compose exit code 137 on production Linux host" }],
      },
      availableModels: ACTIVE_MODELS,
      nowMs: 1_000_000,
    };

    const d1 = await gate.route(req1);
    expect(d1.action).toBe("route");
    expect(d1.cachePinned).toBe(true);
    expect(d1.headers["X-Router-Benchmark-Source"]).toBe("llm-stats.com");
    // Picks verified cost-optimal mid-tier specialist (gemini-4-hydrogen $1.5/$7.5 or claude-sonnet-5-5 $2/$10), never hy4/mimo!
    expect(["google-antigravity/gemini-4-hydrogen", "anthropic/claude-sonnet-5-5"]).toContain(d1.selectedModel);
    expect(clf.calls).toBe(1);

    // Turn 2 on same session -> 0ms sticky reuse, classifier NOT called again!
    const d2 = await gate.route({
      ...req1,
      body: {
        ...req1.body,
        messages: [
          { role: "user", content: "Fix docker compose exit code 137 on production Linux host" },
          { role: "assistant", content: "Checking memory limits." },
          { role: "user", content: "Also check nginx worker_connections" },
        ],
      },
      nowMs: 1_005_000,
    });
    expect(d2.action).toBe("pin_reuse");
    expect(d2.selectedModel).toBe(d1.selectedModel);
    expect(clf.calls).toBe(1);
    expect(gate.getStats().pinReuses).toBe(1);
  });

  it("routes spawned subagents dynamically via X-Claude-Code-Agent-Id without breaking parent pin", async () => {
    const clf = mockClassifier({
      domain: "software_engineering",
      domainProbabilities: { software_engineering: 0.95 },
      complexity: 0.85,
      needsFlagshipOrchestrator: 0.92,
    });
    const gate = createS1Gate({ defaultProfile: "balanced", classifier: clf });

    // Parent main session with Agent tool -> gets orchestrator
    const parent = await gate.route({
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: { "x-claude-code-session-id": "sess-multi" },
      body: {
        model: "s1gate",
        tools: [{ name: "Agent" }, { name: "bash" }],
        messages: [{ role: "user", content: "Refactor auth system across backend, middleware, and frontend" }],
      },
      availableModels: ACTIVE_MODELS,
    });
    expect(parent.role).toBe("orchestrator");
    expect(["anthropic/claude-opus-5-5", "anthropic/claude-fable-5-2"]).toContain(parent.selectedModel);

    // Subagent spawn -> dynamically classified as subagent, gets cheaper specialist!
    const subClfSpy = vi.spyOn(clf, "classify").mockResolvedValueOnce({
      domain: "terminal_devops",
      domainProbabilities: { terminal_devops: 0.96 },
      domainConfidence: 0.94,
      complexity: 0.35,
      needsFlagshipOrchestrator: 0.1,
      backend: "mock-systemone",
      latencyMs: 10,
      fromFallback: false,
    });
    const sub = await gate.route({
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: {
        "x-claude-code-session-id": "sess-multi",
        "x-claude-code-agent-id": "agent-sub-01",
      },
      body: {
        model: "s1gate",
        tools: [{ name: "bash" }],
        messages: [{ role: "user", content: "Inspect docker container logs" }],
      },
      availableModels: ACTIVE_MODELS,
    });
    expect(subClfSpy).toHaveBeenCalled();
    expect(sub.role).toBe("subagent");
    expect(sub.sessionKey).not.toBe(parent.sessionKey);
    expect(["google-antigravity/gemini-4-hydrogen", "anthropic/claude-sonnet-5-5"]).toContain(sub.selectedModel);
  });

  it("bypasses auxiliary calls at 0ms and unlocks pin on /compact boundary", async () => {
    const clf = mockClassifier();
    const gate = createS1Gate({ defaultProfile: "balanced", classifier: clf });

    const aux = await gate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-aux" },
      body: {
        model: "s1gate",
        max_tokens: 64,
        system: "Generate a concise 4-word title for this conversation.",
        messages: [{ role: "user", content: "Docker OOM fix" }],
      },
      availableModels: ACTIVE_MODELS,
    });
    expect(aux.action).toBe("auxiliary");
    expect(aux.selectedModel).toBe("anthropic/claude-haiku-5-5");
    expect(clf.calls).toBe(0);

    // Compact endpoint clears session pin and routes to fast summarizer
    const compact = await gate.route({
      protocol: "openai_responses",
      path: "/v1/responses/compact",
      headers: { "session-id": "sess-c1" },
      body: { model: "s1gate", input: "Summarize history" },
      availableModels: ACTIVE_MODELS,
    });
    expect(compact.action).toBe("compact");
    expect(clf.calls).toBe(0);
  });

  it("escalates immediately at 0ms on hard stuck loop and locks ratchet hysteresis", async () => {
    const clf = mockClassifier({
      domain: "terminal_devops",
      domainProbabilities: { terminal_devops: 0.95 },
      complexity: 0.4,
      needsFlagshipOrchestrator: 0.1,
    });
    const gate = createS1Gate({ defaultProfile: "balanced", classifier: clf });

    const turn1 = await gate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-stuck" },
      body: {
        model: "s1gate",
        tools: [{ name: "bash" }],
        messages: [{ role: "user", content: "Fix failing migration" }],
      },
      availableModels: ACTIVE_MODELS,
    });
    expect(["google-antigravity/gemini-4-hydrogen", "anthropic/claude-sonnet-5-5"]).toContain(turn1.selectedModel);
    expect(clf.calls).toBe(1);

    // 3 identical tool errors -> hard 0ms escalation (no extra classifier call!)
    const stuckMessages = [
      { role: "user", content: "Fix failing migration" },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "bash", input: { cmd: "pytest" } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", is_error: true, content: "OperationalError: relation users does not exist" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t2", name: "bash", input: { cmd: "pytest" } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t2", is_error: true, content: "OperationalError: relation users does not exist" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t3", name: "bash", input: { cmd: "pytest" } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t3", is_error: true, content: "OperationalError: relation users does not exist" }] },
    ];
    const esc = await gate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-stuck" },
      body: { model: "s1gate", tools: [{ name: "bash" }], messages: stuckMessages },
      availableModels: ACTIVE_MODELS,
    });
    expect(esc.action).toBe("escalate");
    expect(esc.selectedModel).not.toBe(turn1.selectedModel);
    expect(clf.calls).toBe(1); // 0ms deterministic hard escalation!

    // Next turn immediately after escalation -> hysteresis lock keeps escalated model!
    const lockedTurn = await gate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-stuck" },
      body: { model: "s1gate", tools: [{ name: "bash" }], messages: stuckMessages },
      availableModels: ACTIVE_MODELS,
    });
    expect(lockedTurn.action).toBe("pin_reuse");
    expect(lockedTurn.selectedModel).toBe(esc.selectedModel);
  });

  it("fails open cleanly if custom classifier throws and supports local_only privacyMode", async () => {
    const brokenClf: ClassifierAdapter = {
      name: "broken",
      async classify() {
        throw new Error("network down");
      },
    };
    const gate = createS1Gate({ classifier: brokenClf });
    const res = await gate.route({
      protocol: "openai_chat",
      headers: {},
      body: { model: "s1gate", messages: [{ role: "user", content: "Refactor hospital SIMRS patient billing API" }] },
      availableModels: ACTIVE_MODELS,
    });
    expect(res.selectedModel).toBeTruthy();
    expect(res.fallbackReason).toBe("classifier_exception");
    expect(res.headers["X-Router-Fallback-Reason"]).toBe("classifier_exception");

    const localGate = createS1Gate({ privacyMode: "local_only", classifier: brokenClf });
    const localRes = await localGate.route({
      protocol: "openai_chat",
      headers: {},
      body: { model: "s1gate", messages: [{ role: "user", content: "Review clinical triage rules for SIMRS EMR" }] },
      availableModels: ACTIVE_MODELS,
    });
    expect(localRes.signals?.fromFallback).toBe(true);
    expect(localRes.signals?.domain).toBe("healthcare");
  });

  it("sanitizes every response header value against CRLF, control chars, non-ASCII, and length > 256", () => {
    const longModel = `anthropic/claude\r\nX-Injected: evil\x00🚀${"a".repeat(300)}`;
    const headers = buildResponseHeaders({
      selectedModel: longModel,
      rankedFallbacks: [],
      action: "route",
      role: "specialist",
      profile: "balanced",
      sessionKey: "abc",
      cachePinned: false,
      fallbackReason: "bad\r\nreason\twith\x7fctrl",
      signals: {
        domain: "software\r\nengineering🚀",
        domainProbabilities: { software_engineering: 1 },
        domainConfidence: 1,
        complexity: 0.5,
        needsFlagshipOrchestrator: 0.2,
        backend: "test",
        latencyMs: 1,
        fromFallback: false,
      },
      latencyMs: { total: 1, fastPath: 1, classifier: 0, scoring: 0 },
    });

    for (const val of Object.values(headers)) {
      expect(val).not.toMatch(/[^\x20-\x7e]/);
      expect(val.length).toBeLessThanOrEqual(256);
    }
    expect(headers["X-Router-Model"]).toBe(`anthropic/claudeX-Injected: evil${"a".repeat(224)}`);
    expect(headers["X-Router-Domain"]).toBe("softwareengineering");
    expect(headers["X-Router-Fallback-Reason"]).toBe("badreasonwithctrl");
  });

  it("returns graceful passthrough when all availableModels are non-chat utility models", async () => {
    const clf = mockClassifier();
    const gate = createS1Gate({ classifier: clf });
    const res = await gate.route({
      protocol: "openai_chat",
      headers: {},
      body: { model: "s1gate", messages: [{ role: "user", content: "Refactor backend auth middleware" }] },
      availableModels: ["google/gemini-2.5-flash-image", "openai/whisper-1", "openai/text-embedding-3-small"],
    });
    expect(res.action).toBe("passthrough");
    expect(res.fallbackReason).toBe("no_eligible_chat_models");
    expect(clf.calls).toBe(0);
  });

  it("invalidates sticky pin on capability mismatch, harness incompatibility, or explicit profile/role override change", async () => {
    const clf = mockClassifier({
      domain: "terminal_devops",
      domainProbabilities: { terminal_devops: 0.95 },
      complexity: 0.4,
      needsFlagshipOrchestrator: 0.1,
    });
    const gate = createS1Gate({
      defaultProfile: "balanced",
      classifier: clf,
      modelOverrides: {
        "anthropic/claude-sonnet-5-5": {
          supportsTools: true,
          contextWindow: 100_000,
          harnessCompat: { claude_code: "native", codex: "unsupported" },
        },
        "google-antigravity/gemini-4-hydrogen": {
          supportsTools: true,
          contextWindow: 100_000,
          harnessCompat: { claude_code: "native", codex: "unsupported" },
        },
      },
    });

    // Turn 1 pins balanced specialist
    const t1 = await gate.route({
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: { "x-claude-code-session-id": "sess-inv" },
      body: {
        model: "s1gate",
        tools: [{ name: "bash" }],
        messages: [{ role: "user", content: "Check nginx error logs" }],
      },
      availableModels: ACTIVE_MODELS,
    });
    expect(t1.action).toBe("route");
    expect(t1.profile).toBe("balanced");
    expect(clf.calls).toBe(1);

    // Explicit profile override change (s1gate:quality) invalidates pin & re-routes
    const t2 = await gate.route({
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: { "x-claude-code-session-id": "sess-inv" },
      body: {
        model: "s1gate:quality",
        tools: [{ name: "bash" }],
        messages: [{ role: "user", content: "Check nginx error logs" }],
      },
      availableModels: ACTIVE_MODELS,
    });
    expect(t2.action).toBe("route");
    expect(t2.profile).toBe("quality");
    expect(clf.calls).toBe(2);

    // Explicit role override change (s1gate:quality:orchestrator) invalidates pin & honors virtual role hint
    const t3 = await gate.route({
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: { "x-claude-code-session-id": "sess-inv" },
      body: {
        model: "s1gate:quality:orchestrator",
        tools: [{ name: "bash" }],
        messages: [{ role: "user", content: "Check nginx error logs" }],
      },
      availableModels: ACTIVE_MODELS,
    });
    expect(t3.action).toBe("route");
    expect(t3.role).toBe("orchestrator");
    expect(clf.calls).toBe(3);

    // Context overflow invalidates pin when pinned model's contextWindow < estimatedInputTokens
    const smallCtxGate = createS1Gate({
      defaultProfile: "balanced",
      classifier: mockClassifier(),
      modelOverrides: {
        "google-antigravity/gemini-4-hydrogen": { contextWindow: 40_000 },
      },
    });
    const smallT1 = await smallCtxGate.route({
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: { "x-claude-code-session-id": "sess-ctx" },
      body: {
        model: "s1gate",
        tools: [{ name: "bash" }],
        messages: [{ role: "user", content: "Check logs" }],
      },
      availableModels: ["google-antigravity/gemini-4-hydrogen", "anthropic/claude-opus-5-5"],
    });
    expect(smallT1.selectedModel).toBe("google-antigravity/gemini-4-hydrogen");

    const hugeT2 = await smallCtxGate.route({
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: { "x-claude-code-session-id": "sess-ctx" },
      body: {
        model: "s1gate",
        tools: [{ name: "bash" }],
        messages: [{ role: "user", content: "x".repeat(170_000) }],
      },
      availableModels: ["google-antigravity/gemini-4-hydrogen", "anthropic/claude-opus-5-5"],
    });
    expect(hugeT2.action).toBe("route");
    expect(hugeT2.selectedModel).toBe("anthropic/claude-opus-5-5");

    // Tools added on turn 2 invalidates pin when pinned model does not support tools
    const noToolsGate = createS1Gate({
      defaultProfile: "balanced",
      classifier: mockClassifier(),
      modelOverrides: {
        "google-antigravity/gemini-4-hydrogen": { supportsTools: false },
      },
    });
    const nt1 = await noToolsGate.route({
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: { "x-claude-code-session-id": "sess-tools" },
      body: {
        model: "s1gate",
        messages: [{ role: "user", content: "Explain Linux systemd OOM killer behavior in detail" }],
      },
      availableModels: ["google-antigravity/gemini-4-hydrogen", "anthropic/claude-sonnet-5-5"],
    });
    expect(nt1.selectedModel).toBe("google-antigravity/gemini-4-hydrogen");

    const nt2 = await noToolsGate.route({
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: { "x-claude-code-session-id": "sess-tools" },
      body: {
        model: "s1gate",
        tools: [{ name: "bash" }],
        messages: [{ role: "user", content: "Now inspect journalctl" }],
      },
      availableModels: ["google-antigravity/gemini-4-hydrogen", "anthropic/claude-sonnet-5-5"],
    });
    expect(nt2.action).toBe("route");
    expect(nt2.selectedModel).toBe("anthropic/claude-sonnet-5-5");

    // Harness unsupported invalidates pin
    const harnessGate = createS1Gate({
      defaultProfile: "balanced",
      classifier: mockClassifier(),
      modelOverrides: {
        "google-antigravity/gemini-4-hydrogen": {
          harnessCompat: { claude_code: "native", codex: "unsupported" },
        },
      },
    });
    const h1 = await harnessGate.route({
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: { "x-claude-code-session-id": "sess-harness" },
      body: { model: "s1gate", tools: [{ name: "bash" }], messages: [{ role: "user", content: "Inspect docker" }] },
      availableModels: ["google-antigravity/gemini-4-hydrogen", "anthropic/claude-sonnet-5-5"],
    });
    expect(h1.selectedModel).toBe("google-antigravity/gemini-4-hydrogen");

    const h2 = await harnessGate.route({
      protocol: "anthropic_messages",
      harness: "codex",
      headers: { "x-claude-code-session-id": "sess-harness" },
      body: { model: "s1gate", tools: [{ name: "bash" }], messages: [{ role: "user", content: "Inspect docker" }] },
      availableModels: ["google-antigravity/gemini-4-hydrogen", "anthropic/claude-sonnet-5-5"],
    });
    expect(h2.action).toBe("route");
    expect(h2.selectedModel).toBe("anthropic/claude-sonnet-5-5");
  });

  it("persists quality profile on hard escalation, never downgrades a lone flagship, and honors thinking_signature_lock on soft escalation", async () => {
    const stuckHardMessages = [
      { role: "user", content: "Fix failing migration" },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "bash", input: { cmd: "pytest" } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", is_error: true, content: "OperationalError: relation users does not exist" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t2", name: "bash", input: { cmd: "pytest" } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t2", is_error: true, content: "OperationalError: relation users does not exist" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t3", name: "bash", input: { cmd: "pytest" } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t3", is_error: true, content: "OperationalError: relation users does not exist" }] },
    ];

    // 1. Hard escalation persists profile: "quality" onto the pin
    const gate = createS1Gate({ defaultProfile: "balanced", classifier: mockClassifier() });
    await gate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-persist-q" },
      body: { model: "s1gate", tools: [{ name: "bash" }], messages: [{ role: "user", content: "Fix failing migration" }] },
      availableModels: ACTIVE_MODELS,
    });
    const esc = await gate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-persist-q" },
      body: { model: "s1gate", tools: [{ name: "bash" }], messages: stuckHardMessages },
      availableModels: ACTIVE_MODELS,
    });
    expect(esc.action).toBe("escalate");
    expect(esc.profile).toBe("quality");

    const nextReuse = await gate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-persist-q" },
      body: { model: "s1gate", tools: [{ name: "bash" }], messages: stuckHardMessages },
      availableModels: ACTIVE_MODELS,
    });
    expect(nextReuse.action).toBe("pin_reuse");
    expect(nextReuse.profile).toBe("quality");

    // 2. Already-flagship session is NEVER downgraded to a weaker model when escalating
    const flagshipGate = createS1Gate({
      defaultProfile: "quality",
      classifier: mockClassifier({
        domain: "software_engineering",
        domainProbabilities: { software_engineering: 1 },
        complexity: 0.95,
        needsFlagshipOrchestrator: 0.99,
      }),
    });
    const loneFlagshipPool = ["anthropic/claude-opus-5-5", "anthropic/claude-haiku-5-5"];
    const f1 = await flagshipGate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-flagship" },
      body: {
        model: "s1gate",
        tools: [{ name: "Agent" }, { name: "bash" }],
        messages: [{ role: "user", content: "Design distributed consensus engine" }],
      },
      availableModels: loneFlagshipPool,
    });
    expect(f1.selectedModel).toBe("anthropic/claude-opus-5-5");

    const f2 = await flagshipGate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-flagship" },
      body: {
        model: "s1gate",
        tools: [{ name: "Agent" }, { name: "bash" }],
        messages: stuckHardMessages,
      },
      availableModels: loneFlagshipPool,
    });
    expect(f2.selectedModel).toBe("anthropic/claude-opus-5-5");

    // 3. Soft escalation respects thinking_signature_lock when assistant carries a thinking signature
    const softClf = mockClassifier({
      domain: "software_engineering",
      domainProbabilities: { software_engineering: 0.95 },
      complexity: 0.5,
      needsFlagshipOrchestrator: 0.2,
      shouldEscalate: 0.95,
    });
    const softGate = createS1Gate({ defaultProfile: "balanced", classifier: softClf });
    const s1 = await softGate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-sig" },
      body: {
        model: "s1gate",
        tools: [{ name: "bash" }],
        messages: [{ role: "user", content: "Fix bug in auth service" }],
      },
      availableModels: ["anthropic/claude-sonnet-5-5", "anthropic/claude-opus-5-5"],
    });
    expect(s1.selectedModel).toBe("anthropic/claude-sonnet-5-5");

    // 2 identical tool failures -> soft stuck loop + thinking signature block -> locked to sonnet!
    const softStuckWithSig = [
      { role: "user", content: "Fix bug in auth service" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "Analyzing...", signature: "EqoBCgIYAhIM..." },
          { type: "tool_use", id: "t1", name: "bash", input: { cmd: "pytest" } },
        ],
      },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", is_error: true, content: "AssertionError: expected 200 got 500" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t2", name: "bash", input: { cmd: "pytest" } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t2", is_error: true, content: "AssertionError: expected 200 got 500" }] },
    ];
    const s2 = await softGate.route({
      protocol: "anthropic_messages",
      headers: { "x-claude-code-session-id": "sess-sig" },
      body: {
        model: "s1gate",
        tools: [{ name: "bash" }],
        messages: softStuckWithSig,
      },
      availableModels: ["anthropic/claude-sonnet-5-5", "anthropic/claude-opus-5-5"],
    });
    expect(s2.action).toBe("pin_reuse");
    expect(s2.selectedModel).toBe("anthropic/claude-sonnet-5-5");
  });
});

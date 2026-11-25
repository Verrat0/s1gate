import { describe, expect, it } from "vitest";
import {
  resolveSessionKey,
  SessionPinStore,
  shouldAllowMidSessionSwitch,
} from "../src/core/session.js";
import type { CandidateModel, S1GateRequest, SessionPin } from "../src/types.js";

function makeReq(overrides: Partial<S1GateRequest> = {}): S1GateRequest {
  return {
    protocol: "anthropic_messages",
    path: "/v1/messages",
    headers: {},
    body: {
      model: "s1gate",
      system: "You are an engineer.",
      messages: [{ role: "user", content: "Fix the cache bug" }],
    },
    availableModels: ["anthropic/claude-opus-4", "anthropic/claude-sonnet-5-5"],
    ...overrides,
  };
}

function makeModel(overrides: Partial<CandidateModel> = {}): CandidateModel {
  return {
    proxyId: "anthropic/claude-opus-4",
    canonicalId: "claude-opus-4",
    name: "Claude Opus 4",
    organization: "Anthropic",
    contextWindow: 200_000,
    inputPricePerMTok: 15,
    outputPricePerMTok: 75,
    cacheReadMultiplier: 0.1,
    cacheWriteMultiplier: 1.25,
    proxyStripsCacheControl: false,
    thinkingOutputRatio: 1,
    quotaWeightMultiplier: 1,
    multimodal: true,
    supportsTools: true,
    requiresThinkingSignature: true,
    harnessCompat: {},
    scores: {},
    ...overrides,
  };
}

function makePin(overrides: Partial<SessionPin> = {}): SessionPin {
  return {
    sessionKey: "a".repeat(32),
    model: "anthropic/claude-sonnet-5-5",
    role: "specialist",
    profile: "balanced",
    domain: "coding",
    complexity: 0.5,
    lastEstimatedTokens: 4000,
    turnCount: 1,
    escalatedAtTurn: null,
    escalationCount: 0,
    pinnedAtMs: 1_000_000,
    lastUsedAtMs: 1_000_000,
    ...overrides,
  };
}

describe("resolveSessionKey", () => {
  it("always returns a deterministic opaque 32-char lowercase hex digest without exposing PII", () => {
    const req = makeReq({
      body: {
        model: "s1gate",
        metadata: { user_id: "alice@example.com_account_12345" },
        messages: [{ role: "user", content: "Hello" }],
      },
    });
    const k1 = resolveSessionKey(req, false);
    const k2 = resolveSessionKey(req, false);
    expect(k1).toBe(k2);
    expect(k1).toMatch(/^[0-9a-f]{32}$/);
    expect(k1).not.toContain("alice");
    expect(k1).not.toContain("12345");
  });

  it("enforces strict header and body priority order", () => {
    const base = makeReq({
      headers: {
        "x-session-id": "x-sess-1",
        "x-opencode-session": "oc-1",
        "session-id": "sid-dash-1",
        session_id: "sid-under-1",
        "thread-id": "thread-1",
        "x-codex-parent-thread-id": "parent-1",
        "X-Claude-Code-Session-Id": "cc-1",
      },
      body: {
        model: "s1gate",
        prompt_cache_key: "pck-1",
        metadata: { user_id: "uid-1" },
        messages: [{ role: "user", content: "Hi" }],
      },
    });

    // Changing lower-priority values should not affect key when x-claude-code-session-id is set
    const withDifferentLower = makeReq({
      headers: { "x-claude-code-session-id": "cc-1", "x-session-id": "different" },
      body: { model: "s1gate", metadata: { user_id: "different" }, messages: [] },
    });
    expect(resolveSessionKey(base, false)).toBe(resolveSessionKey(withDifferentLower, false));

    // Drop x-claude-code-session-id -> Claude Code format metadata.user_id (_session_) wins over codex thread headers
    const p2a = makeReq({
      headers: { "x-codex-parent-thread-id": "p1", "thread-id": "t1" },
      body: { model: "s1gate", metadata: { user_id: "user_abc_account_1_session_xyz" } },
    });
    const p2b = makeReq({
      headers: { "x-codex-parent-thread-id": "p2", "thread-id": "t2" },
      body: { model: "s1gate", metadata: { user_id: "user_abc_account_1_session_xyz" } },
    });
    expect(resolveSessionKey(p2a, false)).toBe(resolveSessionKey(p2b, false));

    // Generic metadata.user_id (without _session_) yields to explicit session headers
    const genVsCodexA = makeReq({
      headers: { "x-codex-parent-thread-id": "p1", "thread-id": "t1" },
      body: { model: "s1gate", metadata: { user_id: "shared-org-user" } },
    });
    const genVsCodexB = makeReq({
      headers: { "x-codex-parent-thread-id": "p1", "thread-id": "t2" },
      body: { model: "s1gate", metadata: { user_id: "shared-org-user" } },
    });
    expect(resolveSessionKey(genVsCodexA, false)).not.toBe(resolveSessionKey(genVsCodexB, false));

    const genVsXSessionA = makeReq({
      headers: { "x-session-id": "xs-1" },
      body: { model: "s1gate", metadata: { user_id: "shared-org-user" } },
    });
    const genVsXSessionB = makeReq({
      headers: { "x-session-id": "xs-2" },
      body: { model: "s1gate", metadata: { user_id: "shared-org-user" } },
    });
    expect(resolveSessionKey(genVsXSessionA, false)).not.toBe(resolveSessionKey(genVsXSessionB, false));

    // Generic metadata.user_id still beats fallback hash when no explicit session headers are set
    const genFallbackA = makeReq({
      body: {
        model: "s1gate",
        metadata: { user_id: "uid-generic-1" },
        messages: [{ role: "user", content: "First prompt A" }],
      },
    });
    const genFallbackB = makeReq({
      body: {
        model: "s1gate",
        metadata: { user_id: "uid-generic-1" },
        messages: [{ role: "user", content: "First prompt B" }],
      },
    });
    expect(resolveSessionKey(genFallbackA, false)).toBe(resolveSessionKey(genFallbackB, false));

    // Drop metadata.user_id -> codex parent + thread wins over bare thread-id
    const p3a = makeReq({
      headers: { "x-codex-parent-thread-id": "p1", "thread-id": "t1" },
    });
    const p3b = makeReq({
      headers: { "x-codex-parent-thread-id": "p2", "thread-id": "t1" },
    });
    expect(resolveSessionKey(p3a, false)).not.toBe(resolveSessionKey(p3b, false));

    // Bare thread-id wins over session_id
    const p4a = makeReq({ headers: { "thread-id": "t1", session_id: "s1" } });
    const p4b = makeReq({ headers: { "thread-id": "t1", session_id: "s2" } });
    expect(resolveSessionKey(p4a, false)).toBe(resolveSessionKey(p4b, false));

    // session_id / session-id wins over prompt_cache_key
    const p5a = makeReq({
      headers: { "session-id": "s1" },
      body: { model: "s1gate", prompt_cache_key: "pck-a" },
    });
    const p5b = makeReq({
      headers: { "session-id": "s1" },
      body: { model: "s1gate", prompt_cache_key: "pck-b" },
    });
    expect(resolveSessionKey(p5a, false)).toBe(resolveSessionKey(p5b, false));

    // prompt_cache_key wins over x-opencode-session
    const p6a = makeReq({
      headers: { "x-opencode-session": "oc-a" },
      body: { model: "s1gate", prompt_cache_key: "pck-1" },
    });
    const p6b = makeReq({
      headers: { "x-opencode-session": "oc-b" },
      body: { model: "s1gate", prompt_cache_key: "pck-1" },
    });
    expect(resolveSessionKey(p6a, false)).toBe(resolveSessionKey(p6b, false));

    // x-opencode-session wins over x-session-id
    const p7a = makeReq({ headers: { "x-opencode-session": "oc-1", "x-session-id": "x-a" } });
    const p7b = makeReq({ headers: { "x-opencode-session": "oc-1", "x-session-id": "x-b" } });
    expect(resolveSessionKey(p7a, false)).toBe(resolveSessionKey(p7b, false));
  });

  it("falls back to stable hash of system prompt + first user message across turns", () => {
    const turn1 = makeReq({
      body: {
        model: "s1gate",
        system: "You are a Rust expert.",
        messages: [{ role: "user", content: "Explain lifetimes" }],
      },
    });
    const turn2 = makeReq({
      body: {
        model: "s1gate",
        system: "You are a Rust expert.",
        messages: [
          { role: "user", content: "Explain lifetimes" },
          { role: "assistant", content: "Lifetimes describe..." },
          { role: "user", content: "Show an example with structs" },
        ],
      },
    });
    const differentSession = makeReq({
      body: {
        model: "s1gate",
        system: "You are a Rust expert.",
        messages: [{ role: "user", content: "Explain traits" }],
      },
    });
    expect(resolveSessionKey(turn1, false)).toBe(resolveSessionKey(turn2, false));
    expect(resolveSessionKey(turn1, false)).not.toBe(resolveSessionKey(differentSession, false));

    // Also works for OpenAI Responses (instructions + string/array input)
    const resp1 = makeReq({
      protocol: "openai_responses",
      body: {
        model: "s1gate",
        instructions: "System rules",
        input: "Initial task",
      },
    });
    const resp2 = makeReq({
      protocol: "openai_responses",
      body: {
        model: "s1gate",
        instructions: "System rules",
        input: [
          { role: "user", content: [{ type: "input_text", text: "Initial task" }] },
          { role: "assistant", content: [{ type: "output_text", text: "Done" }] },
          { role: "user", content: [{ type: "input_text", text: "Follow up" }] },
        ],
      },
    });
    expect(resolveSessionKey(resp1, false)).toBe(resolveSessionKey(resp2, false));
  });

  it("namespaces subagents so parallel subagents never collide with parent or each other", () => {
    const parent = makeReq({ headers: { "x-claude-code-session-id": "sess-1" } });
    const subA = makeReq({
      headers: { "x-claude-code-session-id": "sess-1", "x-claude-code-agent-id": "sub-A" },
    });
    const subB = makeReq({
      headers: { "x-claude-code-session-id": "sess-1", "x-claude-code-agent-id": "sub-B" },
    });

    const parentKey = resolveSessionKey(parent, false);
    const subAKey = resolveSessionKey(subA, true);
    const subBKey = resolveSessionKey(subB, true);

    expect(subAKey).not.toBe(parentKey);
    expect(subBKey).not.toBe(parentKey);
    expect(subAKey).not.toBe(subBKey);

    // Subagent without agent-id / thread-id uses first user message hash
    const subPrompt1 = makeReq({
      headers: { "x-claude-code-session-id": "sess-1" },
      body: { model: "s1gate", messages: [{ role: "user", content: "Research auth module" }] },
    });
    const subPrompt2 = makeReq({
      headers: { "x-claude-code-session-id": "sess-1" },
      body: { model: "s1gate", messages: [{ role: "user", content: "Research billing module" }] },
    });
    expect(resolveSessionKey(subPrompt1, true)).not.toBe(resolveSessionKey(subPrompt2, true));
    expect(resolveSessionKey(subPrompt1, true)).not.toBe(parentKey);
  });
});

describe("SessionPinStore", () => {
  it("stores, retrieves, deletes, clears, and preserves immutability", () => {
    const store = new SessionPinStore();
    const pin = makePin({ sessionKey: "k1" });
    store.set(pin);
    expect(store.size()).toBe(1);

    pin.model = "mutated-outside";
    const loaded = store.get("k1", 1_000_000);
    expect(loaded?.model).toBe("anthropic/claude-sonnet-5-5");

    if (loaded) loaded.model = "mutated-returned";
    expect(store.get("k1", 1_000_000)?.model).toBe("anthropic/claude-sonnet-5-5");

    expect(store.delete("k1")).toBe(true);
    expect(store.size()).toBe(0);
    expect(store.get("k1", 1_000_000)).toBeUndefined();

    store.set(makePin({ sessionKey: "k2" }));
    store.clear();
    expect(store.size()).toBe(0);
  });

  it("evicts expired sessions according to TTL", () => {
    const store = new SessionPinStore({ ttlMs: 10_000 });
    store.set(makePin({ sessionKey: "k1", lastUsedAtMs: 50_000 }));

    expect(store.get("k1", 60_000)).toBeDefined();
    expect(store.get("k1", 60_001)).toBeUndefined();
    expect(store.size()).toBe(0);
  });

  it("performs LRU eviction when maxSessions is exceeded", () => {
    const store = new SessionPinStore({ maxSessions: 2 });
    store.set(makePin({ sessionKey: "k1", lastUsedAtMs: 1000 }));
    store.set(makePin({ sessionKey: "k2", lastUsedAtMs: 1000 }));

    // Touch k1 so k2 becomes least recently used
    expect(store.get("k1", 1500)).toBeDefined();

    store.set(makePin({ sessionKey: "k3", lastUsedAtMs: 2000 }));
    expect(store.size()).toBe(2);
    expect(store.get("k2", 2000)).toBeUndefined();
    expect(store.get("k1", 2000)).toBeDefined();
    expect(store.get("k3", 2000)).toBeDefined();
  });

  it("detects compaction when tokens drop by >50% from >= 8000 tokens", () => {
    const store = new SessionPinStore();
    expect(store.didCompact(makePin({ lastEstimatedTokens: 10_000 }), 4999)).toBe(true);
    expect(store.didCompact(makePin({ lastEstimatedTokens: 10_000 }), 5000)).toBe(false);
    expect(store.didCompact(makePin({ lastEstimatedTokens: 8000 }), 3999)).toBe(true);
    expect(store.didCompact(makePin({ lastEstimatedTokens: 7999 }), 1000)).toBe(false);
  });
});

describe("shouldAllowMidSessionSwitch", () => {
  const opus = makeModel({
    proxyId: "anthropic/claude-opus-4",
    inputPricePerMTok: 15,
    cacheWriteMultiplier: 1.25,
    requiresThinkingSignature: true,
  });
  const sonnet = makeModel({
    proxyId: "anthropic/claude-sonnet-5-5",
    inputPricePerMTok: 3,
    cacheWriteMultiplier: 1.25,
    requiresThinkingSignature: true,
  });
  const cheapModel = makeModel({
    proxyId: "google/gemini-flash",
    inputPricePerMTok: 0.25,
    cacheWriteMultiplier: 1.25,
    requiresThinkingSignature: false,
  });
  const expensiveNoSig = makeModel({
    proxyId: "openai/gpt-5",
    inputPricePerMTok: 10,
    cacheWriteMultiplier: 1.0,
    requiresThinkingSignature: false,
  });

  it("always allows on compaction boundary or hard escalation", () => {
    const compactRes = shouldAllowMidSessionSwitch({
      currentModel: opus,
      nextModel: sonnet,
      currentContextTokens: 80_000,
      hasThinkingSignature: true,
      isCompactionBoundary: true,
      isHardEscalation: false,
    });
    expect(compactRes.allow).toBe(true);

    const escRes = shouldAllowMidSessionSwitch({
      currentModel: sonnet,
      nextModel: opus,
      currentContextTokens: 80_000,
      hasThinkingSignature: true,
      isCompactionBoundary: false,
      isHardEscalation: true,
    });
    expect(escRes.allow).toBe(true);
  });

  it("blocks switching away from a model with requiresThinkingSignature when thinking signatures are present", () => {
    const res = shouldAllowMidSessionSwitch({
      currentModel: opus,
      nextModel: expensiveNoSig,
      currentContextTokens: 5_000,
      hasThinkingSignature: true,
      isCompactionBoundary: false,
      isHardEscalation: false,
    });
    expect(res).toEqual({ allow: false, reason: "thinking_signature_lock" });

    // Allowed when current model does not require thinking signature
    const resNoReq = shouldAllowMidSessionSwitch({
      currentModel: expensiveNoSig,
      nextModel: opus,
      currentContextTokens: 5_000,
      hasThinkingSignature: true,
      isCompactionBoundary: false,
      isHardEscalation: false,
    });
    expect(resNoReq.allow).toBe(true);
  });

  it("blocks large-context downgrades when cache rewrite tax exceeds expected savings", () => {
    // cur=10, next=8, next.cacheWriteMult=1.25, ctx=50_000, turns=4
    // writeTax = 0.05 * 8 * 1.25 = 0.50
    // savings  = 4 * 0.05 * 0.1 * (10 - 8) = 0.04 -> blocked!
    const slightlyCheaper = makeModel({
      proxyId: "openai/gpt-5-mini-ish",
      inputPricePerMTok: 8,
      cacheWriteMultiplier: 1.25,
      requiresThinkingSignature: false,
    });
    const blocked = shouldAllowMidSessionSwitch({
      currentModel: expensiveNoSig,
      nextModel: slightlyCheaper,
      currentContextTokens: 50_000,
      expectedRemainingTurns: 4,
      hasThinkingSignature: false,
      isCompactionBoundary: false,
      isHardEscalation: false,
    });
    expect(blocked).toEqual({ allow: false, reason: "cache_breakeven_guard" });

    // cur=10, next=0.25, next.cacheWriteMult=1.25, ctx=50_000, turns=4
    // writeTax = 0.05 * 0.25 * 1.25 = 0.015625
    // savings  = 4 * 0.05 * 0.1 * (10 - 0.25) = 0.195 -> allowed!
    const allowed = shouldAllowMidSessionSwitch({
      currentModel: expensiveNoSig,
      nextModel: cheapModel,
      currentContextTokens: 50_000,
      expectedRemainingTurns: 4,
      hasThinkingSignature: false,
      isCompactionBoundary: false,
      isHardEscalation: false,
    });
    expect(allowed.allow).toBe(true);
  });

  it("allows downgrades on small context (<12k) and allows upgrades on large context", () => {
    const smallCtx = shouldAllowMidSessionSwitch({
      currentModel: expensiveNoSig,
      nextModel: makeModel({
        proxyId: "openai/gpt-5-mid",
        inputPricePerMTok: 8,
        cacheWriteMultiplier: 1.25,
        requiresThinkingSignature: false,
      }),
      currentContextTokens: 11_999,
      hasThinkingSignature: false,
      isCompactionBoundary: false,
      isHardEscalation: false,
    });
    expect(smallCtx.allow).toBe(true);

    const upgrade = shouldAllowMidSessionSwitch({
      currentModel: cheapModel,
      nextModel: expensiveNoSig,
      currentContextTokens: 80_000,
      hasThinkingSignature: false,
      isCompactionBoundary: false,
      isHardEscalation: false,
    });
    expect(upgrade.allow).toBe(true);
  });

  it("uses actual cacheReadMultiplier of both models for cache breakeven calculation", () => {
    // Current has low cache read (10 * 0.1 = 1.0/MTok), next has lower raw input (5) but high cacheReadMultiplier (0.5 -> 2.5/MTok)
    // Actual cached-read savings = max(0, 1.0 - 2.5) = 0, so rewrite tax > 0 -> must block!
    const currentLowCacheRead = makeModel({
      proxyId: "anthropic/claude-expensive",
      inputPricePerMTok: 10,
      cacheReadMultiplier: 0.1,
      cacheWriteMultiplier: 1.25,
      requiresThinkingSignature: false,
    });
    const nextHighCacheRead = makeModel({
      proxyId: "other/mid-poor-cache",
      inputPricePerMTok: 5,
      cacheReadMultiplier: 0.5,
      cacheWriteMultiplier: 1.0,
      requiresThinkingSignature: false,
    });
    const blocked = shouldAllowMidSessionSwitch({
      currentModel: currentLowCacheRead,
      nextModel: nextHighCacheRead,
      currentContextTokens: 50_000,
      expectedRemainingTurns: 10,
      hasThinkingSignature: false,
      isCompactionBoundary: false,
      isHardEscalation: false,
    });
    expect(blocked).toEqual({ allow: false, reason: "cache_breakeven_guard" });

    // Current has expensive cache read (10 * 0.5 = 5.0/MTok), next has cheap cache read (4 * 0.1 = 0.4/MTok)
    // ctx=50k (0.05 MTok), turns=4 -> writeTax = 0.05 * 4 * 1.0 = 0.20, savings = 4 * 0.05 * (5.0 - 0.4) = 0.92 -> allowed!
    const currentHighCacheRead = makeModel({
      proxyId: "other/expensive-poor-cache",
      inputPricePerMTok: 10,
      cacheReadMultiplier: 0.5,
      cacheWriteMultiplier: 1.0,
      requiresThinkingSignature: false,
    });
    const nextLowCacheRead = makeModel({
      proxyId: "anthropic/good-cache",
      inputPricePerMTok: 4,
      cacheReadMultiplier: 0.1,
      cacheWriteMultiplier: 1.0,
      requiresThinkingSignature: false,
    });
    const allowed = shouldAllowMidSessionSwitch({
      currentModel: currentHighCacheRead,
      nextModel: nextLowCacheRead,
      currentContextTokens: 50_000,
      expectedRemainingTurns: 4,
      hasThinkingSignature: false,
      isCompactionBoundary: false,
      isHardEscalation: false,
    });
    expect(allowed).toEqual({ allow: true, reason: "downgrade_breakeven_passed" });
  });
});

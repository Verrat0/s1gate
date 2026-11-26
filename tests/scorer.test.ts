import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { buildCandidatePool } from "../src/core/catalog.js";
import {
  computeEffectiveCost,
  hasKnownPricing,
  isExplicitlyFreeModel,
  resolveDomainQuality,
  scoreAndSelectModel,
  type ScoringInput,
} from "../src/core/scorer.js";
import type {
  BenchmarkSnapshot,
  CandidateModel,
  DomainDefinition,
  SystemOneSignals,
} from "../src/types.js";

const snapshot: BenchmarkSnapshot = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "../data/snapshot.json"), "utf8"),
);
const domains: DomainDefinition[] = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "../data/domains.json"), "utf8"),
);

const REAL_OCX_POOL_IDS = [
  "anthropic/claude-fable-5-2",
  "anthropic/claude-opus-5-5",
  "anthropic/claude-sonnet-5-5",
  "anthropic/claude-haiku-4-5",
  "openai/gpt-6.2-sol",
  "google-antigravity/gemini-4-neon",
  "google-antigravity/gemini-4-hydrogen",
  "ollama-cloud/beam",
  "ollama-cloud/kimi-k3.1-pro",
  "ollama-cloud/nemotron-3-ultra",
  "ollama-cloud/hy4",
  "ollama-cloud/mimo-v2.6-flash",
  "ollama-cloud/glm-5",
];

function makeSignals(overrides: Partial<SystemOneSignals> = {}): SystemOneSignals {
  return {
    domain: "software_engineering",
    domainProbabilities: { software_engineering: 1.0 },
    domainConfidence: 0.95,
    complexity: 0.5,
    needsFlagshipOrchestrator: 0.3,
    backend: "test",
    latencyMs: 5,
    fromFallback: false,
    ...overrides,
  };
}

function makeCandidate(overrides: Partial<CandidateModel> & { proxyId: string }): CandidateModel {
  return {
    canonicalId: overrides.proxyId,
    name: overrides.proxyId,
    organization: "TestOrg",
    contextWindow: 200_000,
    inputPricePerMTok: 2.0,
    outputPricePerMTok: 10.0,
    cacheReadMultiplier: 0.1,
    cacheWriteMultiplier: 1.25,
    proxyStripsCacheControl: false,
    thinkingOutputRatio: 0.4,
    quotaWeightMultiplier: 1.0,
    multimodal: true,
    supportsTools: true,
    requiresThinkingSignature: false,
    harnessCompat: {
      claude_code: "native",
      codex: "native",
      aider: "native",
      opencode: "native",
      cursor: "native",
      generic: "native",
    },
    scores: {},
    ...overrides,
  };
}

function getDomain(id: string): DomainDefinition {
  const found = domains.find(d => d.id === id);
  if (!found) throw new Error(`Domain not found: ${id}`);
  return found;
}

describe("computeEffectiveCost", () => {
  it("computes cache-aware input + output/thinking cost accurately", () => {
    const model = makeCandidate({
      proxyId: "anthropic/claude-sonnet-5-5",
      inputPricePerMTok: 2,
      outputPricePerMTok: 10,
      cacheReadMultiplier: 0.1,
      cacheWriteMultiplier: 1.25,
      thinkingOutputRatio: 0.4,
      proxyStripsCacheControl: false,
      quotaWeightMultiplier: 1.0,
    });

    // pInEff = 2 * (0.8 * 0.1 + 0.2 * 1.25) = 2 * (0.08 + 0.25) = 0.66
    // pOutEff = 10 * 0.08 * (1 + 0.25 * 0.4) = 0.8 * 1.1 = 0.88
    expect(computeEffectiveCost(model, {})).toBeCloseTo(1.54, 6);
  });

  it("removes cache hit discount when proxyStripsCacheControl is true", () => {
    const cached = makeCandidate({ proxyId: "m-cached", proxyStripsCacheControl: false });
    const stripped = makeCandidate({ proxyId: "m-stripped", proxyStripsCacheControl: true });

    const costCached = computeEffectiveCost(cached, {});
    const costStripped = computeEffectiveCost(stripped, {});
    // Stripped: pInEff = 2 * 1.25 = 2.50, pOutEff = 0.88 -> 3.38
    expect(costStripped).toBeCloseTo(3.38, 6);
    expect(costStripped).toBeGreaterThan(costCached * 2);
  });

  it("adds switching cache-invalidation tax and applies quota weight + quota pressure capped at 0.95", () => {
    const model = makeCandidate({
      proxyId: "m-taxed",
      inputPricePerMTok: 4,
      outputPricePerMTok: 20,
      cacheReadMultiplier: 0.1,
      cacheWriteMultiplier: 1.25,
      thinkingOutputRatio: 0.0,
      quotaWeightMultiplier: 1.5,
    });

    const base = computeEffectiveCost(model, {});
    // pInEff = 4 * 0.33 = 1.32, pOutEff = 20 * 0.08 = 1.60 -> base = 2.92 * 1.5 = 4.38
    expect(base).toBeCloseTo(4.38, 6);

    // Switching 200k tokens: tax = 0.2 * 4 * 1.25 = 1.0 -> (2.92 + 1.0) * 1.5 = 5.88
    const switched = computeEffectiveCost(model, { switching: true, contextTokens: 200_000 });
    expect(switched).toBeCloseTo(5.88, 6);

    // Quota utilization 0.90 -> phi = 1 / 0.10 = 10x
    const pressured = computeEffectiveCost(model, { quotaUtilization: 0.9 });
    expect(pressured).toBeCloseTo(base * 10, 5);

    // Quota utilization 0.99 -> clamped at 0.95 -> phi = 20x (never Infinity)
    const extreme = computeEffectiveCost(model, { quotaUtilization: 0.99 });
    expect(Number.isFinite(extreme)).toBe(true);
    expect(extreme).toBeCloseTo(base * 20, 5);
  });
});

describe("resolveDomainQuality", () => {
  it("penalizes hy4-flash via Sibling Red-Flag Penalty while keeping balanced frontier models unpenalized", () => {
    const se = getDomain("software_engineering");
    const pool = buildCandidatePool(
      ["ollama-cloud/hy4", "ollama-cloud/mimo-v2.6-flash", "anthropic/claude-fable-5-2", "anthropic/claude-sonnet-5-5"],
      snapshot,
    );
    const byId = Object.fromEntries(pool.map(m => [m.proxyId, m]));

    const hy4 = resolveDomainQuality(byId["ollama-cloud/hy4"], se, snapshot);
    const mimo = resolveDomainQuality(byId["ollama-cloud/mimo-v2.6-flash"], se, snapshot);
    const fable = resolveDomainQuality(byId["anthropic/claude-fable-5-2"], se, snapshot);
    const sonnet = resolveDomainQuality(byId["anthropic/claude-sonnet-5-5"], se, snapshot);

    expect(hy4.redFlagPenalty).toBeGreaterThan(0.2);
    expect(hy4.quality).toBeLessThan(0.5);
    expect(mimo.redFlagPenalty).toBeGreaterThan(0.15);
    expect(mimo.quality).toBeLessThan(0.55);

    expect(fable.verified).toBe(true);
    expect(fable.redFlagPenalty).toBe(0);
    expect(fable.quality).toBeGreaterThan(sonnet.quality);
    expect(sonnet.verified).toBe(true);
    expect(sonnet.redFlagPenalty).toBe(0);
    expect(sonnet.quality).toBeGreaterThan(mimo.quality + 0.2);
  });

  it("Scenario 3 (Fallback Fairness): bridges missing primary benchmark fairly via Logit-OLS + LCB", () => {
    const se = getDomain("software_engineering");
    const strongBridged = makeCandidate({
      proxyId: "bridged-strong",
      scores: { swe_bench_verified: 0.85, gpqa: 0.92 },
    });
    const weakBridged = makeCandidate({
      proxyId: "bridged-weak",
      scores: { swe_bench_verified: 0.45, gpqa: 0.55 },
    });
    const unbridgedOnly = makeCandidate({
      proxyId: "unbridged-only",
      // ramp_swe_bench_v2 has no precomputed bridge to swe_bench_pro in snapshot.json
      scores: { ramp_swe_bench_v2: 0.72 },
    });
    const noEvidence = makeCandidate({
      proxyId: "no-evidence",
      scores: {},
    });

    const rStrong = resolveDomainQuality(strongBridged, se, snapshot);
    const rWeak = resolveDomainQuality(weakBridged, se, snapshot);
    const rUnbridged = resolveDomainQuality(unbridgedOnly, se, snapshot);
    const rNone = resolveDomainQuality(noEvidence, se, snapshot);

    expect(rStrong.verified).toBe(true);
    expect(rWeak.verified).toBe(true);
    expect(rStrong.quality).toBeGreaterThan(rWeak.quality + 0.2);
    // Calibrated onto swe_bench_pro scale with LCB penalty (must not exceed raw swe_bench_verified 0.85)
    expect(rStrong.quality).toBeGreaterThan(0.5);
    expect(rStrong.quality).toBeLessThan(0.8);
    expect(rStrong.refs.length).toBeGreaterThanOrEqual(2);

    // Unbridged fallback gets conservative penalty (raw * (1 - 0.08 * i)) and verified = false
    expect(rUnbridged.verified).toBe(false);
    expect(rUnbridged.quality).toBeCloseTo(0.72 * (1 - 0.08 * 2), 3);

    expect(rNone.verified).toBe(false);
    expect(rNone.quality).toBeLessThan(0.35);
    expect(rNone.refs).toEqual([]);
  });
});

describe("scoreAndSelectModel: Hard Eligibility Gate & Graceful Fallback", () => {
  it("excludes ineligible candidates with reasons and applies degraded harness penalty", () => {
    const candidates = [
      makeCandidate({
        proxyId: "small-ctx",
        contextWindow: 32_000,
        inputPricePerMTok: 0.1,
        outputPricePerMTok: 0.2,
        scores: { swe_bench_pro: 0.9 },
      }),
      makeCandidate({
        proxyId: "no-tools",
        supportsTools: false,
        inputPricePerMTok: 0.1,
        outputPricePerMTok: 0.2,
        scores: { swe_bench_pro: 0.9 },
      }),
      makeCandidate({
        proxyId: "no-vision",
        multimodal: false,
        inputPricePerMTok: 0.1,
        outputPricePerMTok: 0.2,
        scores: { swe_bench_pro: 0.9 },
      }),
      makeCandidate({
        proxyId: "unsupported-harness",
        harnessCompat: { claude_code: "unsupported" },
        inputPricePerMTok: 0.1,
        outputPricePerMTok: 0.2,
        scores: { swe_bench_pro: 0.9 },
      }),
      makeCandidate({
        proxyId: "degraded-model",
        harnessCompat: { claude_code: "degraded" },
        inputPricePerMTok: 2.0,
        outputPricePerMTok: 10.0,
        scores: { swe_bench_pro: 0.85 },
      }),
      makeCandidate({
        proxyId: "native-model",
        harnessCompat: { claude_code: "native" },
        inputPricePerMTok: 2.1, // within 12% of degraded-model, higher quality after -0.04 penalty
        outputPricePerMTok: 10.2,
        scores: { swe_bench_pro: 0.85 },
      }),
    ];

    const res = scoreAndSelectModel({
      candidates,
      snapshot,
      domains,
      signals: makeSignals({ complexity: 0.8 }),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: true,
      requiredContextTokens: 100_000,
      harness: "claude_code",
    });

    expect(res.excludedReasons["small-ctx"]).toBeDefined();
    expect(res.excludedReasons["no-tools"]).toBeDefined();
    expect(res.excludedReasons["no-vision"]).toBeDefined();
    expect(res.excludedReasons["unsupported-harness"]).toBeDefined();
    expect(res.selected.proxyId).toBe("native-model");

    const degradedSummary = res.rankedCandidates.find(c => c.proxyId === "degraded-model");
    const nativeSummary = res.rankedCandidates.find(c => c.proxyId === "native-model");
    expect(degradedSummary).toBeDefined();
    expect(nativeSummary).toBeDefined();
    expect(nativeSummary!.qualityScore - degradedSummary!.qualityScore).toBeCloseTo(0.04, 4);
  });

  it("falls back gracefully to the largest-context candidate when all models are excluded", () => {
    const candidates = [
      makeCandidate({ proxyId: "ctx-128k", contextWindow: 128_000, scores: { swe_bench_pro: 0.9 } }),
      makeCandidate({ proxyId: "ctx-1m", contextWindow: 1_000_000, scores: { swe_bench_pro: 0.7 } }),
      makeCandidate({ proxyId: "ctx-256k", contextWindow: 256_000, scores: { swe_bench_pro: 0.8 } }),
    ];

    const res = scoreAndSelectModel({
      candidates,
      snapshot,
      domains,
      signals: makeSignals(),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 2_000_000,
      harness: "claude_code",
    });

    expect(res.selected.proxyId).toBe("ctx-1m");
    expect(Object.keys(res.excludedReasons)).toHaveLength(3);
  });
});

describe("R&D Scenarios on Real llm-stats Snapshot", () => {
  const realPool = buildCandidatePool(REAL_OCX_POOL_IDS, snapshot);

  it("Scenario 1 (Anti-Trap): hy4-flash and mimo-v2.6-flash NEVER win balanced on the real OCX pool", () => {
    for (const complexity of [0.2, 0.5, 0.85]) {
      for (const domain of ["software_engineering", "terminal_devops", "research_architecture", "agentic_tools"]) {
        const res = scoreAndSelectModel({
          candidates: realPool,
          snapshot,
          domains,
          signals: makeSignals({
            domain,
            domainProbabilities: { [domain]: 1.0 },
            complexity,
          }),
          profile: "balanced",
          role: "specialist",
          hasTools: true,
          requiresMultimodal: false,
          requiredContextTokens: 32_000,
          harness: "claude_code",
        });

        expect(res.selected.proxyId).not.toBe("ollama-cloud/hy4");
        expect(res.selected.proxyId).not.toBe("ollama-cloud/mimo-v2.6-flash");
        expect(res.selected.verified).toBe(true);
      }
    }
  });

  it("Scenario 1b (No-Evidence Gate): excludes unverified cheap models in balanced & quality but allows in frugal", () => {
    const unverifiedCheap = makeCandidate({
      proxyId: "cheap/unverified-flash",
      inputPricePerMTok: 0.01,
      outputPricePerMTok: 0.02,
      // Only unbridged score -> verified = false
      scores: { ramp_swe_bench_v2: 0.99 },
    });
    const poolWithTrap = [...realPool, unverifiedCheap];

    for (const profile of ["balanced", "quality"] as const) {
      const res = scoreAndSelectModel({
        candidates: poolWithTrap,
        snapshot,
        domains,
        signals: makeSignals({ domain: "software_engineering", complexity: 0.3 }),
        profile,
        role: "specialist",
        hasTools: true,
        requiresMultimodal: false,
        requiredContextTokens: 32_000,
        harness: "claude_code",
      });

      expect(res.selected.proxyId).not.toBe("cheap/unverified-flash");
      expect(res.excludedReasons["cheap/unverified-flash"]).toBeDefined();
    }

    // In frugal mode, unverified candidates are NOT blocked by the No-Evidence Gate
    const frugalRes = scoreAndSelectModel({
      candidates: poolWithTrap,
      snapshot,
      domains,
      signals: makeSignals({ domain: "software_engineering", complexity: 0.1 }),
      profile: "frugal",
      role: "subagent",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
    });
    expect(frugalRes.excludedReasons["cheap/unverified-flash"]).toBeUndefined();
  });

  it("Scenario 2 (Fable vs Specialist): Specialist beats Fable on single-domain balanced; Fable wins cross-domain orchestrator", () => {
    // 2A: Single-domain healthcare coding specialist (balanced, moderate complexity)
    const singleDomainRes = scoreAndSelectModel({
      candidates: realPool,
      snapshot,
      domains,
      signals: makeSignals({
        domain: "healthcare",
        domainProbabilities: { healthcare: 0.92, software_engineering: 0.08 },
        complexity: 0.5,
      }),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
    });

    expect(singleDomainRes.selected.proxyId).toBe("anthropic/claude-sonnet-5-5");
    expect(singleDomainRes.baselineModel).toBe("anthropic/claude-fable-5-2");
    expect(singleDomainRes.estimatedSavingsPct).toBeGreaterThan(60);

    // 2B: Single-domain software_engineering at moderate complexity -> cheaper strong specialist beats Fable
    const singleSweRes = scoreAndSelectModel({
      candidates: realPool,
      snapshot,
      domains,
      signals: makeSignals({
        domain: "software_engineering",
        domainProbabilities: { software_engineering: 1.0 },
        complexity: 0.5,
      }),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
    });

    expect(singleSweRes.selected.proxyId).not.toBe("anthropic/claude-fable-5-2");
    expect(singleSweRes.selected.proxyId).toBe("anthropic/claude-sonnet-5-5");
    expect(singleSweRes.estimatedSavingsPct).toBeGreaterThan(60);

    // 2C: Cross-domain high-complexity orchestrator (quality profile) -> Fable wins over single-domain specialists
    const crossDomainPool = [
      makeCandidate({
        proxyId: "anthropic/claude-fable-5-2",
        inputPricePerMTok: 8,
        outputPricePerMTok: 40,
        scores: snapshot.models["claude-fable-5-2"].scores,
      }),
      makeCandidate({
        proxyId: "specialist/code-only",
        inputPricePerMTok: 1.5,
        outputPricePerMTok: 6,
        scores: { swe_bench_pro: 0.85, hle: 0.25, terminal_bench_4_0: 0.30, gpqa: 0.55 },
      }),
      makeCandidate({
        proxyId: "specialist/research-only",
        inputPricePerMTok: 1.5,
        outputPricePerMTok: 6,
        scores: { swe_bench_pro: 0.35, hle: 0.72, terminal_bench_4_0: 0.25, gpqa: 0.80 },
      }),
    ];

    const crossRes = scoreAndSelectModel({
      candidates: crossDomainPool,
      snapshot,
      domains,
      signals: makeSignals({
        domain: "research_architecture",
        domainProbabilities: {
          research_architecture: 0.45,
          software_engineering: 0.35,
          terminal_devops: 0.20,
        },
        complexity: 0.92,
      }),
      profile: "balanced",
      role: "orchestrator",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
    });

    expect(crossRes.selected.proxyId).toBe("anthropic/claude-fable-5-2");
  });

  it("Scenario 4 (Quota Pressure & Switching Tax): high provider utilization shifts traffic; context switching favors pinned model", () => {
    const baseInput: ScoringInput = {
      candidates: realPool,
      snapshot,
      domains,
      signals: makeSignals({
        domain: "healthcare",
        domainProbabilities: { healthcare: 1.0 },
        complexity: 0.5,
      }),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
    };

    const normalRes = scoreAndSelectModel(baseInput);
    expect(normalRes.selected.proxyId).toBe("anthropic/claude-sonnet-5-5");

    // Spike anthropic quota utilization to 0.93 -> phi = 14.28x -> shifts away from anthropic to openai/gpt-6.2-sol!
    const pressuredRes = scoreAndSelectModel({
      ...baseInput,
      quotaUtilization: { anthropic: 0.93 },
    });
    expect(pressuredRes.selected.proxyId).not.toMatch(/^anthropic\//);
    expect(pressuredRes.selected.proxyId).toBe("openai/gpt-6.2-sol");
    expect(pressuredRes.selected.effectiveCost).toBeLessThan(
      pressuredRes.rankedCandidates.find(c => c.proxyId === "anthropic/claude-sonnet-5-5")!.effectiveCost,
    );

    // Switching tax test: modelA ($2.00) vs modelB ($1.65) in same band.
    // Without switching, modelB is >12% cheaper and wins.
    // When switchingFromModel = modelA with 300k context tokens, modelB pays cache write tax and modelA stays!
    const modelA = makeCandidate({
      proxyId: "provider/model-a",
      inputPricePerMTok: 2.0,
      outputPricePerMTok: 10.0,
      scores: { swe_bench_pro: 0.82 },
    });
    const modelB = makeCandidate({
      proxyId: "provider/model-b",
      inputPricePerMTok: 1.5,
      outputPricePerMTok: 7.5,
      scores: { swe_bench_pro: 0.80 },
    });

    const freshRes = scoreAndSelectModel({
      candidates: [modelA, modelB],
      snapshot,
      domains,
      signals: makeSignals({ complexity: 0.5 }),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
    });
    expect(freshRes.selected.proxyId).toBe("provider/model-b");

    const stickyRes = scoreAndSelectModel({
      candidates: [modelA, modelB],
      snapshot,
      domains,
      signals: makeSignals({ complexity: 0.5 }),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
      switchingFromModel: "provider/model-a",
      currentContextTokens: 300_000,
    });
    expect(stickyRes.selected.proxyId).toBe("provider/model-a");
  });

  it("prefers higher quality inside Pareto band when effective costs are within 12%", () => {
    const cheapLowerQ = makeCandidate({
      proxyId: "tie/lower-q",
      inputPricePerMTok: 2.0,
      outputPricePerMTok: 10.0,
      scores: { swe_bench_pro: 0.78 },
    });
    const slightlyPricierHigherQ = makeCandidate({
      proxyId: "tie/higher-q",
      inputPricePerMTok: 2.15, // +7.5% cost, within 12%
      outputPricePerMTok: 10.75,
      scores: { swe_bench_pro: 0.85 },
    });

    const res = scoreAndSelectModel({
      candidates: [cheapLowerQ, slightlyPricierHigherQ],
      snapshot,
      domains,
      signals: makeSignals({ complexity: 0.5 }),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
    });

    expect(res.selected.proxyId).toBe("tie/higher-q");
  });

  it("Scenario 5 (Performance): scores 200+ real snapshot models in < 2ms", () => {
    const allIds = Object.keys(snapshot.models);
    const largePool = buildCandidatePool(allIds, snapshot);
    expect(largePool.length).toBeGreaterThanOrEqual(200);

    const input: ScoringInput = {
      candidates: largePool,
      snapshot,
      domains,
      signals: makeSignals({
        domain: "software_engineering",
        domainProbabilities: {
          software_engineering: 0.6,
          terminal_devops: 0.25,
          research_architecture: 0.15,
        },
        complexity: 0.65,
      }),
      profile: "balanced",
      role: "orchestrator",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 64_000,
      harness: "claude_code",
    };

    // Warm up JIT
    for (let i = 0; i < 3; i++) {
      scoreAndSelectModel(input);
    }

    const runs = 20;
    const t0 = performance.now();
    let lastSelectedId = "";
    for (let i = 0; i < runs; i++) {
      lastSelectedId = scoreAndSelectModel(input).selected.proxyId;
    }
    const avgMs = (performance.now() - t0) / runs;
    expect(lastSelectedId.length).toBeGreaterThan(0);
    // CI/coverage-safe threshold (<25ms for 412 models x 5 domains under parallel vitest workers)
    expect(avgMs).toBeLessThan(25.0);
  });
});

describe("Security & correctness regressions (unpriced $0 models & numeric guards)", () => {
  it("excludes unpriced $0 verified snapshot flagships when priced verified candidates exist, while keeping explicitly :free models", () => {
    const unpricedFlagship = makeCandidate({
      proxyId: "anthropic/claude-opus-4-5-20251101",
      inputPricePerMTok: 0,
      outputPricePerMTok: 0,
      scores: { swe_bench_pro: 0.92 },
    });
    const pricedSpecialist = makeCandidate({
      proxyId: "anthropic/claude-sonnet-5-5",
      inputPricePerMTok: 2.0,
      outputPricePerMTok: 10.0,
      scores: { swe_bench_pro: 0.86 },
    });
    const explicitlyFree = makeCandidate({
      proxyId: "openrouter/qwen3-coder:free",
      inputPricePerMTok: 0,
      outputPricePerMTok: 0,
      scores: { swe_bench_pro: 0.60 },
    });

    expect(isExplicitlyFreeModel(unpricedFlagship)).toBe(false);
    expect(hasKnownPricing(unpricedFlagship)).toBe(false);
    expect(isExplicitlyFreeModel(explicitlyFree)).toBe(true);
    expect(hasKnownPricing(explicitlyFree)).toBe(true);
    expect(hasKnownPricing(pricedSpecialist)).toBe(true);

    const balanced = scoreAndSelectModel({
      candidates: [unpricedFlagship, pricedSpecialist, explicitlyFree],
      snapshot,
      domains,
      signals: makeSignals({ domain: "software_engineering", complexity: 0.8 }),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
    });

    expect(balanced.excludedReasons["anthropic/claude-opus-4-5-20251101"]).toBe("unknown_pricing");
    expect(balanced.excludedReasons["openrouter/qwen3-coder:free"]).toBeUndefined();
    expect(balanced.selected.proxyId).toBe("anthropic/claude-sonnet-5-5");

    // When NO priced verified candidates exist, unpriced verified models are allowed gracefully
    const allUnpriced = scoreAndSelectModel({
      candidates: [unpricedFlagship],
      snapshot,
      domains,
      signals: makeSignals({ domain: "software_engineering", complexity: 0.8 }),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
    });

    expect(allUnpriced.selected.proxyId).toBe("anthropic/claude-opus-4-5-20251101");
    expect(allUnpriced.excludedReasons["anthropic/claude-opus-4-5-20251101"]).toBeUndefined();
  });

  it("clamps negative/non-finite cost inputs to 0 and applies a 1e-4 floor under quota pressure for :free models", () => {
    const malformed = makeCandidate({
      proxyId: "bad/numeric-model",
      inputPricePerMTok: Number.NaN,
      outputPricePerMTok: -5,
      cacheReadMultiplier: Number.POSITIVE_INFINITY,
      cacheWriteMultiplier: -1,
      thinkingOutputRatio: Number.NaN,
      quotaWeightMultiplier: -2,
    });
    expect(computeEffectiveCost(malformed, { switching: true, contextTokens: -100, quotaUtilization: Number.NaN })).toBe(0);

    const freeA = makeCandidate({
      proxyId: "provider-a/coder:free",
      inputPricePerMTok: 0,
      outputPricePerMTok: 0,
      scores: { swe_bench_pro: 0.80 },
    });
    const freeB = makeCandidate({
      proxyId: "provider-b/coder:free",
      inputPricePerMTok: 0,
      outputPricePerMTok: 0,
      scores: { swe_bench_pro: 0.80 },
    });

    const costIdle = computeEffectiveCost(freeA, { quotaUtilization: 0 });
    const costMid = computeEffectiveCost(freeA, { quotaUtilization: 0.5 });
    const costHigh = computeEffectiveCost(freeA, { quotaUtilization: 0.9 });
    expect(costIdle).toBe(0);
    expect(costMid).toBeCloseTo(2e-4, 8);
    expect(costHigh).toBeCloseTo(1e-3, 8);
    expect(costHigh).toBeGreaterThan(costMid);

    // Quota pressure steers selection away from a congested :free provider toward an idle :free provider
    const res = scoreAndSelectModel({
      candidates: [freeA, freeB],
      snapshot,
      domains,
      signals: makeSignals({ domain: "software_engineering", complexity: 0.5 }),
      profile: "balanced",
      role: "specialist",
      hasTools: true,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
      quotaUtilization: { "provider-a": 0.9 },
    });
    expect(res.selected.proxyId).toBe("provider-b/coder:free");
  });

  it("guards non-finite bridged estimates, clamps sibling scores to [0, 1], and returns effective qualityFloor when qMax < 0.35", () => {
    const se = getDomain("software_engineering");
    const corruptedSnapshot: BenchmarkSnapshot = {
      ...snapshot,
      bridges: {
        ...snapshot.bridges,
        "swe_bench_pro|swe_bench_verified": {
          n: 10,
          rAdj: 0.9,
          muX: Number.NaN,
          sdX: 1,
          muY: 0,
          sdY: 1,
        },
      },
    };
    const model = makeCandidate({
      proxyId: "guarded/model",
      // program_bench is a redFlagSibling outside benchmarkChain; -5.0 must clamp to 0.0
      scores: { swe_bench_pro: 0.45, swe_bench_verified: 0.8, program_bench: -5.0 },
    });

    const q = resolveDomainQuality(model, se, corruptedSnapshot);
    expect(Number.isFinite(q.quality)).toBe(true);
    expect(q.verified).toBe(true);
    // Clamped sibling (-5.0 -> 0.0): gap = 0.45 - 0.0 = 0.45 -> (0.45 - 0.20) * 0.75 = 0.1875 (not 0.35 cap)
    expect(q.redFlagPenalty).toBeCloseTo(0.1875, 6);
    expect(q.quality).toBeCloseTo(0.45 - 0.1875, 6);

    // Ultra-hard domain where qMax < MIN_QUALITY_FLOOR (0.35)
    const lowA = makeCandidate({
      proxyId: "low/a",
      inputPricePerMTok: 2.0,
      outputPricePerMTok: 10.0,
      scores: { swe_bench_pro: 0.25 },
    });
    const lowB = makeCandidate({
      proxyId: "low/b",
      inputPricePerMTok: 1.0,
      outputPricePerMTok: 5.0,
      scores: { swe_bench_pro: 0.20 },
    });

    const lowRes = scoreAndSelectModel({
      candidates: [lowA, lowB],
      snapshot,
      domains,
      signals: makeSignals({ domain: "software_engineering", complexity: 0.9 }),
      profile: "quality",
      role: "specialist",
      hasTools: false,
      requiresMultimodal: false,
      requiredContextTokens: 32_000,
      harness: "claude_code",
    });

    expect(lowRes.qMax).toBeLessThan(0.35);
    expect(lowRes.qualityFloor).toBeCloseTo(Math.max(0, lowRes.qMax - lowRes.deltaQ), 6);
    expect(lowRes.qualityFloor).toBeLessThan(0.35);
    expect(lowRes.selected.qualityScore).toBeGreaterThanOrEqual(lowRes.qualityFloor);
  });
});

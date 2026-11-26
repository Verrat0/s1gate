import bundledDomains from "../../data/domains.json";
import bundledSnapshot from "../../data/snapshot.json";
import type {
  BenchmarkSnapshot,
  BudgetProfile,
  CandidateModel,
  ClassifierAdapter,
  DomainDefinition,
  ExecutionRole,
  ReasoningEffort,
  RoutingAction,
  S1GateConfig,
  S1GateDecision,
  S1GateRequest,
  SessionPin,
  StuckLevel,
  StuckLoopReport,
  SystemOneSignals,
} from "../types.js";
import { buildCandidatePool } from "./catalog.js";
import { buildRoutingEnvelope } from "./envelope.js";
import { inspectFastPath, pickAuxiliaryModel } from "./fast-path.js";
import { scoreAndSelectModel, type ScoringInput } from "./scorer.js";
import { resolveSessionKey, SessionPinStore, shouldAllowMidSessionSwitch } from "./session.js";
import { inspectStuckLoop, isRatchetLocked } from "./stuck-loop.js";
import {
  BUILTIN_BACKEND_CALIBRATIONS,
  HeuristicFallbackClassifier,
  SystemOneClient,
} from "./systemone-client.js";

export interface S1GateStats {
  totalRequests: number;
  pinReuses: number;
  classifierCalls: number;
  auxiliaryBypasses: number;
  compactions: number;
  escalations: number;
  passthroughs: number;
  fallbacks: number;
  activePins: number;
}

export interface S1GateEngine {
  route(req: S1GateRequest): Promise<S1GateDecision>;
  getStats(): S1GateStats;
  clearSessions(): void;
  readonly snapshot: BenchmarkSnapshot;
  readonly domains: readonly DomainDefinition[];
}

const ESCALATION_QUALITY_TOLERANCE = 0.03;
const MAX_CALLER_CONTEXT_TOKENS = 10_000_000;
const MIN_REQUIRED_CONTEXT_TOKENS = 32_000;
const EFFORT_ORDER: readonly ReasoningEffort[] = ["none", "low", "medium", "high", "max"];

function bumpEffortOneStep(effort: ReasoningEffort): ReasoningEffort {
  const idx = EFFORT_ORDER.indexOf(effort);
  if (idx < 0 || idx >= EFFORT_ORDER.length - 1) return "max";
  return EFFORT_ORDER[idx + 1];
}

function computeRouteEffort(
  profile: BudgetProfile,
  role: ExecutionRole,
  complexity: number,
): ReasoningEffort {
  if (profile === "frugal") {
    const base: ReasoningEffort =
      complexity >= 0.75 ? "medium" : complexity >= 0.40 ? "low" : "none";
    return role === "orchestrator" ? bumpEffortOneStep(base) : base;
  }
  if (profile === "balanced") {
    if (complexity >= 0.70) return "high";
    if (complexity >= 0.35) return "medium";
    return "low";
  }
  // quality
  if (role === "orchestrator" && complexity >= 0.85) return "max";
  if (complexity >= 0.65 || role === "orchestrator") return "high";
  return "medium";
}

export function resolveRecommendedEffort(params: {
  action: RoutingAction;
  profile: BudgetProfile;
  role: ExecutionRole;
  complexity: number;
  stuckLevel?: StuckLevel;
  priorEffort?: ReasoningEffort;
}): ReasoningEffort | undefined {
  const { action, profile, role, stuckLevel, priorEffort } = params;
  if (action === "passthrough") return undefined;
  if (action === "compact") return "low";
  if (action === "auxiliary") return profile === "frugal" ? "none" : "low";

  const complexity = Number.isFinite(params.complexity)
    ? Math.min(1, Math.max(0, params.complexity))
    : 0.5;

  if (action === "pin_reuse") {
    return priorEffort ?? computeRouteEffort(profile, role, complexity);
  }

  if (action === "escalate") {
    if (
      priorEffort === "max" ||
      priorEffort === "high" ||
      (profile === "quality" && complexity >= 0.75) ||
      (stuckLevel === "hard" && complexity >= 0.75)
    ) {
      return "max";
    }
    return "high";
  }

  return computeRouteEffort(profile, role, complexity);
}

export function resolveCallerContextTokens(req: S1GateRequest): number {
  const fieldTokens =
    typeof req.estimatedContextTokens === "number" &&
    Number.isFinite(req.estimatedContextTokens) &&
    req.estimatedContextTokens > 0
      ? req.estimatedContextTokens
      : 0;

  let headerTokens = 0;
  const headers = req.headers;
  if (headers && typeof headers === "object") {
    for (const [key, value] of Object.entries(headers)) {
      if (key.toLowerCase() !== "x-s1gate-context-tokens") continue;
      if (typeof value !== "string") continue;
      const raw = value.trim();
      if (!/^\d+$/.test(raw)) break;
      const parsed = Number(raw);
      if (Number.isFinite(parsed) && parsed > 0) {
        headerTokens = parsed;
      }
      break;
    }
  }

  return Math.min(MAX_CALLER_CONTEXT_TOKENS, Math.max(fieldTokens, headerTokens));
}

function sanitizeHeaderValue(val: string): string {
  return val.replace(/[^\x20-\x7e]/g, "").slice(0, 256);
}

export function buildResponseHeaders(
  decision: Omit<S1GateDecision, "headers">,
  contextWindow?: number,
): Record<string, string> {
  const raw: Record<string, string> = {
    "X-Router-Engine": "s1gate",
    "X-Router-Model": decision.selectedModel,
    "X-Router-Action": decision.action,
    "X-Router-Role": decision.role,
    "X-Router-Profile": decision.profile,
    "X-Router-Cache-Pinned": String(decision.cachePinned),
    "X-Router-Latency-Ms": String(Math.round(decision.latencyMs.total * 100) / 100),
    "X-Router-Benchmark-Source": "llm-stats.com",
  };
  if (typeof contextWindow === "number" && Number.isFinite(contextWindow) && contextWindow > 0) {
    raw["X-Router-Context-Window"] = String(Math.floor(contextWindow));
  }
  if (decision.recommendedEffort) {
    raw["X-Router-Effort"] = decision.recommendedEffort;
  }
  if (decision.signals?.domain) {
    raw["X-Router-Domain"] = decision.signals.domain;
    raw["X-Router-Complexity"] = decision.signals.complexity.toFixed(2);
  }
  if (decision.scoring) {
    raw["X-Router-Estimated-Savings-Pct"] = decision.scoring.estimatedSavingsPct.toFixed(1);
  }
  if (decision.fallbackReason) {
    raw["X-Router-Fallback-Reason"] = decision.fallbackReason;
  }
  const headers: Record<string, string> = {};
  for (const [k, v] of Object.entries(raw)) {
    headers[k] = sanitizeHeaderValue(v);
  }
  return headers;
}

function finalizeDecision(
  base: Omit<S1GateDecision, "headers">,
  contextWindow?: number,
): S1GateDecision {
  return {
    ...base,
    headers: buildResponseHeaders(base, contextWindow),
  };
}

function resolveDomains(custom?: DomainDefinition[]): DomainDefinition[] {
  const defaults = bundledDomains as DomainDefinition[];
  if (!custom || custom.length === 0) return defaults;
  const map = new Map<string, DomainDefinition>(defaults.map(d => [d.id, d]));
  for (const d of custom) map.set(d.id, d);
  return [...map.values()];
}

function resolveNoulRole(
  noul: number,
  profile: BudgetProfile,
  threshold: number,
): ExecutionRole {
  if (noul >= threshold) return "orchestrator";
  if (noul <= 1 - threshold) return "specialist";
  // Uncertainty band (0.35..0.65): fail upward on quality, stay economical + ratchet on balanced/frugal
  return profile === "quality" ? "orchestrator" : "specialist";
}

function shouldSoftEscalate(
  shouldEscalateProb: number | undefined,
  loopReport: StuckLoopReport,
  threshold: number,
): boolean {
  if (typeof shouldEscalateProb !== "number" || !Number.isFinite(shouldEscalateProb)) {
    return loopReport.reasons.length >= 2;
  }
  if (shouldEscalateProb >= threshold) return true;
  if (shouldEscalateProb >= 0.5 && loopReport.reasons.length >= 2) return true;
  return false;
}

function selectEscalationModel(
  input: ScoringInput,
  currentPinModel: string,
): ReturnType<typeof scoreAndSelectModel> {
  const fullScoring = scoreAndSelectModel(input);
  const bestOverall = fullScoring.selected;
  if (bestOverall.proxyId !== currentPinModel) {
    return fullScoring;
  }

  const currentPinQuality = bestOverall.qualityScore;
  const minPeerQuality = currentPinQuality - ESCALATION_QUALITY_TOLERANCE;
  const hasComparableAlternative = fullScoring.rankedCandidates.some(
    c => c.proxyId !== currentPinModel && c.qualityScore >= minPeerQuality,
  );
  if (!hasComparableAlternative) {
    return fullScoring;
  }

  const alternativePool = input.candidates.filter(c => c.proxyId !== currentPinModel);
  if (alternativePool.length === 0) {
    return fullScoring;
  }

  const altScoring = scoreAndSelectModel({
    ...input,
    candidates: alternativePool,
  });
  return altScoring.selected.qualityScore >= minPeerQuality ? altScoring : fullScoring;
}

export function createS1Gate(config: S1GateConfig = {}): S1GateEngine {
  const snapshot: BenchmarkSnapshot = config.snapshot ?? (bundledSnapshot as unknown as BenchmarkSnapshot);
  const domains = resolveDomains(config.domains);
  const defaultProfile: BudgetProfile = config.defaultProfile ?? "balanced";
  const privacyMode = config.privacyMode ?? "standard";
  const classifierTimeoutMs = config.classifierTimeoutMs ?? 1500;
  const enableRatchet = config.enableEscalationRatchet ?? true;

  const sessionStore = new SessionPinStore({
    ttlMs: config.sessionTtlMs,
    maxSessions: config.maxSessions,
  });

  const fallbackClassifier = new HeuristicFallbackClassifier();
  const classifier: ClassifierAdapter =
    privacyMode === "local_only"
      ? fallbackClassifier
      : config.classifier
        ? config.classifier
        : config.systemOne
          ? new SystemOneClient(config.systemOne)
          : fallbackClassifier;

  const sysModel = config.systemOne?.model ?? "";
  const lowerSysModel = sysModel.toLowerCase();
  const builtinCalibration =
    BUILTIN_BACKEND_CALIBRATIONS[sysModel] ??
    Object.entries(BUILTIN_BACKEND_CALIBRATIONS).find(
      ([key]) =>
        lowerSysModel === key.toLowerCase() || lowerSysModel.endsWith(`/${key.toLowerCase()}`),
    )?.[1];
  const noulOrchThreshold =
    config.systemOne?.calibration?.noulOrchestratorThreshold ??
    builtinCalibration?.noulOrchestratorThreshold ??
    0.65;
  const noulEscThreshold =
    config.systemOne?.calibration?.noulEscalateThreshold ??
    builtinCalibration?.noulEscalateThreshold ??
    0.70;

  const stats: Omit<S1GateStats, "activePins"> = {
    totalRequests: 0,
    pinReuses: 0,
    classifierCalls: 0,
    auxiliaryBypasses: 0,
    compactions: 0,
    escalations: 0,
    passthroughs: 0,
    fallbacks: 0,
  };

  function resolveContextWindow(
    modelId: string,
    candidates?: readonly CandidateModel[],
  ): number | undefined {
    if (!modelId) return undefined;
    if (candidates) {
      const matched = candidates.find(c => c.proxyId === modelId);
      return matched && matched.contextWindow > 0 ? matched.contextWindow : undefined;
    }
    const single = buildCandidatePool([modelId], snapshot, config.modelOverrides)[0];
    return single && single.contextWindow > 0 ? single.contextWindow : undefined;
  }

  async function route(req: S1GateRequest): Promise<S1GateDecision> {
    const tStart = performance.now();
    stats.totalRequests += 1;
    const nowMs = req.nowMs ?? Date.now();
    const harness = req.harness ?? "generic";
    const availableModels = req.availableModels.filter(m => typeof m === "string" && m.trim().length > 0);

    const rawBodyModel = typeof req.body?.model === "string" ? req.body.model : "";
    const safeDefaultModel = availableModels[0] ?? rawBodyModel;

    // ---------- STAGE 1a: Local Fast-Path (0ms) ----------
    const tFastStart = performance.now();
    const verdict = inspectFastPath(req, {
      triggerModels: config.triggerModels,
      defaultProfile,
    });
    const effectiveContextTokens = Math.max(
      verdict.estimatedInputTokens,
      resolveCallerContextTokens(req),
    );
    const requiredContextTokens = Math.max(
      MIN_REQUIRED_CONTEXT_TOKENS,
      Math.ceil((effectiveContextTokens * 10) / 9),
    );
    const profile: BudgetProfile = req.profile ?? verdict.profileOverride ?? defaultProfile;
    const isSubagent = verdict.isSubagentHeader || verdict.role === "subagent";
    const sessionKey = resolveSessionKey(req, isSubagent);

    // 1. Passthrough (opt-out or count_tokens)
    if (verdict.kind === "passthrough") {
      stats.passthroughs += 1;
      const fastMs = performance.now() - tFastStart;
      const selectedModel = rawBodyModel || safeDefaultModel;
      const ctxWin = availableModels.includes(selectedModel)
        ? resolveContextWindow(selectedModel)
        : undefined;
      return finalizeDecision(
        {
          selectedModel,
          rankedFallbacks: availableModels.filter(m => m !== rawBodyModel),
          action: "passthrough",
          role: verdict.role,
          profile,
          sessionKey: null,
          cachePinned: false,
          latencyMs: { total: performance.now() - tStart, fastPath: fastMs, classifier: 0, scoring: 0 },
        },
        ctxWin,
      );
    }

    // 2. Compaction boundary -> unlock pin & route compaction request to fast/long-context summarizer
    if (verdict.kind === "compact") {
      stats.compactions += 1;
      sessionStore.delete(sessionKey);
      const compactModel = pickAuxiliaryModel(availableModels, config.auxiliaryModelHints);
      const fastMs = performance.now() - tFastStart;
      const selectedModel = compactModel || safeDefaultModel;
      const ctxWin = availableModels.includes(selectedModel)
        ? resolveContextWindow(selectedModel)
        : undefined;
      const recommendedEffort = resolveRecommendedEffort({
        action: "compact",
        profile,
        role: "auxiliary",
        complexity: 0,
      });
      return finalizeDecision(
        {
          selectedModel,
          rankedFallbacks: availableModels.filter(m => m !== compactModel),
          action: "compact",
          role: "auxiliary",
          profile,
          ...(recommendedEffort ? { recommendedEffort } : {}),
          sessionKey,
          cachePinned: false,
          latencyMs: { total: performance.now() - tStart, fastPath: fastMs, classifier: 0, scoring: 0 },
        },
        ctxWin,
      );
    }

    // 3. Auxiliary call -> 0ms bypass without calling System-1 or disturbing parent session pin
    if (verdict.kind === "auxiliary") {
      stats.auxiliaryBypasses += 1;
      const auxModel = pickAuxiliaryModel(availableModels, config.auxiliaryModelHints);
      const fastMs = performance.now() - tFastStart;
      const selectedModel = auxModel || safeDefaultModel;
      const ctxWin = availableModels.includes(selectedModel)
        ? resolveContextWindow(selectedModel)
        : undefined;
      const recommendedEffort = resolveRecommendedEffort({
        action: "auxiliary",
        profile,
        role: "auxiliary",
        complexity: 0,
      });
      return finalizeDecision(
        {
          selectedModel,
          rankedFallbacks: availableModels.filter(m => m !== auxModel),
          action: "auxiliary",
          role: "auxiliary",
          profile,
          ...(recommendedEffort ? { recommendedEffort } : {}),
          sessionKey: null,
          cachePinned: false,
          latencyMs: { total: performance.now() - tStart, fastPath: fastMs, classifier: 0, scoring: 0 },
        },
        ctxWin,
      );
    }

    // Empty model pool failsafe
    if (availableModels.length === 0) {
      stats.fallbacks += 1;
      const fastMs = performance.now() - tFastStart;
      return finalizeDecision({
        selectedModel: rawBodyModel,
        rankedFallbacks: [],
        action: "passthrough",
        role: verdict.role,
        profile,
        sessionKey,
        cachePinned: false,
        fallbackReason: "empty_available_models",
        latencyMs: { total: performance.now() - tStart, fastPath: fastMs, classifier: 0, scoring: 0 },
      });
    }

    const candidates = buildCandidatePool(availableModels, snapshot, config.modelOverrides);
    if (candidates.length === 0) {
      stats.fallbacks += 1;
      const fastMs = performance.now() - tFastStart;
      return finalizeDecision({
        selectedModel: rawBodyModel || safeDefaultModel,
        rankedFallbacks: [],
        action: "passthrough",
        role: verdict.role,
        profile,
        sessionKey,
        cachePinned: false,
        fallbackReason: "no_eligible_chat_models",
        latencyMs: { total: performance.now() - tStart, fastPath: fastMs, classifier: 0, scoring: 0 },
      });
    }

    const explicitProfileOverride = req.profile ?? verdict.profileOverride;
    const effectiveRoleOverride = req.roleOverride ?? verdict.roleOverride;

    const rawExistingPin = sessionStore.get(sessionKey, nowMs);
    const pinCandidate = rawExistingPin
      ? candidates.find(c => c.proxyId === rawExistingPin.model)
      : undefined;
    const pinInvalidated =
      rawExistingPin !== undefined &&
      (!pinCandidate ||
        (verdict.requiresMultimodal && !pinCandidate.multimodal) ||
        (verdict.hasTools && !pinCandidate.supportsTools) ||
        pinCandidate.contextWindow * 9 < effectiveContextTokens * 10 ||
        pinCandidate.harnessCompat[harness] === "unsupported" ||
        (explicitProfileOverride !== undefined && explicitProfileOverride !== rawExistingPin.profile) ||
        (effectiveRoleOverride !== undefined && effectiveRoleOverride !== rawExistingPin.role));

    if (rawExistingPin && pinInvalidated) {
      sessionStore.delete(sessionKey);
    }

    const existingPin = pinInvalidated ? undefined : rawExistingPin;
    const pinModelStillAvailable = existingPin !== undefined && pinCandidate !== undefined;

    let isCompactionBoundary = false;
    let stuckLoop: StuckLoopReport | undefined;

    // 4. Sticky Session Pin Check + Stuck-Loop Ratchet
    if (existingPin && pinModelStillAvailable) {
      if (sessionStore.didCompact(existingPin, effectiveContextTokens)) {
        isCompactionBoundary = true;
        stats.compactions += 1;
        sessionStore.delete(sessionKey);
      } else {
        stuckLoop = enableRatchet ? inspectStuckLoop(req.body, req.protocol) : undefined;
        const locked = isRatchetLocked(existingPin);

        // Reuse sticky pin immediately (0ms!) when healthy or locked by ratchet hysteresis
        if (!stuckLoop || stuckLoop.level === "none" || locked) {
          stats.pinReuses += 1;
          const recommendedEffort = resolveRecommendedEffort({
            action: "pin_reuse",
            profile: existingPin.profile,
            role: existingPin.role,
            complexity: existingPin.complexity,
            priorEffort: existingPin.effort,
          });
          const updatedPin: SessionPin = {
            ...existingPin,
            ...(recommendedEffort ? { effort: recommendedEffort } : {}),
            turnCount: existingPin.turnCount + 1,
            lastEstimatedTokens: effectiveContextTokens,
            lastUsedAtMs: nowMs,
          };
          sessionStore.set(updatedPin);
          const fastMs = performance.now() - tFastStart;
          return finalizeDecision(
            {
              selectedModel: existingPin.model,
              rankedFallbacks: availableModels.filter(m => m !== existingPin.model),
              action: "pin_reuse",
              role: existingPin.role,
              profile: existingPin.profile,
              ...(recommendedEffort ? { recommendedEffort } : {}),
              sessionKey,
              cachePinned: true,
              ...(stuckLoop ? { stuckLoop } : {}),
              latencyMs: { total: performance.now() - tStart, fastPath: fastMs, classifier: 0, scoring: 0 },
            },
            resolveContextWindow(existingPin.model, candidates),
          );
        }

        // Hard 0ms deterministic escalation (skip System-1!)
        if (stuckLoop.level === "hard") {
          const fastMs = performance.now() - tFastStart;
          const tScoreStart = performance.now();
          const escSignals: SystemOneSignals = {
            domain: existingPin.domain,
            domainProbabilities: { [existingPin.domain]: 1 },
            domainConfidence: 1,
            complexity: Math.max(0.85, existingPin.complexity),
            needsFlagshipOrchestrator: 1,
            shouldEscalate: 1,
            backend: "deterministic-ratchet",
            latencyMs: 0,
            fromFallback: false,
          };
          const scoring = selectEscalationModel(
            {
              candidates,
              snapshot,
              domains,
              signals: escSignals,
              profile: "quality",
              role: "orchestrator",
              hasTools: verdict.hasTools,
              requiresMultimodal: verdict.requiresMultimodal,
              requiredContextTokens,
              harness,
              quotaUtilization: req.quotaUtilization,
            },
            existingPin.model,
          );
          const scoreMs = performance.now() - tScoreStart;
          const nextModel = scoring.selected.proxyId;
          const recommendedEffort = resolveRecommendedEffort({
            action: "escalate",
            profile: "quality",
            role: "orchestrator",
            complexity: escSignals.complexity,
            stuckLevel: stuckLoop.level,
            priorEffort: existingPin.effort,
          });
          stats.escalations += 1;

          sessionStore.set({
            ...existingPin,
            model: nextModel,
            role: "orchestrator",
            profile: "quality",
            ...(recommendedEffort ? { effort: recommendedEffort } : {}),
            complexity: escSignals.complexity,
            turnCount: existingPin.turnCount + 1,
            escalatedAtTurn: existingPin.turnCount + 1,
            escalationCount: existingPin.escalationCount + 1,
            lastEstimatedTokens: effectiveContextTokens,
            lastUsedAtMs: nowMs,
          });

          return finalizeDecision(
            {
              selectedModel: nextModel,
              rankedFallbacks: scoring.rankedCandidates.filter(c => c.proxyId !== nextModel).map(c => c.proxyId),
              action: "escalate",
              role: "orchestrator",
              profile: "quality",
              ...(recommendedEffort ? { recommendedEffort } : {}),
              sessionKey,
              cachePinned: true,
              stuckLoop,
              scoring: {
                qualityScore: scoring.selected.qualityScore,
                qMax: scoring.qMax,
                qualityFloor: scoring.qualityFloor,
                effectiveCost: scoring.selected.effectiveCost,
                benchmarkRefs: scoring.selected.benchmarkRefs,
                baselineModel: scoring.baselineModel,
                estimatedSavingsPct: scoring.estimatedSavingsPct,
              },
              latencyMs: { total: performance.now() - tStart, fastPath: fastMs, classifier: 0, scoring: scoreMs },
            },
            resolveContextWindow(nextModel, candidates),
          );
        }
      }
    }

    const fastMs = performance.now() - tFastStart;

    // ---------- STAGE 1b: System-1 Cognitive Gate ----------
    const isSoftStuckCheck = Boolean(existingPin && pinModelStillAvailable && !isCompactionBoundary && stuckLoop?.level === "soft");
    const envelope = buildRoutingEnvelope(req, {
      role: verdict.role,
      stuckLoop,
    });

    const tClfStart = performance.now();
    stats.classifierCalls += 1;
    let signals: SystemOneSignals;
    try {
      signals = await classifier.classify({
        envelope,
        domains,
        askOrchestrator: !isSubagent && verdict.canSpawnSubagent && !effectiveRoleOverride && !isSoftStuckCheck,
        askEscalate: isSoftStuckCheck,
        timeoutMs: classifierTimeoutMs,
      });
    } catch {
      signals = await fallbackClassifier.classify({
        envelope,
        domains,
        askOrchestrator: !isSubagent && verdict.canSpawnSubagent && !effectiveRoleOverride && !isSoftStuckCheck,
        askEscalate: isSoftStuckCheck,
        timeoutMs: classifierTimeoutMs,
      });
      signals = { ...signals, fromFallback: true, fallbackReason: "classifier_exception" };
    }
    if (signals.fromFallback) stats.fallbacks += 1;
    const clfMs = performance.now() - tClfStart;

    // Soft stuck check decided NOT to escalate -> keep sticky pin!
    if (isSoftStuckCheck && existingPin && stuckLoop && !shouldSoftEscalate(signals.shouldEscalate, stuckLoop, noulEscThreshold)) {
      stats.pinReuses += 1;
      const recommendedEffort = resolveRecommendedEffort({
        action: "pin_reuse",
        profile: existingPin.profile,
        role: existingPin.role,
        complexity: existingPin.complexity,
        priorEffort: existingPin.effort,
      });
      sessionStore.set({
        ...existingPin,
        ...(recommendedEffort ? { effort: recommendedEffort } : {}),
        turnCount: existingPin.turnCount + 1,
        lastEstimatedTokens: effectiveContextTokens,
        lastUsedAtMs: nowMs,
      });
      return finalizeDecision(
        {
          selectedModel: existingPin.model,
          rankedFallbacks: availableModels.filter(m => m !== existingPin.model),
          action: "pin_reuse",
          role: existingPin.role,
          profile: existingPin.profile,
          ...(recommendedEffort ? { recommendedEffort } : {}),
          sessionKey,
          cachePinned: true,
          signals,
          stuckLoop,
          ...(signals.fallbackReason ? { fallbackReason: signals.fallbackReason } : {}),
          latencyMs: { total: performance.now() - tStart, fastPath: fastMs, classifier: clfMs, scoring: 0 },
        },
        resolveContextWindow(existingPin.model, candidates),
      );
    }

    // ---------- STAGE 2: Deterministic Scoring & Selection (<0.1ms) ----------
    const tScoreStart = performance.now();
    const isEscalating = isSoftStuckCheck;
    const resolvedRole: ExecutionRole = isEscalating
      ? "orchestrator"
      : effectiveRoleOverride
        ? effectiveRoleOverride
        : isSubagent
          ? "subagent"
          : verdict.canSpawnSubagent
            ? resolveNoulRole(signals.needsFlagshipOrchestrator, profile, noulOrchThreshold)
            : "specialist";

    const effectiveProfile: BudgetProfile = isEscalating ? "quality" : profile;
    const baseScoringInput: ScoringInput = {
      candidates,
      snapshot,
      domains,
      signals,
      profile: effectiveProfile,
      role: resolvedRole,
      hasTools: verdict.hasTools,
      requiresMultimodal: verdict.requiresMultimodal,
      requiredContextTokens,
      harness,
      quotaUtilization: req.quotaUtilization,
      ...(existingPin && pinModelStillAvailable && !isCompactionBoundary && !isEscalating
        ? { switchingFromModel: existingPin.model, currentContextTokens: effectiveContextTokens }
        : {}),
    };

    const scoring =
      isEscalating && existingPin
        ? selectEscalationModel(baseScoringInput, existingPin.model)
        : scoreAndSelectModel(baseScoringInput);

    let selectedModel = scoring.selected.proxyId;
    let action: S1GateDecision["action"] = isEscalating ? "escalate" : "route";
    let finalRole = resolvedRole;
    let finalProfile = effectiveProfile;

    // Cache Break-Even & Thinking-Signature Guard on non-hard mid-session switches
    if (existingPin && pinModelStillAvailable && selectedModel !== existingPin.model) {
      const curCandidate = pinCandidate;
      const nextCandidate = candidates.find(c => c.proxyId === selectedModel);
      if (curCandidate && nextCandidate) {
        const switchCheck = shouldAllowMidSessionSwitch({
          currentModel: curCandidate,
          nextModel: nextCandidate,
          currentContextTokens: effectiveContextTokens,
          hasThinkingSignature: verdict.hasThinkingSignature,
          isCompactionBoundary,
          isHardEscalation: false,
        });
        if (!switchCheck.allow) {
          selectedModel = existingPin.model;
          action = "pin_reuse";
          finalRole = existingPin.role;
          finalProfile = existingPin.profile;
          stats.pinReuses += 1;
        }
      }
    }

    if (action === "escalate") stats.escalations += 1;
    const recommendedEffort = resolveRecommendedEffort({
      action,
      profile: finalProfile,
      role: finalRole,
      complexity: action === "pin_reuse" && existingPin ? existingPin.complexity : signals.complexity,
      ...(stuckLoop ? { stuckLevel: stuckLoop.level } : {}),
      ...(action !== "route" && existingPin?.effort ? { priorEffort: existingPin.effort } : {}),
    });
    const scoreMs = performance.now() - tScoreStart;

    // Persist sticky pin for main sessions and multi-turn subagents
    const nextTurnCount = existingPin && !isCompactionBoundary ? existingPin.turnCount + 1 : 1;
    sessionStore.set({
      sessionKey,
      model: selectedModel,
      role: finalRole,
      profile: finalProfile,
      ...(recommendedEffort ? { effort: recommendedEffort } : {}),
      domain: signals.domain,
      complexity: signals.complexity,
      lastEstimatedTokens: effectiveContextTokens,
      turnCount: nextTurnCount,
      escalatedAtTurn: action === "escalate" ? nextTurnCount : (existingPin?.escalatedAtTurn ?? null),
      escalationCount: action === "escalate" ? (existingPin?.escalationCount ?? 0) + 1 : (existingPin?.escalationCount ?? 0),
      pinnedAtMs: existingPin && !isCompactionBoundary ? existingPin.pinnedAtMs : nowMs,
      lastUsedAtMs: nowMs,
    });

    return finalizeDecision(
      {
        selectedModel,
        rankedFallbacks: scoring.rankedCandidates.filter(c => c.proxyId !== selectedModel).map(c => c.proxyId),
        action,
        role: finalRole,
        profile: finalProfile,
        ...(recommendedEffort ? { recommendedEffort } : {}),
        sessionKey,
        cachePinned: true,
        signals,
        ...(stuckLoop ? { stuckLoop } : {}),
        scoring: {
          qualityScore: scoring.selected.qualityScore,
          qMax: scoring.qMax,
          qualityFloor: scoring.qualityFloor,
          effectiveCost: scoring.selected.effectiveCost,
          benchmarkRefs: scoring.selected.benchmarkRefs,
          baselineModel: scoring.baselineModel,
          estimatedSavingsPct: scoring.estimatedSavingsPct,
        },
        ...(signals.fallbackReason ? { fallbackReason: signals.fallbackReason } : {}),
        latencyMs: {
          total: performance.now() - tStart,
          fastPath: fastMs,
          classifier: clfMs,
          scoring: scoreMs,
        },
      },
      resolveContextWindow(selectedModel, candidates),
    );
  }

  return {
    route,
    getStats: () => ({ ...stats, activePins: sessionStore.size() }),
    clearSessions: () => sessionStore.clear(),
    snapshot,
    domains,
  };
}

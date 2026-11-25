import type {
  BenchmarkSnapshot,
  BridgeStats,
  BudgetProfile,
  CandidateModel,
  DomainDefinition,
  ExecutionRole,
  HarnessId,
  ScoringCandidateSummary,
  ScoringResult,
  SystemOneSignals,
} from "../types.js";

export interface ScoringInput {
  candidates: readonly CandidateModel[];
  snapshot: BenchmarkSnapshot;
  domains: readonly DomainDefinition[];
  signals: SystemOneSignals;
  profile: BudgetProfile;
  role: ExecutionRole;
  hasTools: boolean;
  requiresMultimodal: boolean;
  requiredContextTokens: number;
  harness: HarnessId;
  quotaUtilization?: Record<string, number>;
  switchingFromModel?: string;
  currentContextTokens?: number;
}

export interface DomainQualityResult {
  quality: number;
  verified: boolean;
  redFlagPenalty: number;
  refs: string[];
}

const LOGIT_EPSILON = 0.01;
const LCB_LAMBDA = 0.35;
const MIN_VALID_BRIDGE_RADJ = 0.30;
const CHAIN_DECAY_BASE = 0.85;
const CHAIN_DECAYS = Array.from({ length: 16 }, (_, i) => Math.pow(CHAIN_DECAY_BASE, i));
const UNBRIDGED_PENALTY_PER_HOP = 0.08;
const UNBRIDGED_WEIGHT_FACTOR = 0.40;
const UNVERIFIED_PRIOR_QUALITY = 0.25;
const MIN_QUALITY_SCORE = 0.05;
const FALLBACK_SD = 0.25;

const RED_FLAG_GAP_THRESHOLD = 0.30;
const RED_FLAG_OFFSET = 0.20;
const RED_FLAG_SLOPE = 0.75;
const MAX_RED_FLAG_PENALTY = 0.35;

const MIN_DOMAIN_PROB = 0.05;
const TOP_K_DOMAINS = 3;
const CODING_DOMAIN_ID = "software_engineering";
const TOOLS_DOMAIN_WEIGHT = 0.65;
const TOOLS_CODING_WEIGHT = 0.35;
const ORCHESTRATOR_MAIN_WEIGHT = 0.75;
const ORCHESTRATOR_GENERALIST_WEIGHT = 0.25;
const DEGRADED_HARNESS_PENALTY = 0.04;
const GENERALIST_BENCHMARKS = [
  "swe_bench_pro",
  "deepswe_1_1",
  "agents_last_exam",
  "gpqa",
] as const;

const CACHE_HIT_RATE = 0.80;
const OUTPUT_RATIO_FACTOR = 0.08;
const THINKING_RATIO_WEIGHT = 0.25;
const MAX_QUOTA_UTILIZATION = 0.95;
const QUOTA_PRESSURE_COST_FLOOR = 1e-4;

const MIN_QUALITY_FLOOR = 0.35;
const COST_TIE_RATIO = 0.12;

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}

function clampNonNegative(value: number | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return 0;
  return value;
}

export function isExplicitlyFreeModel(model: CandidateModel): boolean {
  return (
    /:free\b/i.test(model.proxyId) ||
    /-free\b/i.test(model.proxyId) ||
    model.proxyId.toLowerCase().includes("free")
  );
}

export function hasKnownPricing(model: CandidateModel): boolean {
  const knownInput = Number.isFinite(model.inputPricePerMTok) && model.inputPricePerMTok > 0;
  const knownOutput = Number.isFinite(model.outputPricePerMTok) && model.outputPricePerMTok > 0;
  return knownInput || knownOutput || isExplicitlyFreeModel(model);
}

function logit(p: number): number {
  const clipped = clamp(p, LOGIT_EPSILON, 1 - LOGIT_EPSILON);
  return Math.log(clipped / (1 - clipped));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function predictBridgedScore(rawFallback: number, bridge: BridgeStats): number {
  const y = logit(rawFallback);
  const sdY = bridge.sdY > 0 ? bridge.sdY : FALLBACK_SD;
  const xHat = bridge.muX + bridge.rAdj * (bridge.sdX / sdY) * (y - bridge.muY);
  const sigmaRes = bridge.sdX * Math.sqrt(Math.max(0, 1 - bridge.rAdj * bridge.rAdj));
  return sigmoid(xHat - LCB_LAMBDA * sigmaRes);
}

/**
 * Compute cache-, thinking-, switching-, and quota-pressure-adjusted effective cost per MTok-equivalent.
 */
export function computeEffectiveCost(
  model: CandidateModel,
  opts: { quotaUtilization?: number; switching?: boolean; contextTokens?: number } = {},
): number {
  const inputPrice = clampNonNegative(model.inputPricePerMTok);
  const outputPrice = clampNonNegative(model.outputPricePerMTok);
  const cacheReadMult = clampNonNegative(model.cacheReadMultiplier);
  const cacheWriteMult = clampNonNegative(model.cacheWriteMultiplier);
  const thinkingRatio = clampNonNegative(model.thinkingOutputRatio);
  const quotaWeight = clampNonNegative(model.quotaWeightMultiplier);

  const hEff = model.proxyStripsCacheControl ? 0 : CACHE_HIT_RATE;
  const pInEff = inputPrice * (hEff * cacheReadMult + (1 - hEff) * cacheWriteMult);
  const pOutEff = outputPrice * OUTPUT_RATIO_FACTOR * (1 + THINKING_RATIO_WEIGHT * thinkingRatio);

  const contextTokens = clampNonNegative(opts.contextTokens);
  const switchTax =
    opts.switching && contextTokens > 0
      ? (contextTokens / 1_000_000) * inputPrice * cacheWriteMult
      : 0;

  const baseCost = (pInEff + pOutEff + switchTax) * quotaWeight;
  const u = clampNonNegative(opts.quotaUtilization);
  if (u > 0) {
    const quotaPressure = 1 / (1 - Math.min(u, MAX_QUOTA_UTILIZATION));
    return Math.max(baseCost, QUOTA_PRESSURE_COST_FLOOR) * quotaPressure;
  }

  return baseCost;
}

/**
 * Resolve a model's calibrated quality score on a single domain using multi-benchmark
 * Logit-OLS/LCB pooling and the Sibling Red-Flag Penalty.
 */
export function resolveDomainQuality(
  model: CandidateModel,
  domain: DomainDefinition,
  snapshot: BenchmarkSnapshot,
): DomainQualityResult {
  const chain = domain.benchmarkChain;
  if (chain.length === 0) {
    return { quality: UNVERIFIED_PRIOR_QUALITY, verified: false, redFlagPenalty: 0, refs: [] };
  }

  const primary = chain[0];
  let verWeightSum = 0;
  let verWeightedSum = 0;
  let minVerifiedQ = Number.POSITIVE_INFINITY;
  const verRefs: string[] = [];

  let fbWeightSum = 0;
  let fbWeightedSum = 0;
  const fbRefs: string[] = [];

  for (let i = 0; i < chain.length; i++) {
    const bench = chain[i];
    const raw = model.scores[bench];
    if (typeof raw !== "number" || !Number.isFinite(raw)) continue;

    if (i === 0) {
      const q = clamp(raw, 0, 1);
      verWeightSum += 1.0;
      verWeightedSum += q;
      if (q < minVerifiedQ) minVerifiedQ = q;
      verRefs.push(`${bench}=${raw.toFixed(3)}`);
      continue;
    }

    const decay = i < CHAIN_DECAYS.length ? CHAIN_DECAYS[i] : Math.pow(CHAIN_DECAY_BASE, i);
    const bridge = snapshot.bridges[`${primary}|${bench}`];
    if (bridge && Number.isFinite(bridge.rAdj) && bridge.rAdj >= MIN_VALID_BRIDGE_RADJ) {
      const qEst = predictBridgedScore(raw, bridge);
      if (!Number.isFinite(qEst)) continue;
      const w = decay * bridge.rAdj;
      if (!Number.isFinite(w) || w <= 0) continue;
      verWeightSum += w;
      verWeightedSum += w * qEst;
      if (qEst < minVerifiedQ) minVerifiedQ = qEst;
      verRefs.push(`${bench}->${primary}=${qEst.toFixed(3)}`);
    } else if (verWeightSum === 0) {
      const penaltyFactor = Math.max(0.1, 1 - UNBRIDGED_PENALTY_PER_HOP * i);
      const qEst = clamp(raw * penaltyFactor, LOGIT_EPSILON, 1);
      if (!Number.isFinite(qEst)) continue;
      const w = UNBRIDGED_WEIGHT_FACTOR * decay;
      fbWeightSum += w;
      fbWeightedSum += w * qEst;
      fbRefs.push(`${bench}~${qEst.toFixed(3)}`);
    }
  }

  const verified = verWeightSum > 0;
  const weightSum = verified ? verWeightSum : fbWeightSum;
  if (weightSum <= 0) {
    return { quality: UNVERIFIED_PRIOR_QUALITY, verified: false, redFlagPenalty: 0, refs: [] };
  }

  const refs = verified ? verRefs : fbRefs;
  let qPooled = (verified ? verWeightedSum : fbWeightedSum) / weightSum;
  let redFlagPenalty = 0;

  let minSibling = verified ? minVerifiedQ : Number.POSITIVE_INFINITY;
  const siblings = domain.redFlagSiblings;
  if (siblings) {
    for (let i = 0; i < siblings.length; i++) {
      const score = model.scores[siblings[i]];
      if (typeof score === "number" && Number.isFinite(score)) {
        const clampedScore = clamp(score, 0, 1);
        if (clampedScore < minSibling) {
          minSibling = clampedScore;
        }
      }
    }
  }

  if (Number.isFinite(minSibling) && qPooled - minSibling > RED_FLAG_GAP_THRESHOLD) {
    redFlagPenalty = Math.min(
      MAX_RED_FLAG_PENALTY,
      (qPooled - minSibling - RED_FLAG_OFFSET) * RED_FLAG_SLOPE,
    );
    qPooled = Math.max(MIN_QUALITY_SCORE, qPooled - redFlagPenalty);
  }

  return { quality: clamp(qPooled, 0, 1), verified, redFlagPenalty, refs };
}

function checkHardEligibility(model: CandidateModel, input: ScoringInput): string | null {
  const reasons: string[] = [];
  if (model.contextWindow < input.requiredContextTokens) {
    reasons.push(`context_window_too_small(${model.contextWindow}<${input.requiredContextTokens})`);
  }
  if (input.hasTools && !model.supportsTools) {
    reasons.push("tools_unsupported");
  }
  if (input.requiresMultimodal && !model.multimodal) {
    reasons.push("multimodal_unsupported");
  }
  if (model.harnessCompat[input.harness] === "unsupported") {
    reasons.push(`harness_unsupported(${input.harness})`);
  }
  return reasons.length > 0 ? reasons.join("; ") : null;
}

function resolveQuotaUtilization(
  model: CandidateModel,
  quotaUtilization?: Record<string, number>,
): number {
  if (!quotaUtilization) return 0;

  const direct = quotaUtilization[model.proxyId];
  if (typeof direct === "number" && Number.isFinite(direct)) return direct;

  if (model.canonicalId) {
    const canonical = quotaUtilization[model.canonicalId];
    if (typeof canonical === "number" && Number.isFinite(canonical)) return canonical;
  }

  const slashIdx = model.proxyId.indexOf("/");
  if (slashIdx > 0) {
    const prefix = model.proxyId.slice(0, slashIdx);
    const byPrefix = quotaUtilization[prefix] ?? quotaUtilization[`${prefix}/`];
    if (typeof byPrefix === "number" && Number.isFinite(byPrefix)) return byPrefix;
  }

  const ocxMatch = /^ocx-claude-([a-z0-9._-]+)--/i.exec(model.proxyId);
  if (ocxMatch) {
    const byOcx = quotaUtilization[ocxMatch[1]];
    if (typeof byOcx === "number" && Number.isFinite(byOcx)) return byOcx;
  }

  const orgKey = model.organization.toLowerCase();
  const byOrg = quotaUtilization[orgKey];
  if (typeof byOrg === "number" && Number.isFinite(byOrg)) return byOrg;

  let bestLen = 0;
  let bestVal = 0;
  for (const [key, val] of Object.entries(quotaUtilization)) {
    if (
      key.length > bestLen &&
      typeof val === "number" &&
      Number.isFinite(val) &&
      model.proxyId.startsWith(key)
    ) {
      bestLen = key.length;
      bestVal = val;
    }
  }
  return bestVal;
}

function resolveActiveDomains(
  signals: SystemOneSignals,
  domainById: ReadonlyMap<string, DomainDefinition>,
  domains: readonly DomainDefinition[],
): Array<{ domain: DomainDefinition; weight: number }> {
  const candidates: Array<{ domain: DomainDefinition; prob: number }> = [];
  for (const [id, prob] of Object.entries(signals.domainProbabilities)) {
    if (typeof prob !== "number" || !Number.isFinite(prob) || prob < MIN_DOMAIN_PROB) continue;
    const def = domainById.get(id);
    if (def) candidates.push({ domain: def, prob });
  }

  candidates.sort((a, b) => b.prob - a.prob);
  const top = candidates.slice(0, TOP_K_DOMAINS);

  if (top.length > 0) {
    const total = top.reduce((sum, item) => sum + item.prob, 0);
    return top.map(item => ({ domain: item.domain, weight: item.prob / total }));
  }

  const fallbackDomain = domainById.get(signals.domain) ?? domains[0];
  return fallbackDomain ? [{ domain: fallbackDomain, weight: 1.0 }] : [];
}

function computeGeneralistQuality(model: CandidateModel): number {
  let sum = 0;
  let count = 0;
  for (const bench of GENERALIST_BENCHMARKS) {
    const score = model.scores[bench];
    if (typeof score === "number" && Number.isFinite(score)) {
      sum += clamp(score, 0, 1);
      count += 1;
    }
  }
  return count > 0 ? sum / count : UNVERIFIED_PRIOR_QUALITY;
}

function computeDeltaQ(profile: BudgetProfile, role: ExecutionRole, complexity: number): number {
  const c = clamp(complexity, 0, 1);
  if (profile === "quality") {
    return (0.02 + (1 - c) * 0.04) * (role === "orchestrator" ? 0.7 : 1);
  }
  if (profile === "frugal") {
    return (0.10 + (1 - c) * 0.22) * (role === "orchestrator" ? 0.7 : 1);
  }
  const roleMult = role === "orchestrator" ? 0.65 : role === "subagent" ? 1.15 : 1;
  return (0.05 + (1 - c) * 0.14) * roleMult;
}

function scoreCandidate(
  model: CandidateModel,
  activeDomains: readonly { domain: DomainDefinition; weight: number }[],
  codingDomain: DomainDefinition | undefined,
  input: ScoringInput,
): ScoringCandidateSummary {
  let qDom = 0;
  let verifiedWeight = 0;
  let weightedRedFlag = 0;
  const refSet = new Set<string>();
  let codingRes: DomainQualityResult | undefined;

  for (const { domain, weight } of activeDomains) {
    const res = resolveDomainQuality(model, domain, input.snapshot);
    qDom += weight * res.quality;
    if (res.verified) verifiedWeight += weight;
    weightedRedFlag += weight * res.redFlagPenalty;
    for (const ref of res.refs) refSet.add(ref);
    if (domain.id === CODING_DOMAIN_ID) codingRes = res;
  }

  let qEff = activeDomains.length > 0 ? qDom : UNVERIFIED_PRIOR_QUALITY;
  let redFlagPenalty = weightedRedFlag;

  if (input.hasTools && codingDomain) {
    const cRes = codingRes ?? resolveDomainQuality(model, codingDomain, input.snapshot);
    qEff = TOOLS_DOMAIN_WEIGHT * qEff + TOOLS_CODING_WEIGHT * cRes.quality;
    redFlagPenalty = TOOLS_DOMAIN_WEIGHT * redFlagPenalty + TOOLS_CODING_WEIGHT * cRes.redFlagPenalty;
    if (!codingRes) {
      for (const ref of cRes.refs) refSet.add(ref);
    }
  }

  if (input.role === "orchestrator") {
    const qGeneralist = computeGeneralistQuality(model);
    qEff = ORCHESTRATOR_MAIN_WEIGHT * qEff + ORCHESTRATOR_GENERALIST_WEIGHT * qGeneralist;
  }

  if (model.harnessCompat[input.harness] === "degraded") {
    qEff = Math.max(0, qEff - DEGRADED_HARNESS_PENALTY);
  }

  const utilization = resolveQuotaUtilization(model, input.quotaUtilization);
  const switching = Boolean(
    input.switchingFromModel && input.switchingFromModel !== model.proxyId,
  );
  const effectiveCost = computeEffectiveCost(model, {
    quotaUtilization: utilization,
    switching,
    contextTokens: input.currentContextTokens,
  });

  return {
    proxyId: model.proxyId,
    canonicalId: model.canonicalId,
    qualityScore: clamp(qEff, 0, 1),
    effectiveCost,
    verified: verifiedWeight >= 0.5,
    redFlagPenalty,
    benchmarkRefs: [...refSet],
  };
}

function rankInBand(band: readonly ScoringCandidateSummary[]): ScoringCandidateSummary[] {
  const remaining = [...band].sort((a, b) => {
    if (a.effectiveCost !== b.effectiveCost) return a.effectiveCost - b.effectiveCost;
    if (b.qualityScore !== a.qualityScore) return b.qualityScore - a.qualityScore;
    return a.proxyId.localeCompare(b.proxyId);
  });

  const ranked: ScoringCandidateSummary[] = [];
  while (remaining.length > 0) {
    const minCost = remaining[0].effectiveCost;
    let bestIdx = 0;

    for (let i = 1; i < remaining.length; i++) {
      const candidate = remaining[i];
      const withinTieBand =
        candidate.effectiveCost === 0 && minCost === 0
          ? true
          : candidate.effectiveCost > 0 &&
            (candidate.effectiveCost - minCost) / candidate.effectiveCost <= COST_TIE_RATIO;
      if (!withinTieBand) break;

      const best = remaining[bestIdx];
      if (
        candidate.qualityScore > best.qualityScore ||
        (candidate.qualityScore === best.qualityScore &&
          candidate.effectiveCost < best.effectiveCost)
      ) {
        bestIdx = i;
      }
    }

    ranked.push(remaining[bestIdx]);
    remaining.splice(bestIdx, 1);
  }

  return ranked;
}

function pickBaseline(pool: readonly ScoringCandidateSummary[]): ScoringCandidateSummary {
  let baseline = pool[0];
  for (let i = 1; i < pool.length; i++) {
    const c = pool[i];
    if (
      c.effectiveCost > baseline.effectiveCost ||
      (c.effectiveCost === baseline.effectiveCost && c.qualityScore > baseline.qualityScore)
    ) {
      baseline = c;
    }
  }
  return baseline;
}

/**
 * Score all candidate models and select the optimal Pareto-efficient model for the request.
 */
export function scoreAndSelectModel(input: ScoringInput): ScoringResult {
  if (input.candidates.length === 0) {
    throw new Error("scoreAndSelectModel requires at least one candidate model");
  }

  const excludedReasons: Record<string, string> = {};
  const eligible: CandidateModel[] = [];

  for (const model of input.candidates) {
    const reason = checkHardEligibility(model, input);
    if (reason) {
      excludedReasons[model.proxyId] = reason;
    } else {
      eligible.push(model);
    }
  }

  // Graceful fallback: never throw if all candidates are excluded; pick largest context window
  if (eligible.length === 0) {
    let largest = input.candidates[0];
    for (let i = 1; i < input.candidates.length; i++) {
      if (input.candidates[i].contextWindow > largest.contextWindow) {
        largest = input.candidates[i];
      }
    }
    eligible.push(largest);
  }

  const domainById = new Map<string, DomainDefinition>(input.domains.map(d => [d.id, d]));
  const activeDomains = resolveActiveDomains(input.signals, domainById, input.domains);
  const codingDomain = domainById.get(CODING_DOMAIN_ID);

  const scoredEntries = eligible.map(model => ({
    summary: scoreCandidate(model, activeDomains, codingDomain, input),
    knownPricing: hasKnownPricing(model),
  }));

  // Exclude verified models with unknown ($0) pricing when verified priced candidates exist
  const hasVerifiedPriced = scoredEntries.some(e => e.summary.verified && e.knownPricing);
  const pricedEntries = hasVerifiedPriced
    ? scoredEntries.filter(e => {
        if (e.summary.verified && !e.knownPricing) {
          excludedReasons[e.summary.proxyId] = "unknown_pricing";
          return false;
        }
        return true;
      })
    : scoredEntries;

  // No-Evidence Gate: exclude unverified models in balanced/quality when verified models exist
  const verifiedPool = pricedEntries.filter(e => e.summary.verified).map(e => e.summary);
  const enforceVerified =
    (input.profile === "balanced" || input.profile === "quality") && verifiedPool.length > 0;

  let activePool: ScoringCandidateSummary[];
  if (enforceVerified) {
    activePool = verifiedPool;
    for (const e of pricedEntries) {
      if (!e.summary.verified) {
        excludedReasons[e.summary.proxyId] = "unverified_benchmark_evidence";
      }
    }
  } else {
    activePool = pricedEntries.map(e => e.summary);
  }

  let qMax = activePool[0].qualityScore;
  for (let i = 1; i < activePool.length; i++) {
    if (activePool[i].qualityScore > qMax) qMax = activePool[i].qualityScore;
  }

  const deltaQ = computeDeltaQ(input.profile, input.role, input.signals.complexity);
  const rawFloor = Math.max(0, qMax - deltaQ);
  let effectiveQualityFloor = Math.max(MIN_QUALITY_FLOOR, rawFloor);

  let inBand = activePool.filter(c => c.qualityScore >= effectiveQualityFloor);
  if (inBand.length === 0) {
    // Handle ultra-hard domains where qMax < 0.35 without throwing
    effectiveQualityFloor = rawFloor;
    inBand = activePool.filter(c => c.qualityScore >= effectiveQualityFloor);
  }
  const inBandIds = new Set(inBand.map(c => c.proxyId));
  const outOfBand = activePool
    .filter(c => !inBandIds.has(c.proxyId))
    .sort((a, b) => {
      if (b.qualityScore !== a.qualityScore) return b.qualityScore - a.qualityScore;
      return a.effectiveCost - b.effectiveCost;
    });

  const rankedBand = rankInBand(inBand);
  const rankedCandidates = [...rankedBand, ...outOfBand];
  const selected = rankedCandidates[0];

  const baselinePool = verifiedPool.length > 0 ? verifiedPool : activePool;
  const baseline = pickBaseline(baselinePool);
  const estimatedSavingsPct =
    baseline.effectiveCost > 0
      ? Math.max(0, ((baseline.effectiveCost - selected.effectiveCost) / baseline.effectiveCost) * 100)
      : 0;

  return {
    selected,
    rankedCandidates,
    qMax,
    qualityFloor: effectiveQualityFloor,
    deltaQ,
    baselineModel: baseline.proxyId,
    baselineEffectiveCost: baseline.effectiveCost,
    estimatedSavingsPct,
    excludedReasons,
  };
}

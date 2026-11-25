import { existsSync, lstatSync, readdirSync, readFileSync } from "node:fs";
import { basename, join } from "node:path";
import bundledDomains from "../../data/domains.json";
import bundledSnapshot from "../../data/snapshot.json";
import { buildCandidatePool, resolveCanonicalModelId } from "../core/catalog.js";
import { pickAuxiliaryModel } from "../core/fast-path.js";
import { scoreAndSelectModel } from "../core/scorer.js";
import { BUILTIN_BACKEND_CALIBRATIONS } from "../core/systemone-client.js";
import type {
  BenchmarkSnapshot,
  BudgetProfile,
  CandidateModel,
  DomainDefinition,
  ExecutionRole,
  HarnessId,
  SnapshotModelEntry,
  SystemOneSignals,
} from "../types.js";

export interface ReplaySessionInput {
  id: string;
  project?: string;
  isSubagent: boolean;
  hasTools: boolean;
  signals: SystemOneSignals;
  turns: Array<{
    actualModel: string;
    inputTokens: number;
    cacheReadTokens: number;
    cacheWriteTokens: number;
    outputTokens: number;
    toolUses?: number;
    toolErrors?: number;
  }>;
}

export interface ReplayOptions {
  sessions: readonly ReplaySessionInput[];
  availableModels: readonly string[];
  snapshot?: BenchmarkSnapshot;
  domains?: readonly DomainDefinition[];
  flagshipModel?: string;
  auxiliaryModelHints?: readonly string[];
  harness?: HarnessId;
}

export interface ReplayProfileSummary {
  totalCostUsd: number;
  savedVsFlagshipPct: number;
  savedVsActualPct: number;
  modelMix: Record<string, number>;
}

export interface ReplayReport {
  sessionsCount: number;
  totalTurns: number;
  cacheReadSharePct: number;
  actualCostUsd: number;
  flagshipCostUsd: number;
  naivePerTurnCostUsd: number;
  profiles: Record<BudgetProfile, ReplayProfileSummary>;
}

interface PricingSpec {
  inputPricePerMTok: number;
  outputPricePerMTok: number;
  cacheReadMultiplier: number;
  cacheWriteMultiplier: number;
}

type BackendKey = "clef27b" | "tev1_4b" | "tev1_08b";

const PROFILES: readonly BudgetProfile[] = ["frugal", "balanced", "quality"];
const DEFAULT_FLAGSHIP_CANONICAL = "claude-opus-5-5";
const DEFAULT_AUX_CANONICAL = "claude-haiku-5-5";
const NOUL_ORCH_THRESHOLD = 0.65;
const MIN_CONTEXT_TOKENS = 32_000;

const BACKEND_METADATA: Record<BackendKey, { name: string; complexityOffset: number }> = {
  clef27b: {
    name: "clef:27b",
    complexityOffset: BUILTIN_BACKEND_CALIBRATIONS["clef:27b"]?.complexityOffset ?? 0,
  },
  tev1_4b: {
    name: "tev1:4b",
    complexityOffset: BUILTIN_BACKEND_CALIBRATIONS["tev1:4b"]?.complexityOffset ?? 0,
  },
  tev1_08b: {
    name: "tev1:0.8b",
    complexityOffset: BUILTIN_BACKEND_CALIBRATIONS["tev1:0.8b"]?.complexityOffset ?? -0.15,
  },
};

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function roundUsd(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function roundPct(value: number): number {
  return Math.round(value * 100) / 100;
}

function computeSavedPct(cost: number, baseline: number): number {
  if (baseline <= 0) return 0;
  return roundPct(((baseline - cost) / baseline) * 100);
}

function pricingFromCandidate(candidate: CandidateModel): PricingSpec {
  return {
    inputPricePerMTok: candidate.inputPricePerMTok,
    outputPricePerMTok: candidate.outputPricePerMTok,
    cacheReadMultiplier: candidate.cacheReadMultiplier,
    cacheWriteMultiplier: candidate.cacheWriteMultiplier,
  };
}

function pricingFromSnapshotEntry(entry: SnapshotModelEntry): PricingSpec {
  return {
    inputPricePerMTok: entry.inputPricePerMTok,
    outputPricePerMTok: entry.outputPricePerMTok,
    cacheReadMultiplier: entry.cacheReadMultiplier ?? 0.25,
    cacheWriteMultiplier: entry.cacheWriteMultiplier ?? 1.0,
  };
}

function hasPositivePricing(spec: PricingSpec): boolean {
  return spec.inputPricePerMTok > 0 || spec.outputPricePerMTok > 0;
}

function resolvePricingForModelId(
  modelId: string,
  byProxyId: ReadonlyMap<string, CandidateModel>,
  snapshot: BenchmarkSnapshot,
  fallback: PricingSpec,
): PricingSpec {
  const directCandidate = byProxyId.get(modelId);
  if (directCandidate) {
    const spec = pricingFromCandidate(directCandidate);
    if (hasPositivePricing(spec)) return spec;
  }

  const canonicalId = resolveCanonicalModelId(modelId, snapshot);
  if (canonicalId) {
    const entry = snapshot.models[canonicalId];
    if (entry) {
      const spec = pricingFromSnapshotEntry(entry);
      if (hasPositivePricing(spec)) return spec;
    }
  }

  return fallback;
}

function resolveFlagship(
  flagshipModel: string | undefined,
  candidates: readonly CandidateModel[],
  byProxyId: ReadonlyMap<string, CandidateModel>,
  snapshot: BenchmarkSnapshot,
): { id: string; pricing: PricingSpec } {
  if (flagshipModel) {
    const direct = byProxyId.get(flagshipModel);
    if (direct) {
      return { id: direct.proxyId, pricing: pricingFromCandidate(direct) };
    }
    const canonical = resolveCanonicalModelId(flagshipModel, snapshot);
    if (canonical && snapshot.models[canonical]) {
      const cand = candidates.find(c => c.canonicalId === canonical);
      return {
        id: cand?.proxyId ?? flagshipModel,
        pricing: cand ? pricingFromCandidate(cand) : pricingFromSnapshotEntry(snapshot.models[canonical]),
      };
    }
  }

  const opusCandidate = candidates.find(c => c.canonicalId === DEFAULT_FLAGSHIP_CANONICAL);
  if (opusCandidate && hasPositivePricing(pricingFromCandidate(opusCandidate))) {
    return { id: opusCandidate.proxyId, pricing: pricingFromCandidate(opusCandidate) };
  }

  const opusEntry = snapshot.models[DEFAULT_FLAGSHIP_CANONICAL];
  if (opusEntry && hasPositivePricing(pricingFromSnapshotEntry(opusEntry))) {
    return { id: DEFAULT_FLAGSHIP_CANONICAL, pricing: pricingFromSnapshotEntry(opusEntry) };
  }

  let highest = candidates[0];
  let highestScore = highest.inputPricePerMTok + highest.outputPricePerMTok;
  for (let i = 1; i < candidates.length; i++) {
    const score = candidates[i].inputPricePerMTok + candidates[i].outputPricePerMTok;
    if (score > highestScore) {
      highest = candidates[i];
      highestScore = score;
    }
  }
  return { id: highest.proxyId, pricing: pricingFromCandidate(highest) };
}

function resolveAuxiliaryPricing(
  availableModels: readonly string[],
  hints: readonly string[] | undefined,
  byProxyId: ReadonlyMap<string, CandidateModel>,
  snapshot: BenchmarkSnapshot,
): PricingSpec | null {
  const picked = pickAuxiliaryModel(availableModels, hints);
  if (picked) {
    const cand = byProxyId.get(picked);
    if (cand) {
      const spec = pricingFromCandidate(cand);
      if (hasPositivePricing(spec)) return spec;
    }
    const canonical = resolveCanonicalModelId(picked, snapshot);
    if (canonical && snapshot.models[canonical]) {
      const spec = pricingFromSnapshotEntry(snapshot.models[canonical]);
      if (hasPositivePricing(spec)) return spec;
    }
  }

  const haikuEntry = snapshot.models[DEFAULT_AUX_CANONICAL];
  if (haikuEntry) {
    const spec = pricingFromSnapshotEntry(haikuEntry);
    if (hasPositivePricing(spec)) return spec;
  }
  return null;
}

function computeTurnCostUsd(
  pricing: PricingSpec,
  turn: ReplaySessionInput["turns"][number],
  cacheMiss = false,
): number {
  const inPerTok = pricing.inputPricePerMTok / 1_000_000;
  const outPerTok = pricing.outputPricePerMTok / 1_000_000;
  if (cacheMiss) {
    const missMult = Math.max(1.0, pricing.cacheWriteMultiplier);
    return (
      (turn.inputTokens + turn.cacheReadTokens + turn.cacheWriteTokens) * inPerTok * missMult +
      turn.outputTokens * outPerTok
    );
  }
  return (
    turn.inputTokens * inPerTok +
    turn.cacheReadTokens * inPerTok * pricing.cacheReadMultiplier +
    turn.cacheWriteTokens * inPerTok * pricing.cacheWriteMultiplier +
    turn.outputTokens * outPerTok
  );
}

function resolveReplayRole(session: ReplaySessionInput, profile: BudgetProfile): ExecutionRole {
  if (session.isSubagent) return "subagent";
  const noul = session.signals.needsFlagshipOrchestrator;
  if (noul >= NOUL_ORCH_THRESHOLD) return "orchestrator";
  if (noul <= 1 - NOUL_ORCH_THRESHOLD) return "specialist";
  return profile === "quality" ? "orchestrator" : "specialist";
}

function isAuxiliaryTurn(turn: ReplaySessionInput["turns"][number]): boolean {
  return (
    (turn.toolUses ?? 0) === 0 &&
    turn.inputTokens + turn.cacheReadTokens < 1500 &&
    turn.outputTokens < 300
  );
}

function sortModelMixDescending(counts: ReadonlyMap<string, number>): Record<string, number> {
  const entries = [...counts.entries()].sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1];
    return a[0].localeCompare(b[0]);
  });
  return Object.fromEntries(entries);
}

function emptyReport(): ReplayReport {
  const emptyProfile = (): ReplayProfileSummary => ({
    totalCostUsd: 0,
    savedVsFlagshipPct: 0,
    savedVsActualPct: 0,
    modelMix: {},
  });
  return {
    sessionsCount: 0,
    totalTurns: 0,
    cacheReadSharePct: 0,
    actualCostUsd: 0,
    flagshipCostUsd: 0,
    naivePerTurnCostUsd: 0,
    profiles: {
      frugal: emptyProfile(),
      balanced: emptyProfile(),
      quality: emptyProfile(),
    },
  };
}

/**
 * Run deterministic counterfactual cost & model-mix replay across recorded sessions
 * using the production candidate catalog and Pareto scorer.
 */
export function runCounterfactualReplay(opts: ReplayOptions): ReplayReport {
  if (opts.sessions.length === 0) {
    return emptyReport();
  }

  const snapshot: BenchmarkSnapshot =
    opts.snapshot ?? (bundledSnapshot as unknown as BenchmarkSnapshot);
  const domains: readonly DomainDefinition[] =
    opts.domains ?? (bundledDomains as DomainDefinition[]);
  const harness: HarnessId = opts.harness ?? "claude_code";

  const allCandidates = buildCandidatePool(opts.availableModels, snapshot);
  if (allCandidates.length === 0) {
    throw new Error("runCounterfactualReplay requires at least one eligible available model");
  }

  const pricedCandidates = allCandidates.filter(c =>
    hasPositivePricing(pricingFromCandidate(c)),
  );
  const scoringCandidates = pricedCandidates.length > 0 ? pricedCandidates : allCandidates;
  const byProxyId = new Map<string, CandidateModel>(
    allCandidates.map(c => [c.proxyId, c]),
  );

  const flagship = resolveFlagship(opts.flagshipModel, scoringCandidates, byProxyId, snapshot);
  const auxPricing = resolveAuxiliaryPricing(
    opts.availableModels,
    opts.auxiliaryModelHints,
    byProxyId,
    snapshot,
  );

  let totalTurns = 0;
  let totalInputTokens = 0;
  let totalCacheReadTokens = 0;
  let actualCostUsd = 0;
  let flagshipCostUsd = 0;
  let naivePerTurnCostUsd = 0;

  const profileCosts: Record<BudgetProfile, number> = {
    frugal: 0,
    balanced: 0,
    quality: 0,
  };
  const profileMixCounts: Record<BudgetProfile, Map<string, number>> = {
    frugal: new Map<string, number>(),
    balanced: new Map<string, number>(),
    quality: new Map<string, number>(),
  };

  for (const session of opts.sessions) {
    let maxContextTokens = MIN_CONTEXT_TOKENS;
    for (const turn of session.turns) {
      const turnCtx = turn.inputTokens + turn.cacheReadTokens + turn.cacheWriteTokens;
      if (turnCtx > maxContextTokens) maxContextTokens = turnCtx;
    }

    const selectedByProfile: Record<BudgetProfile, { proxyId: string; pricing: PricingSpec }> = {
      frugal: { proxyId: flagship.id, pricing: flagship.pricing },
      balanced: { proxyId: flagship.id, pricing: flagship.pricing },
      quality: { proxyId: flagship.id, pricing: flagship.pricing },
    };

    for (const profile of PROFILES) {
      const role = resolveReplayRole(session, profile);
      const scoring = scoreAndSelectModel({
        candidates: scoringCandidates,
        snapshot,
        domains,
        signals: session.signals,
        profile,
        role,
        hasTools: session.hasTools,
        requiresMultimodal: false,
        requiredContextTokens: maxContextTokens,
        harness,
      });
      const proxyId = scoring.selected.proxyId;
      const pricing = resolvePricingForModelId(proxyId, byProxyId, snapshot, flagship.pricing);
      selectedByProfile[profile] = { proxyId, pricing };
      profileMixCounts[profile].set(proxyId, (profileMixCounts[profile].get(proxyId) ?? 0) + 1);
    }

    let prevNaiveModelId = "";
    for (const turn of session.turns) {
      totalTurns += 1;
      totalInputTokens += turn.inputTokens + turn.cacheReadTokens + turn.cacheWriteTokens;
      totalCacheReadTokens += turn.cacheReadTokens;

      const actualPricing = resolvePricingForModelId(
        turn.actualModel,
        byProxyId,
        snapshot,
        flagship.pricing,
      );
      actualCostUsd += computeTurnCostUsd(actualPricing, turn, false);
      flagshipCostUsd += computeTurnCostUsd(flagship.pricing, turn, false);

      const auxTurn = isAuxiliaryTurn(turn);
      for (const profile of PROFILES) {
        const pinned = selectedByProfile[profile].pricing;
        const turnPricing =
          auxTurn &&
          auxPricing !== null &&
          auxPricing.inputPricePerMTok <= pinned.inputPricePerMTok
            ? auxPricing
            : pinned;
        profileCosts[profile] += computeTurnCostUsd(turnPricing, turn, false);
      }

      const naiveHard = (turn.toolUses ?? 0) >= 2 || (turn.toolErrors ?? 0) > 0;
      const naiveTargetId = naiveHard ? flagship.id : selectedByProfile.balanced.proxyId;
      const naivePricing = naiveHard ? flagship.pricing : selectedByProfile.balanced.pricing;
      const switched = prevNaiveModelId !== "" && prevNaiveModelId !== naiveTargetId;
      prevNaiveModelId = naiveTargetId;
      naivePerTurnCostUsd += computeTurnCostUsd(naivePricing, turn, switched);
    }
  }

  const cacheReadSharePct =
    totalInputTokens > 0 ? roundPct((totalCacheReadTokens / totalInputTokens) * 100) : 0;

  const profiles: Record<BudgetProfile, ReplayProfileSummary> = {
    frugal: {
      totalCostUsd: roundUsd(profileCosts.frugal),
      savedVsFlagshipPct: computeSavedPct(profileCosts.frugal, flagshipCostUsd),
      savedVsActualPct: computeSavedPct(profileCosts.frugal, actualCostUsd),
      modelMix: sortModelMixDescending(profileMixCounts.frugal),
    },
    balanced: {
      totalCostUsd: roundUsd(profileCosts.balanced),
      savedVsFlagshipPct: computeSavedPct(profileCosts.balanced, flagshipCostUsd),
      savedVsActualPct: computeSavedPct(profileCosts.balanced, actualCostUsd),
      modelMix: sortModelMixDescending(profileMixCounts.balanced),
    },
    quality: {
      totalCostUsd: roundUsd(profileCosts.quality),
      savedVsFlagshipPct: computeSavedPct(profileCosts.quality, flagshipCostUsd),
      savedVsActualPct: computeSavedPct(profileCosts.quality, actualCostUsd),
      modelMix: sortModelMixDescending(profileMixCounts.quality),
    },
  };

  return {
    sessionsCount: opts.sessions.length,
    totalTurns,
    cacheReadSharePct,
    actualCostUsd: roundUsd(actualCostUsd),
    flagshipCostUsd: roundUsd(flagshipCostUsd),
    naivePerTurnCostUsd: roundUsd(naivePerTurnCostUsd),
    profiles,
  };
}

interface RawSignalRecord {
  domain?: string;
  probs?: Record<string, number>;
  conf?: number;
  comp?: number;
  noul?: number;
  latencyMs?: number;
  ok?: boolean;
}

interface RawSessionRecord {
  id?: string;
  project?: string;
  isSubagent?: boolean;
  turns?: number;
  hasTools?: boolean;
  clef27b?: RawSignalRecord;
  tev1_4b?: RawSignalRecord;
  tev1_08b?: RawSignalRecord;
}

interface RawSignalsFile {
  sessions?: RawSessionRecord[];
}

const MAX_JSONL_FILE_BYTES = 64 * 1024 * 1024;
const MAX_SYNTHESIZED_TURNS = 10_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sanitizeTokenCount(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : 0;
}

function parseClaudeJsonlTurns(filePath: string): ReplaySessionInput["turns"] {
  let raw: string;
  try {
    const st = lstatSync(filePath);
    if (st.isSymbolicLink() || !st.isFile() || st.size > MAX_JSONL_FILE_BYTES) {
      return [];
    }
    raw = readFileSync(filePath, "utf8");
  } catch {
    return [];
  }

  const turns: ReplaySessionInput["turns"] = [];
  const lines = raw.split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (!isRecord(parsed)) continue;

    if (parsed.type === "user" && isRecord(parsed.message) && Array.isArray(parsed.message.content)) {
      let errs = 0;
      for (const block of parsed.message.content) {
        if (isRecord(block) && block.type === "tool_result" && block.is_error === true) {
          errs += 1;
        }
      }
      if (errs > 0 && turns.length > 0) {
        turns[turns.length - 1].toolErrors = (turns[turns.length - 1].toolErrors ?? 0) + errs;
      }
      continue;
    }

    if (parsed.type === "assistant" && isRecord(parsed.message) && isRecord(parsed.message.usage)) {
      const model = typeof parsed.message.model === "string" ? parsed.message.model.trim() : "";
      if (!model || model === "<synthetic>") continue;

      const usage = parsed.message.usage;
      const inputTokens = sanitizeTokenCount(usage.input_tokens);
      const cacheReadTokens = sanitizeTokenCount(usage.cache_read_input_tokens);
      const cacheWriteTokens = sanitizeTokenCount(usage.cache_creation_input_tokens);
      const outputTokens = sanitizeTokenCount(usage.output_tokens);
      if (inputTokens + cacheReadTokens + cacheWriteTokens + outputTokens === 0) continue;

      let toolUses = 0;
      if (Array.isArray(parsed.message.content)) {
        for (const block of parsed.message.content) {
          if (isRecord(block) && block.type === "tool_use") {
            toolUses += 1;
          }
        }
      }

      turns.push({
        actualModel: model,
        inputTokens,
        cacheReadTokens,
        cacheWriteTokens,
        outputTokens,
        toolUses,
        toolErrors: 0,
      });
    }
  }

  return turns;
}

function indexClaudeSessionFiles(claudeProjectsDir: string): Map<string, string> {
  const index = new Map<string, string>();
  if (!existsSync(claudeProjectsDir)) return index;

  let projectDirs: string[];
  try {
    projectDirs = readdirSync(claudeProjectsDir);
  } catch {
    return index;
  }

  for (const dirName of projectDirs) {
    if (dirName.includes("homunculus")) continue;
    const projectPath = join(claudeProjectsDir, dirName);
    let isDir = false;
    try {
      const st = lstatSync(projectPath);
      if (st.isSymbolicLink()) continue;
      isDir = st.isDirectory();
    } catch {
      continue;
    }
    if (!isDir) continue;

    let entries: string[];
    try {
      entries = readdirSync(projectPath);
    } catch {
      continue;
    }

    for (const entry of entries) {
      const entryPath = join(projectPath, entry);
      let entryStat;
      try {
        entryStat = lstatSync(entryPath);
      } catch {
        continue;
      }
      if (entryStat.isSymbolicLink()) continue;

      if (entry.endsWith(".jsonl")) {
        if (!entryStat.isFile()) continue;
        const sessionId = basename(entry, ".jsonl");
        if (!index.has(sessionId)) {
          index.set(sessionId, entryPath);
        }
        continue;
      }

      if (!entryStat.isDirectory()) continue;

      const subagentsDir = join(entryPath, "subagents");
      try {
        const subDirStat = lstatSync(subagentsDir);
        if (subDirStat.isSymbolicLink() || !subDirStat.isDirectory()) continue;
        for (const subFile of readdirSync(subagentsDir)) {
          if (!subFile.endsWith(".jsonl")) continue;
          const subFilePath = join(subagentsDir, subFile);
          let subFileStat;
          try {
            subFileStat = lstatSync(subFilePath);
          } catch {
            continue;
          }
          if (subFileStat.isSymbolicLink() || !subFileStat.isFile()) continue;
          const subId = basename(subFile, ".jsonl");
          if (!index.has(subId)) {
            index.set(subId, subFilePath);
          }
        }
      } catch {
        // Ignore missing or unreadable subagent folders
      }
    }
  }

  return index;
}

function synthesizeTurns(
  turnCount: number,
  isSubagent: boolean,
  hasTools: boolean,
): ReplaySessionInput["turns"] {
  const count = Number.isFinite(turnCount)
    ? Math.min(MAX_SYNTHESIZED_TURNS, Math.max(1, Math.floor(turnCount)))
    : 1;
  const actualModel = isSubagent ? "claude-sonnet-5-5" : "claude-opus-5-5";
  const turns: ReplaySessionInput["turns"] = [];

  for (let i = 0; i < count; i++) {
    if (i === 0) {
      turns.push({
        actualModel,
        inputTokens: 1_200,
        cacheReadTokens: 0,
        cacheWriteTokens: isSubagent ? 10_000 : 18_000,
        outputTokens: 650,
        toolUses: hasTools ? 2 : 0,
        toolErrors: 0,
      });
    } else {
      const cachedContext = Math.min(160_000, (isSubagent ? 10_000 : 18_000) + i * 2_200);
      turns.push({
        actualModel,
        inputTokens: 450,
        cacheReadTokens: cachedContext,
        cacheWriteTokens: 2_200,
        outputTokens: 550,
        toolUses: hasTools ? (i % 3 === 0 ? 2 : 1) : 0,
        toolErrors: hasTools && i % 7 === 0 ? 1 : 0,
      });
    }
  }

  return turns;
}

function toSystemOneSignals(
  raw: RawSignalRecord | undefined,
  backendKey: BackendKey,
): SystemOneSignals {
  const meta = BACKEND_METADATA[backendKey];
  const domain =
    typeof raw?.domain === "string" && raw.domain.trim().length > 0
      ? raw.domain.trim()
      : "software_engineering";

  const probs: Record<string, number> = {};
  if (raw?.probs && isRecord(raw.probs)) {
    for (const [k, v] of Object.entries(raw.probs)) {
      if (typeof v === "number" && Number.isFinite(v)) {
        probs[k] = clamp01(v);
      }
    }
  }
  if (Object.keys(probs).length === 0) {
    probs[domain] = 1.0;
  }

  const rawComp = typeof raw?.comp === "number" && Number.isFinite(raw.comp) ? raw.comp : 0.5;
  const calibratedComp = clamp01(rawComp + meta.complexityOffset);
  const conf =
    typeof raw?.conf === "number" && Number.isFinite(raw.conf)
      ? clamp01(raw.conf)
      : clamp01(probs[domain] ?? 0.5);
  const noul = typeof raw?.noul === "number" && Number.isFinite(raw.noul) ? clamp01(raw.noul) : 0.2;
  const latencyMs =
    typeof raw?.latencyMs === "number" && Number.isFinite(raw.latencyMs) ? raw.latencyMs : 0;

  return {
    domain,
    domainProbabilities: probs,
    domainConfidence: conf,
    complexity: calibratedComp,
    needsFlagshipOrchestrator: noul,
    backend: meta.name,
    latencyMs,
    fromFallback: raw?.ok === false,
  };
}

/**
 * Load replay sessions from a saved System-1 signals benchmark JSON file,
 * pairing each session with real Claude Code `.jsonl` turn telemetry when `claudeProjectsDir`
 * is provided (or synthesizing realistic multi-turn cache-heavy turns otherwise).
 */
export function loadReplaySessionsFromSignalsFile(
  signalsPath: string,
  backendKey: BackendKey = "clef27b",
  claudeProjectsDir?: string,
): ReplaySessionInput[] {
  const rawText = readFileSync(signalsPath, "utf8");
  const parsed = JSON.parse(rawText) as RawSignalsFile;
  if (!parsed || !Array.isArray(parsed.sessions)) {
    throw new Error(`Invalid signals file format at ${signalsPath}: expected { sessions: [...] }`);
  }

  const sessionFileIndex = claudeProjectsDir
    ? indexClaudeSessionFiles(claudeProjectsDir)
    : new Map<string, string>();

  const result: ReplaySessionInput[] = [];
  for (let i = 0; i < parsed.sessions.length; i++) {
    const rec = parsed.sessions[i];
    const id = typeof rec.id === "string" && rec.id.trim().length > 0 ? rec.id.trim() : `session-${i}`;
    const isSubagent = Boolean(rec.isSubagent);
    const hasTools = rec.hasTools !== undefined ? Boolean(rec.hasTools) : true;
    const rawSignal = rec[backendKey];
    const signals = toSystemOneSignals(rawSignal, backendKey);

    const filePath = sessionFileIndex.get(id);
    const parsedTurns = filePath ? parseClaudeJsonlTurns(filePath) : [];
    const turns =
      parsedTurns.length > 0
        ? parsedTurns
        : synthesizeTurns(typeof rec.turns === "number" ? rec.turns : 1, isSubagent, hasTools);

    result.push({
      id,
      ...(typeof rec.project === "string" ? { project: rec.project } : {}),
      isSubagent,
      hasTools: parsedTurns.length > 0 ? parsedTurns.some(t => (t.toolUses ?? 0) > 0) : hasTools,
      signals,
      turns,
    });
  }

  return result;
}

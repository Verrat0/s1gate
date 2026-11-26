import { readFileSync } from "node:fs";
import bundledDomains from "../../data/domains.json";
import bundledEvalPrompts from "../../data/eval-prompts.json";
import bundledSnapshot from "../../data/snapshot.json";
import { createS1Gate } from "../core/engine.js";
import type {
  BenchmarkSnapshot,
  BudgetProfile,
  ClassifierAdapter,
  DomainDefinition,
  ExecutionRole,
  S1GateDecision,
  S1GateRequest,
  SystemOneClientConfig,
} from "../types.js";

export type EvalComplexityBand = "low" | "medium" | "high";
export type EvalExpectedRole = "auxiliary" | "specialist" | "orchestrator";
export type EvalReportFormat = "table" | "markdown" | "json";

export interface EvalPromptItem {
  id: string;
  prompt: string;
  expectedDomain: string;
  expectedRole: EvalExpectedRole;
  expectedComplexityBand: EvalComplexityBand;
  hasTools?: boolean;
  maxTokens?: number;
}

export interface EvalOptions {
  datasetPath?: string;
  prompts?: readonly EvalPromptItem[];
  localOnly?: boolean;
  systemOne?: SystemOneClientConfig;
  classifier?: ClassifierAdapter;
  profile?: BudgetProfile;
  availableModels?: readonly string[];
  snapshot?: BenchmarkSnapshot;
  domains?: readonly DomainDefinition[];
}

export interface EvalItemResult {
  id: string;
  expectedDomain: string;
  predictedDomain: string | null;
  domainMatch: boolean | null;
  expectedRole: EvalExpectedRole;
  predictedRole: ExecutionRole;
  roleMatch: boolean;
  expectedComplexityBand: EvalComplexityBand;
  predictedComplexity: number | null;
  complexityMatch: boolean;
  action: S1GateDecision["action"];
  selectedModel: string;
  latencyMs: number;
  classifierCalled: boolean;
}

export interface EvalDomainBreakdown {
  domain: string;
  total: number;
  classified: number;
  auxiliaryBypassed: number;
  domainCorrect: number;
  domainAccuracyPct: number;
  roleCorrect: number;
  roleAgreementPct: number;
  complexityCorrect: number;
  complexityAgreementPct: number;
}

export interface EvalReport {
  mode: "local_only" | "systemone";
  backend: string;
  profile: BudgetProfile;
  totalPrompts: number;
  classifiedPrompts: number;
  auxiliaryBypassedPrompts: number;
  classifierCallRatePct: number;
  domainAccuracyPct: number;
  roleAgreementPct: number;
  complexityAgreementPct: number;
  latencyP50Ms: number;
  latencyP95Ms: number;
  estimatedDecisionCostPer1kUsd: number;
  perDomain: EvalDomainBreakdown[];
  items: EvalItemResult[];
}

const REMOTE_SYSTEMONE_COST_PER_1K_USD = 0.04;
const VALID_ROLES = new Set<EvalExpectedRole>(["auxiliary", "specialist", "orchestrator"]);
const VALID_BANDS = new Set<EvalComplexityBand>(["low", "medium", "high"]);

const SYNTHETIC_EVAL_TOOLS = [
  { type: "function", function: { name: "read_file" } },
  { type: "function", function: { name: "edit_file" } },
  { type: "function", function: { name: "inspect_workspace" } },
  { type: "function", function: { name: "task" } },
] as const;

function roundPct(value: number): number {
  return Math.round(value * 100) / 100;
}

function roundMs(value: number): number {
  return Math.round(value * 100) / 100;
}

function roundCost(value: number): number {
  return Math.round(value * 10_000) / 10_000;
}

function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * p) - 1));
  return roundMs(sorted[idx]);
}

function defaultSnapshotModelIds(snapshot: BenchmarkSnapshot): string[] {
  return Object.values(snapshot.models)
    .filter((m) => m.inputPricePerMTok > 0 && m.outputPricePerMTok > 0)
    .map((m) => m.id);
}

function validateEvalItem(raw: unknown, idx: number, source: string): EvalPromptItem {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error(`Invalid eval prompt at index ${idx} in ${source}: expected object`);
  }
  const rec = raw as Record<string, unknown>;
  const id = typeof rec.id === "string" && rec.id.trim().length > 0 ? rec.id.trim() : `eval-${idx + 1}`;
  const prompt = typeof rec.prompt === "string" ? rec.prompt.trim() : "";
  const expectedDomain = typeof rec.expectedDomain === "string" ? rec.expectedDomain.trim() : "";
  const expectedRole = rec.expectedRole as EvalExpectedRole;
  const expectedComplexityBand = rec.expectedComplexityBand as EvalComplexityBand;

  if (!prompt) {
    throw new Error(`Invalid eval prompt '${id}' in ${source}: missing non-empty 'prompt'`);
  }
  if (!expectedDomain) {
    throw new Error(`Invalid eval prompt '${id}' in ${source}: missing 'expectedDomain'`);
  }
  if (!VALID_ROLES.has(expectedRole)) {
    throw new Error(
      `Invalid eval prompt '${id}' in ${source}: expectedRole must be auxiliary, specialist, or orchestrator`,
    );
  }
  if (!VALID_BANDS.has(expectedComplexityBand)) {
    throw new Error(
      `Invalid eval prompt '${id}' in ${source}: expectedComplexityBand must be low, medium, or high`,
    );
  }

  return {
    id,
    prompt,
    expectedDomain,
    expectedRole,
    expectedComplexityBand,
    ...(typeof rec.hasTools === "boolean" ? { hasTools: rec.hasTools } : {}),
    ...(typeof rec.maxTokens === "number" && Number.isFinite(rec.maxTokens) && rec.maxTokens > 0
      ? { maxTokens: rec.maxTokens }
      : {}),
  };
}

export function loadEvalDataset(datasetPath?: string): EvalPromptItem[] {
  if (!datasetPath) {
    return (bundledEvalPrompts as unknown[]).map((item, idx) =>
      validateEvalItem(item, idx, "data/eval-prompts.json"),
    );
  }

  const parsed = JSON.parse(readFileSync(datasetPath, "utf8")) as unknown;
  const rawList = Array.isArray(parsed)
    ? parsed
    : typeof parsed === "object" &&
        parsed !== null &&
        Array.isArray((parsed as { prompts?: unknown }).prompts)
      ? (parsed as { prompts: unknown[] }).prompts
      : null;

  if (!rawList) {
    throw new Error(
      `Invalid eval dataset format in ${datasetPath}: expected an array or { prompts: [...] }`,
    );
  }

  return rawList.map((item, idx) => validateEvalItem(item, idx, datasetPath));
}

export function matchesComplexityBand(complexity: number, band: EvalComplexityBand): boolean {
  if (!Number.isFinite(complexity)) return false;
  switch (band) {
    case "low":
      return complexity <= 0.35;
    case "medium":
      return complexity >= 0.25 && complexity <= 0.75;
    case "high":
      return complexity >= 0.55;
  }
}

export function matchesExpectedRole(
  predicted: ExecutionRole,
  expected: EvalExpectedRole,
): boolean {
  if (predicted === expected) return true;
  // Subagent is a scoped specialist role; auxiliary and orchestrator remain distinct.
  if (expected === "specialist" && predicted === "subagent") return true;
  return false;
}

export async function runPublicEval(options: EvalOptions = {}): Promise<EvalReport> {
  const prompts = options.prompts ? [...options.prompts] : loadEvalDataset(options.datasetPath);
  const snapshot: BenchmarkSnapshot =
    options.snapshot ?? (bundledSnapshot as unknown as BenchmarkSnapshot);
  const domains: DomainDefinition[] = options.domains
    ? [...options.domains]
    : (bundledDomains as DomainDefinition[]);
  const profile: BudgetProfile = options.profile ?? "balanced";
  const availableModels =
    options.availableModels && options.availableModels.length > 0
      ? [...options.availableModels]
      : defaultSnapshotModelIds(snapshot);

  const localOnly = options.localOnly ?? (!options.systemOne && !options.classifier);
  const mode: "local_only" | "systemone" = localOnly ? "local_only" : "systemone";

  const engine = createS1Gate({
    snapshot,
    domains,
    defaultProfile: profile,
    privacyMode: localOnly ? "local_only" : "standard",
    ...(!localOnly && options.classifier ? { classifier: options.classifier } : {}),
    ...(!localOnly && !options.classifier && options.systemOne
      ? { systemOne: options.systemOne }
      : {}),
  });

  const items: EvalItemResult[] = [];
  const latencies: number[] = [];
  let domainEvaluated = 0;
  let domainCorrect = 0;
  let roleCorrect = 0;
  let complexityCorrect = 0;
  let classifiedPrompts = 0;
  let auxiliaryBypassedPrompts = 0;

  interface DomainAccumulator {
    total: number;
    classified: number;
    auxiliaryBypassed: number;
    domainEvaluated: number;
    domainCorrect: number;
    roleCorrect: number;
    complexityCorrect: number;
  }

  const byDomain = new Map<string, DomainAccumulator>();
  const getBucket = (domainId: string): DomainAccumulator => {
    let bucket = byDomain.get(domainId);
    if (!bucket) {
      bucket = {
        total: 0,
        classified: 0,
        auxiliaryBypassed: 0,
        domainEvaluated: 0,
        domainCorrect: 0,
        roleCorrect: 0,
        complexityCorrect: 0,
      };
      byDomain.set(domainId, bucket);
    }
    return bucket;
  };

  for (const item of prompts) {
    const body: Record<string, unknown> = {
      model: "auto",
      messages: [{ role: "user", content: item.prompt }],
    };
    if (item.hasTools) {
      body.tools = SYNTHETIC_EVAL_TOOLS;
    }
    if (typeof item.maxTokens === "number" && Number.isFinite(item.maxTokens) && item.maxTokens > 0) {
      body.max_tokens = item.maxTokens;
    }

    const req: S1GateRequest = {
      protocol: "openai_chat",
      harness: "generic",
      headers: { "x-session-id": `eval-${item.id}` },
      body,
      availableModels,
      profile,
    };

    const decision = await engine.route(req);
    const latencyMs = roundMs(decision.latencyMs.total);
    latencies.push(latencyMs);

    const classifierCalled = decision.signals !== undefined;
    const isAuxBypass = decision.action === "auxiliary";
    const predictedDomain = decision.signals?.domain ?? null;
    const predictedComplexity = decision.signals ? roundMs(decision.signals.complexity) : null;

    let domainMatch: boolean | null;
    if (decision.signals) {
      domainMatch = decision.signals.domain === item.expectedDomain;
    } else if (isAuxBypass && item.expectedRole === "auxiliary") {
      // Expected 0ms auxiliary bypass intentionally skips domain classification.
      domainMatch = null;
    } else {
      domainMatch = false;
    }

    const roleMatch = matchesExpectedRole(decision.role, item.expectedRole);
    const complexityMatch = decision.signals
      ? matchesComplexityBand(decision.signals.complexity, item.expectedComplexityBand)
      : isAuxBypass
        ? item.expectedComplexityBand === "low"
        : false;

    const bucket = getBucket(item.expectedDomain);
    bucket.total += 1;
    if (classifierCalled) {
      classifiedPrompts += 1;
      bucket.classified += 1;
    }
    if (isAuxBypass) {
      auxiliaryBypassedPrompts += 1;
      bucket.auxiliaryBypassed += 1;
    }
    if (domainMatch !== null) {
      domainEvaluated += 1;
      bucket.domainEvaluated += 1;
      if (domainMatch) {
        domainCorrect += 1;
        bucket.domainCorrect += 1;
      }
    }
    if (roleMatch) {
      roleCorrect += 1;
      bucket.roleCorrect += 1;
    }
    if (complexityMatch) {
      complexityCorrect += 1;
      bucket.complexityCorrect += 1;
    }

    items.push({
      id: item.id,
      expectedDomain: item.expectedDomain,
      predictedDomain,
      domainMatch,
      expectedRole: item.expectedRole,
      predictedRole: decision.role,
      roleMatch,
      expectedComplexityBand: item.expectedComplexityBand,
      predictedComplexity,
      complexityMatch,
      action: decision.action,
      selectedModel: decision.selectedModel,
      latencyMs,
      classifierCalled,
    });
  }

  const totalPrompts = prompts.length;
  const stats = engine.getStats();
  const classifierCallRate = totalPrompts > 0 ? stats.classifierCalls / totalPrompts : 0;
  const estimatedDecisionCostPer1kUsd =
    mode === "local_only" ? 0 : roundCost(REMOTE_SYSTEMONE_COST_PER_1K_USD * classifierCallRate);

  const orderedDomainIds = [
    ...domains.map((d) => d.id).filter((id) => byDomain.has(id)),
    ...[...byDomain.keys()]
      .filter((id) => !domains.some((d) => d.id === id))
      .sort((a, b) => a.localeCompare(b)),
  ];

  const perDomain: EvalDomainBreakdown[] = orderedDomainIds.map((domainId) => {
    const b = byDomain.get(domainId)!;
    return {
      domain: domainId,
      total: b.total,
      classified: b.classified,
      auxiliaryBypassed: b.auxiliaryBypassed,
      domainCorrect: b.domainCorrect,
      domainAccuracyPct:
        b.domainEvaluated > 0 ? roundPct((b.domainCorrect / b.domainEvaluated) * 100) : 100,
      roleCorrect: b.roleCorrect,
      roleAgreementPct: b.total > 0 ? roundPct((b.roleCorrect / b.total) * 100) : 0,
      complexityCorrect: b.complexityCorrect,
      complexityAgreementPct: b.total > 0 ? roundPct((b.complexityCorrect / b.total) * 100) : 0,
    };
  });

  const backend = localOnly
    ? "heuristic-fallback"
    : options.classifier?.name ?? options.systemOne?.model ?? "systemone";

  return {
    mode,
    backend,
    profile,
    totalPrompts,
    classifiedPrompts,
    auxiliaryBypassedPrompts,
    classifierCallRatePct: roundPct(classifierCallRate * 100),
    domainAccuracyPct:
      domainEvaluated > 0 ? roundPct((domainCorrect / domainEvaluated) * 100) : 0,
    roleAgreementPct: totalPrompts > 0 ? roundPct((roleCorrect / totalPrompts) * 100) : 0,
    complexityAgreementPct:
      totalPrompts > 0 ? roundPct((complexityCorrect / totalPrompts) * 100) : 0,
    latencyP50Ms: percentile(latencies, 0.5),
    latencyP95Ms: percentile(latencies, 0.95),
    estimatedDecisionCostPer1kUsd,
    perDomain,
    items,
  };
}

export function formatEvalReport(
  result: EvalReport,
  format: EvalReportFormat = "table",
): string {
  if (format === "json") {
    return `${JSON.stringify(result, null, 2)}\n`;
  }

  if (format === "markdown") {
    const lines: string[] = [];
    lines.push("## s1gate Public Evaluation Report");
    lines.push("");
    lines.push(
      `- **Mode:** \`${result.mode}\` (\`${result.backend}\`) | **Profile:** \`${result.profile}\``,
    );
    lines.push(
      `- **Prompts:** ${result.totalPrompts} (${result.classifiedPrompts} classified, ${result.auxiliaryBypassedPrompts} auxiliary fast-path bypass)`,
    );
    lines.push(`- **Domain Accuracy (top-1):** ${result.domainAccuracyPct.toFixed(1)}%`);
    lines.push(`- **Role Agreement:** ${result.roleAgreementPct.toFixed(1)}%`);
    lines.push(`- **Complexity Band Agreement:** ${result.complexityAgreementPct.toFixed(1)}%`);
    lines.push(
      `- **Decision Latency:** p50 ${result.latencyP50Ms.toFixed(2)} ms | p95 ${result.latencyP95Ms.toFixed(2)} ms`,
    );
    lines.push(
      `- **Est. Decision Cost / 1k:** $${result.estimatedDecisionCostPer1kUsd.toFixed(4)} (classifier call rate ${result.classifierCallRatePct.toFixed(1)}%)`,
    );
    lines.push("");
    lines.push("| Domain | Total | Classified | Aux Bypass | Domain Acc | Role Agr | Complexity Agr |");
    lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
    for (const d of result.perDomain) {
      lines.push(
        `| \`${d.domain}\` | ${d.total} | ${d.classified} | ${d.auxiliaryBypassed} | ${d.domainAccuracyPct.toFixed(1)}% | ${d.roleAgreementPct.toFixed(1)}% | ${d.complexityAgreementPct.toFixed(1)}% |`,
      );
    }
    return `${lines.join("\n")}\n`;
  }

  const lines: string[] = [];
  lines.push(
    `s1gate Public Eval | mode=${result.mode} (${result.backend}) | profile=${result.profile} | ${result.totalPrompts} prompts`,
  );
  lines.push(
    `Domain Acc: ${result.domainAccuracyPct.toFixed(1)}% | Role Agr: ${result.roleAgreementPct.toFixed(1)}% | Complexity Agr: ${result.complexityAgreementPct.toFixed(1)}%`,
  );
  lines.push(
    `Latency: p50=${result.latencyP50Ms.toFixed(2)}ms p95=${result.latencyP95Ms.toFixed(2)}ms | Classifier Rate: ${result.classifierCallRatePct.toFixed(1)}% | Est. Cost/1k: $${result.estimatedDecisionCostPer1kUsd.toFixed(4)}`,
  );
  lines.push("");

  const header = `${"Domain".padEnd(23)} | ${"Tot".padStart(3)} | ${"Clf".padStart(3)} | ${"Aux".padStart(3)} | ${"DomAcc".padStart(7)} | ${"RoleAgr".padStart(7)} | ${"CompAgr".padStart(7)}`;
  lines.push(header);
  lines.push("-".repeat(header.length));

  for (const d of result.perDomain) {
    lines.push(
      `${d.domain.padEnd(23)} | ${String(d.total).padStart(3)} | ${String(d.classified).padStart(3)} | ${String(d.auxiliaryBypassed).padStart(3)} | ${`${d.domainAccuracyPct.toFixed(1)}%`.padStart(7)} | ${`${d.roleAgreementPct.toFixed(1)}%`.padStart(7)} | ${`${d.complexityAgreementPct.toFixed(1)}%`.padStart(7)}`,
    );
  }

  return `${lines.join("\n")}\n`;
}

#!/usr/bin/env node
import { readFileSync } from "node:fs";
import bundledDomains from "../../data/domains.json";
import bundledSnapshot from "../../data/snapshot.json";
import { buildCandidatePool } from "../core/catalog.js";
import { createS1Gate } from "../core/engine.js";
import { scoreAndSelectModel } from "../core/scorer.js";
import {
  loadReplaySessionsFromSignalsFile,
  runCounterfactualReplay,
  type ReplayReport,
} from "../replay/replay.js";
import { createHttpBridge } from "../server/http-bridge.js";
import type {
  BenchmarkSnapshot,
  BudgetProfile,
  DomainDefinition,
  ExecutionRole,
  HarnessId,
  S1GateConfig,
  SystemOneSignals,
} from "../types.js";

const VERSION = "0.1.0";

const VALID_PROFILES = new Set<BudgetProfile>(["frugal", "balanced", "quality"]);
const VALID_ROLES = new Set<ExecutionRole>([
  "orchestrator",
  "subagent",
  "specialist",
  "auxiliary",
]);
const VALID_BACKENDS = new Set<"clef27b" | "tev1_4b" | "tev1_08b">([
  "clef27b",
  "tev1_4b",
  "tev1_08b",
]);
const VALID_HARNESSES = new Set<HarnessId>([
  "claude_code",
  "codex",
  "aider",
  "opencode",
  "cursor",
  "generic",
]);
const BOOLEAN_FLAGS = new Set(["local-only", "json", "no-tools", "help", "version", "h", "v"]);

interface ParsedArgs {
  command: string | null;
  flags: Record<string, string | boolean>;
  positionals: string[];
}

function parseArgv(argv: readonly string[]): ParsedArgs {
  const flags: Record<string, string | boolean> = {};
  const positionals: string[] = [];

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--") {
      positionals.push(...argv.slice(i + 1));
      break;
    }
    if (arg.startsWith("--")) {
      const eqIdx = arg.indexOf("=");
      if (eqIdx !== -1) {
        const key = arg.slice(2, eqIdx);
        const val = arg.slice(eqIdx + 1);
        flags[key] = val;
      } else {
        const key = arg.slice(2);
        if (BOOLEAN_FLAGS.has(key)) {
          flags[key] = true;
        } else {
          const next = argv[i + 1];
          if (next !== undefined && !next.startsWith("-")) {
            flags[key] = next;
            i += 1;
          } else {
            flags[key] = true;
          }
        }
      }
      continue;
    }
    if (arg === "-h") {
      flags.help = true;
      continue;
    }
    if (arg === "-v") {
      flags.version = true;
      continue;
    }
    positionals.push(arg);
  }

  return {
    command: positionals[0] ?? null,
    flags,
    positionals: positionals.slice(1),
  };
}

function getStringFlag(flags: Record<string, string | boolean>, key: string): string | undefined {
  const val = flags[key];
  return typeof val === "string" && val.trim().length > 0 ? val.trim() : undefined;
}

function getBoolFlag(flags: Record<string, string | boolean>, key: string): boolean {
  const val = flags[key];
  if (typeof val === "boolean") return val;
  if (typeof val === "string") {
    return val.toLowerCase() !== "false" && val !== "0";
  }
  return false;
}

function parseProfile(raw: string | undefined, fallback: BudgetProfile): BudgetProfile {
  if (!raw) return fallback;
  const lower = raw.toLowerCase() as BudgetProfile;
  if (!VALID_PROFILES.has(lower)) {
    throw new Error(`Invalid profile '${raw}'. Expected one of: frugal, balanced, quality`);
  }
  return lower;
}

function parseRole(raw: string | undefined, fallback: ExecutionRole): ExecutionRole {
  if (!raw) return fallback;
  const lower = raw.toLowerCase() as ExecutionRole;
  if (!VALID_ROLES.has(lower)) {
    throw new Error(`Invalid role '${raw}'. Expected one of: orchestrator, subagent, specialist, auxiliary`);
  }
  return lower;
}

function parseHarness(raw: string | undefined, fallback: HarnessId): HarnessId {
  if (!raw) return fallback;
  const lower = raw.toLowerCase() as HarnessId;
  if (!VALID_HARNESSES.has(lower)) {
    throw new Error(`Invalid harness '${raw}'. Expected one of: claude_code, codex, aider, opencode, cursor, generic`);
  }
  return lower;
}

function parsePort(raw: string | undefined, fallback: number): number {
  if (!raw) return fallback;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 0 || n > 65535) {
    throw new Error(`Invalid port '${raw}'. Expected an integer between 0 and 65535`);
  }
  return n;
}

function loadAvailableModelsFromFile(filePath: string): string[] {
  const raw = JSON.parse(readFileSync(filePath, "utf8")) as unknown;
  if (Array.isArray(raw)) {
    return raw.filter((x): x is string => typeof x === "string" && x.trim().length > 0);
  }
  if (typeof raw === "object" && raw !== null) {
    const obj = raw as Record<string, unknown>;
    if (Array.isArray(obj.data)) {
      return obj.data
        .map(item =>
          typeof item === "string"
            ? item
            : typeof item === "object" && item !== null && typeof (item as { id?: unknown }).id === "string"
              ? (item as { id: string }).id
              : "",
        )
        .filter(id => id.trim().length > 0);
    }
    if (Array.isArray(obj.models)) {
      return obj.models
        .map(item =>
          typeof item === "string"
            ? item
            : typeof item === "object" && item !== null && typeof (item as { id?: unknown }).id === "string"
              ? (item as { id: string }).id
              : "",
        )
        .filter(id => id.trim().length > 0);
    }
  }
  throw new Error(`Unsupported models JSON format in ${filePath}`);
}

function defaultSnapshotModelIds(snapshot: BenchmarkSnapshot): string[] {
  return Object.values(snapshot.models)
    .filter(m => m.inputPricePerMTok > 0 && m.outputPricePerMTok > 0)
    .map(m => m.id);
}

function printHelp(): void {
  const text = `s1gate v${VERSION} — System-1 Cognitive Model Routing Engine

Usage:
  s1gate <command> [options]

Commands:
  serve     Start the zero-fork HTTP routing bridge (/v1/systemone, /v1/route, /health)
  replay    Run offline counterfactual cost & model-mix replay on saved System-1 signals
  inspect   Inspect Pareto benchmark scoring & top candidates for a domain

Serve Options:
  --port <num>               Listen port (env: S1GATE_PORT, default: 8787)
  --host <addr>              Bind address (env: S1GATE_HOST, default: 127.0.0.1)
  --profile <name>           Default profile: frugal | balanced | quality (default: balanced)
  --systemone-url <url>      System-1 endpoint URL (env: S1GATE_SYSTEMONE_URL)
  --systemone-model <model>  System-1 model ID (env: S1GATE_SYSTEMONE_MODEL, default: jev-latest)
  --api-key <key>            API key (env: S1GATE_API_KEY | TYPESAFE_API_KEY | JEV_API_KEY | OLLAMA_API_KEY)
  --local-only               Disable outbound System-1 calls and use local heuristic classifier

Replay Options:
  --signals <path>           Path to saved System-1 signals JSON file (required)
  --ocx-models <path>        Path to proxy models.json (defaults to bundled llm-stats.com catalog)
  --claude-dir <path>        Path to ~/.claude/projects for real turn-level token logs
  --backend <key>            Signal backend: clef27b | tev1_4b | tev1_08b (default: clef27b)
  --flagship <model>         Flagship baseline model ID (default: claude-opus-5-5)
  --json                     Output machine-readable JSON report

Inspect Options:
  --domain <id>              Domain ID (required; e.g. software_engineering, terminal_devops)
  --complexity <0..1>        Task complexity in [0, 1] (default: 0.5)
  --profile <name>           Budget profile: frugal | balanced | quality (default: balanced)
  --role <role>              Execution role: orchestrator | specialist | subagent | auxiliary (default: specialist)
  --models <m1,m2,...>       Comma-separated candidate model IDs (defaults to bundled catalog)
  --harness <id>             Coding harness: claude_code | codex | aider | opencode | cursor | generic
  --no-tools                 Evaluate without coding tool-use weighting
  --json                     Output machine-readable JSON result

General Options:
  -h, --help                 Show this help message
  -v, --version              Show version
`;
  process.stdout.write(text);
}

function formatUsd(amount: number): string {
  return `$${amount.toFixed(2)}`;
}

function formatPct(pct: number): string {
  const sign = pct > 0 ? "-" : pct < 0 ? "+" : "";
  return `${sign}${Math.abs(pct).toFixed(1)}%`;
}

function formatTopMix(mix: Record<string, number>, limit = 5): string {
  const entries = Object.entries(mix).slice(0, limit);
  if (entries.length === 0) return "(none)";
  return entries.map(([model, count]) => `${model} x${count}`).join(", ");
}

function formatReplayTable(
  report: ReplayReport,
  backend: string,
  modelCount: number,
): string {
  const lines: string[] = [];
  lines.push(
    `s1gate Counterfactual Replay | ${report.sessionsCount} sessions | ${report.totalTurns.toLocaleString()} turns | cache-read share: ${report.cacheReadSharePct.toFixed(1)}%`,
  );
  lines.push(`System-1 backend: ${backend} | Candidate pool: ${modelCount} models | Data: llm-stats.com`);
  lines.push("");

  const header = `${"Strategy".padEnd(24)} | ${"Total Cost".padStart(11)} | ${"vs Flagship".padStart(12)} | ${"vs Actual".padStart(10)}`;
  const sep = "-".repeat(header.length);
  lines.push(header);
  lines.push(sep);

  const vsFlag = (cost: number): string =>
    report.flagshipCostUsd > 0
      ? formatPct(((report.flagshipCostUsd - cost) / report.flagshipCostUsd) * 100)
      : "0.0%";
  const vsAct = (cost: number): string =>
    report.actualCostUsd > 0
      ? formatPct(((report.actualCostUsd - cost) / report.actualCostUsd) * 100)
      : "0.0%";

  const rows: Array<[string, number, string, string]> = [
    ["Always Flagship", report.flagshipCostUsd, "baseline", vsAct(report.flagshipCostUsd)],
    ["Actual (logged)", report.actualCostUsd, vsFlag(report.actualCostUsd), "baseline"],
    [
      "Naive Per-Turn (bust)",
      report.naivePerTurnCostUsd,
      vsFlag(report.naivePerTurnCostUsd),
      vsAct(report.naivePerTurnCostUsd),
    ],
    [
      "s1gate Quality",
      report.profiles.quality.totalCostUsd,
      formatPct(report.profiles.quality.savedVsFlagshipPct),
      formatPct(report.profiles.quality.savedVsActualPct),
    ],
    [
      "s1gate Balanced",
      report.profiles.balanced.totalCostUsd,
      formatPct(report.profiles.balanced.savedVsFlagshipPct),
      formatPct(report.profiles.balanced.savedVsActualPct),
    ],
    [
      "s1gate Frugal",
      report.profiles.frugal.totalCostUsd,
      formatPct(report.profiles.frugal.savedVsFlagshipPct),
      formatPct(report.profiles.frugal.savedVsActualPct),
    ],
  ];

  for (const [label, cost, flagDelta, actDelta] of rows) {
    lines.push(
      `${label.padEnd(24)} | ${formatUsd(cost).padStart(11)} | ${flagDelta.padStart(12)} | ${actDelta.padStart(10)}`,
    );
  }

  lines.push("");
  lines.push("Top Model Mix (sessions):");
  lines.push(`  Quality : ${formatTopMix(report.profiles.quality.modelMix)}`);
  lines.push(`  Balanced: ${formatTopMix(report.profiles.balanced.modelMix)}`);
  lines.push(`  Frugal  : ${formatTopMix(report.profiles.frugal.modelMix)}`);

  return `${lines.join("\n")}\n`;
}

function redactUrlForLog(rawUrl: string): string {
  try {
    const u = new URL(rawUrl);
    if (u.origin === "null") return "[redacted-url]";
    return `${u.origin}${u.pathname}`;
  } catch {
    return "[redacted-url]";
  }
}

async function runServe(flags: Record<string, string | boolean>): Promise<void> {
  const port = parsePort(getStringFlag(flags, "port") ?? process.env.S1GATE_PORT, 8787);
  const host = getStringFlag(flags, "host") ?? process.env.S1GATE_HOST ?? "127.0.0.1";
  const profile = parseProfile(
    getStringFlag(flags, "profile") ?? process.env.S1GATE_PROFILE,
    "balanced",
  );
  const localOnly = getBoolFlag(flags, "local-only");
  const systemOneUrl =
    getStringFlag(flags, "systemone-url") ?? process.env.S1GATE_SYSTEMONE_URL;
  const systemOneModel =
    getStringFlag(flags, "systemone-model") ??
    process.env.S1GATE_SYSTEMONE_MODEL ??
    "jev-latest";
  const apiKey =
    getStringFlag(flags, "api-key") ??
    process.env.S1GATE_API_KEY ??
    process.env.TYPESAFE_API_KEY ??
    process.env.JEV_API_KEY ??
    process.env.OLLAMA_API_KEY;

  const engineConfig: S1GateConfig = {
    defaultProfile: profile,
    privacyMode: localOnly ? "local_only" : "standard",
    ...(!localOnly && systemOneUrl
      ? {
          systemOne: {
            endpointUrl: systemOneUrl,
            model: systemOneModel,
            ...(apiKey ? { apiKey } : {}),
          },
        }
      : {}),
  };

  const engine = createS1Gate(engineConfig);
  const bridge = createHttpBridge(engine, {
    host,
    port,
    defaultProfile: profile,
  });

  const bound = await bridge.start();
  const classifierLabel = localOnly
    ? "local_only (heuristic)"
    : systemOneUrl
      ? `${systemOneModel} @ ${redactUrlForLog(systemOneUrl)}`
      : "heuristic-fallback (no --systemone-url configured)";

  process.stdout.write(
    [
      `s1gate v${VERSION} listening on ${bound.url}`,
      `  Profile   : ${profile}`,
      `  System-1  : ${classifierLabel}`,
      `  Endpoints : POST ${bound.url}/v1/systemone | POST ${bound.url}/v1/route | GET ${bound.url}/health`,
      `  Benchmarks: ${Object.keys(engine.snapshot.models).length} models (llm-stats.com)`,
      "",
    ].join("\n"),
  );

  const shutdown = async (): Promise<void> => {
    try {
      await bridge.close();
    } finally {
      process.exit(0);
    }
  };
  process.once("SIGINT", () => {
    void shutdown();
  });
  process.once("SIGTERM", () => {
    void shutdown();
  });
}

function runReplay(flags: Record<string, string | boolean>): void {
  const signalsPath = getStringFlag(flags, "signals");
  if (!signalsPath) {
    throw new Error("Missing required flag: --signals <path>");
  }

  const rawBackend = (getStringFlag(flags, "backend") ?? "clef27b") as
    | "clef27b"
    | "tev1_4b"
    | "tev1_08b";
  if (!VALID_BACKENDS.has(rawBackend)) {
    throw new Error(`Invalid backend '${rawBackend}'. Expected one of: clef27b, tev1_4b, tev1_08b`);
  }

  const snapshot = bundledSnapshot as unknown as BenchmarkSnapshot;
  const ocxModelsPath = getStringFlag(flags, "ocx-models");
  const claudeDir = getStringFlag(flags, "claude-dir");
  const flagshipModel = getStringFlag(flags, "flagship");
  const asJson = getBoolFlag(flags, "json");

  const availableModels = ocxModelsPath
    ? loadAvailableModelsFromFile(ocxModelsPath)
    : defaultSnapshotModelIds(snapshot);

  const sessions = loadReplaySessionsFromSignalsFile(signalsPath, rawBackend, claudeDir);
  const report = runCounterfactualReplay({
    sessions,
    availableModels,
    snapshot,
    flagshipModel,
  });

  if (asJson) {
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    return;
  }

  process.stdout.write(formatReplayTable(report, rawBackend, availableModels.length));
}

function runInspect(flags: Record<string, string | boolean>): void {
  const domainId = getStringFlag(flags, "domain");
  if (!domainId) {
    throw new Error("Missing required flag: --domain <id>");
  }

  const snapshot = bundledSnapshot as unknown as BenchmarkSnapshot;
  const domains = bundledDomains as DomainDefinition[];
  const domainExists = domains.some(d => d.id === domainId);
  if (!domainExists) {
    const validIds = domains.map(d => d.id).join(", ");
    throw new Error(`Unknown domain '${domainId}'. Valid domains: ${validIds}`);
  }

  const rawComplexity = getStringFlag(flags, "complexity");
  const complexity = rawComplexity !== undefined ? Number(rawComplexity) : 0.5;
  if (!Number.isFinite(complexity) || complexity < 0 || complexity > 1) {
    throw new Error(`Invalid complexity '${rawComplexity}'. Expected a number in [0, 1]`);
  }

  const profile = parseProfile(getStringFlag(flags, "profile"), "balanced");
  const role = parseRole(getStringFlag(flags, "role"), "specialist");
  const harness = parseHarness(getStringFlag(flags, "harness"), "claude_code");
  const hasTools = !getBoolFlag(flags, "no-tools");
  const asJson = getBoolFlag(flags, "json");

  const modelsArg = getStringFlag(flags, "models");
  const modelIds = modelsArg
    ? modelsArg.split(",").map(m => m.trim()).filter(Boolean)
    : defaultSnapshotModelIds(snapshot);

  const candidates = buildCandidatePool(modelIds, snapshot);
  const signals: SystemOneSignals = {
    domain: domainId,
    domainProbabilities: { [domainId]: 1.0 },
    domainConfidence: 1.0,
    complexity,
    needsFlagshipOrchestrator: role === "orchestrator" ? 0.9 : 0.1,
    backend: "cli-inspect",
    latencyMs: 0,
    fromFallback: false,
  };

  const result = scoreAndSelectModel({
    candidates,
    snapshot,
    domains,
    signals,
    profile,
    role,
    hasTools,
    requiresMultimodal: false,
    requiredContextTokens: 64_000,
    harness,
  });

  if (asJson) {
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    return;
  }

  const lines: string[] = [];
  lines.push(
    `s1gate Inspect | domain=${domainId} complexity=${complexity.toFixed(2)} profile=${profile} role=${role} tools=${hasTools}`,
  );
  lines.push(
    `Selected: ${result.selected.proxyId} | Q=${result.selected.qualityScore.toFixed(3)} (Qmax=${result.qMax.toFixed(3)}, floor=${result.qualityFloor.toFixed(3)}, deltaQ=${result.deltaQ.toFixed(3)}) | est. savings=${result.estimatedSavingsPct.toFixed(1)}% vs ${result.baselineModel}`,
  );
  lines.push("");

  const header = `${"#".padStart(2)} | ${"Model".padEnd(32)} | ${"Qual".padStart(6)} | ${"EffCost".padStart(8)} | ${"Ver".padStart(3)} | ${"Pen".padStart(5)} | Benchmark Refs`;
  lines.push(header);
  lines.push("-".repeat(header.length + 24));

  const top = result.rankedCandidates.slice(0, 10);
  for (let i = 0; i < top.length; i++) {
    const c = top[i];
    const refs = c.benchmarkRefs.slice(0, 3).join(", ");
    lines.push(
      `${String(i + 1).padStart(2)} | ${c.proxyId.slice(0, 32).padEnd(32)} | ${c.qualityScore.toFixed(3).padStart(6)} | $${c.effectiveCost.toFixed(3).padStart(7)} | ${(c.verified ? "yes" : "no").padStart(3)} | ${c.redFlagPenalty.toFixed(2).padStart(5)} | ${refs}`,
    );
  }
  lines.push("");
  lines.push("Benchmark data attribution: llm-stats.com (ZeroEval)");
  process.stdout.write(`${lines.join("\n")}\n`);
}

async function main(): Promise<void> {
  const parsed = parseArgv(process.argv.slice(2));

  if (parsed.flags.version && !parsed.command) {
    process.stdout.write(`s1gate ${VERSION}\n`);
    return;
  }

  if (parsed.flags.help || !parsed.command || parsed.command === "help") {
    printHelp();
    return;
  }

  switch (parsed.command) {
    case "serve":
      await runServe(parsed.flags);
      return;
    case "replay":
      runReplay(parsed.flags);
      return;
    case "inspect":
      runInspect(parsed.flags);
      return;
    default:
      throw new Error(`Unknown command '${parsed.command}'. Run 's1gate --help' for usage.`);
  }
}

void main().catch((err: unknown) => {
  const message = err instanceof Error ? err.message : String(err);
  process.stderr.write(`Error: ${message}\n`);
  process.exit(1);
});

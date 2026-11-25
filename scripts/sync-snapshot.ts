import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { BenchmarkSnapshot, BridgeStats, DomainDefinition, SnapshotModelEntry } from "../src/types.js";

const DEFAULT_OUT = resolve(import.meta.dirname, "../data/snapshot.json");
const DOMAINS_PATH = resolve(import.meta.dirname, "../data/domains.json");

const CORE_BENCH_IDS = [
  "swe-bench-verified",
  "swe-bench-pro",
  "swe-bench-multilingual",
  "deepswe-1.1",
  "ramp-swe-bench-v2",
  "ramp-swe-bench-v2-bash",
  "terminal-bench-4.0",
  "terminal-bench-2.1",
  "terminal-bench-science-0.1",
  "frontiercode-1.1",
  "frontiercode-2.0",
  "aider-polyglot",
  "livecodebench-v6",
  "program-bench",
  "agents-last-exam",
  "bfcl-v4",
  "automationbench",
  "osworld-2.0",
  "gpqa",
  "browsecomp",
  "frontiermath",
  "aime-2025",
  "healthbench-professional",
  "healthbench",
  "finance-agent-v2",
  "officeqa-pro",
  "legal-agent-benchmark",
  "writingbench",
  "simpleqa",
  "mmmu",
];

const normBenchKey = (k: string): string =>
  k.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

const logit = (p: number): number => {
  const c = Math.max(0.01, Math.min(0.99, p));
  return Math.log(c / (1 - c));
};

function computeBridge(models: SnapshotModelEntry[], bx: string, by: string): BridgeStats | null {
  const pairs: Array<[number, number]> = [];
  for (const m of models) {
    const x = m.scores[bx];
    const y = m.scores[by];
    if (typeof x === "number" && typeof y === "number") {
      pairs.push([logit(x), logit(y)]);
    }
  }
  if (pairs.length < 5) return null;
  const n = pairs.length;
  const muX = pairs.reduce((s, p) => s + p[0], 0) / n;
  const muY = pairs.reduce((s, p) => s + p[1], 0) / n;
  let vx = 0;
  let vy = 0;
  let cov = 0;
  for (const [x, y] of pairs) {
    vx += (x - muX) ** 2;
    vy += (y - muY) ** 2;
    cov += (x - muX) * (y - muY);
  }
  const sdX = Math.sqrt(vx / n) || 0.25;
  const sdY = Math.sqrt(vy / n) || 0.25;
  const r = Math.max(0, Math.min(0.99, cov / (Math.sqrt(vx * vy) || 1)));
  const rAdj = r * Math.sqrt(n / (n + 4));
  return {
    n,
    muX: Number(muX.toFixed(4)),
    sdX: Number(sdX.toFixed(4)),
    muY: Number(muY.toFixed(4)),
    sdY: Number(sdY.toFixed(4)),
    rAdj: Number(rAdj.toFixed(4)),
  };
}

async function fetchPaginatedBenchmark(id: string): Promise<Array<{ model_id: string; score: number }>> {
  const entries: Array<{ model_id: string; score: number }> = [];
  let offset = 0;
  while (true) {
    const r = await fetch(`https://api.zeroeval.com/leaderboard/benchmarks/${encodeURIComponent(id)}?offset=${offset}`);
    if (!r.ok) break;
    const j = (await r.json()) as { total_models?: number; entries?: Array<{ model_id: string; normalized_score?: number; benchmark_score?: number }> };
    const batch = j.entries ?? [];
    for (const e of batch) {
      const s = e.normalized_score ?? e.benchmark_score;
      if (typeof s === "number" && Number.isFinite(s)) entries.push({ model_id: e.model_id, score: s });
    }
    if (batch.length === 0 || entries.length >= (j.total_models ?? batch.length)) break;
    offset += batch.length;
    if (offset > 2000) break;
  }
  return entries;
}

function inferCachingAndThinking(id: string, org: string): Pick<
  SnapshotModelEntry,
  "cacheReadMultiplier" | "cacheWriteMultiplier" | "thinkingOutputRatio" | "requiresThinkingSignature"
> {
  const lower = `${id} ${org}`.toLowerCase();
  if (lower.includes("claude") || lower.includes("anthropic")) {
    return {
      cacheReadMultiplier: 0.1,
      cacheWriteMultiplier: 1.25,
      thinkingOutputRatio: /opus|fable|sonnet-5/.test(lower) ? 1.4 : 0.4,
      requiresThinkingSignature: true,
    };
  }
  if (lower.includes("gpt") || lower.includes("openai") || /\bo[134]\b/.test(lower)) {
    return {
      cacheReadMultiplier: 0.1,
      cacheWriteMultiplier: 1.0,
      thinkingOutputRatio: /astra|sol|o[134]/.test(lower) ? 1.5 : 0.3,
      requiresThinkingSignature: false,
    };
  }
  if (lower.includes("gemini") || lower.includes("google")) {
    return {
      cacheReadMultiplier: 0.1,
      cacheWriteMultiplier: 1.0,
      thinkingOutputRatio: /neon|hydrogen|pro/.test(lower) ? 1.2 : 0.3,
      requiresThinkingSignature: false,
    };
  }
  return {
    cacheReadMultiplier: 0.25,
    cacheWriteMultiplier: 1.0,
    thinkingOutputRatio: /r1|thinking|reason|pro/.test(lower) ? 1.3 : 0.2,
    requiresThinkingSignature: false,
  };
}

async function main() {
  const args = process.argv.slice(2);
  const fromDirIdx = args.indexOf("--from-dir");
  const fromDir = fromDirIdx >= 0 ? args[fromDirIdx + 1] : null;
  const outIdx = args.indexOf("--out");
  const outPath = outIdx >= 0 ? resolve(args[outIdx + 1]) : DEFAULT_OUT;

  let rawModels: Array<Record<string, unknown>> = [];
  const benchEntries: Record<string, Array<{ model_id: string; score: number }>> = {};
  let ocxDetailed: Record<string, { benchmarks?: Array<{ benchmark_id?: string; name?: string; score?: number; normalized_score?: number }> }> = {};

  if (fromDir) {
    rawModels = JSON.parse(readFileSync(resolve(fromDir, "models-full.json"), "utf8"));
    const bDoc = JSON.parse(readFileSync(resolve(fromDir, "benchmarks-entries.json"), "utf8"));
    for (const [bid, doc] of Object.entries<any>(bDoc.benchmarks || {})) {
      benchEntries[bid] = (doc.entries || [])
        .map((e: any) => ({ model_id: e.model_id, score: e.normalized_score ?? e.benchmark_score }))
        .filter((e: any) => typeof e.score === "number");
    }
    try {
      const detDoc = JSON.parse(readFileSync(resolve(fromDir, "ocx-models-detailed.json"), "utf8"));
      ocxDetailed = detDoc.models || {};
    } catch {
      /* optional */
    }
  } else {
    const res = await fetch("https://api.zeroeval.com/leaderboard/models/full?justCanonicals=true");
    if (!res.ok) throw new Error(`Failed to fetch models/full: HTTP ${res.status}`);
    rawModels = (await res.json()) as Array<Record<string, unknown>>;
    for (let i = 0; i < CORE_BENCH_IDS.length; i += 4) {
      const chunk = CORE_BENCH_IDS.slice(i, i + 4);
      const settled = await Promise.all(chunk.map(async id => [id, await fetchPaginatedBenchmark(id)] as const));
      for (const [id, entries] of settled) {
        if (entries.length > 0) benchEntries[id] = entries;
      }
    }
  }

  const modelsMap: Record<string, SnapshotModelEntry> = {};
  for (const m of rawModels) {
    const id = String(m.model_id || "").trim();
    if (!id) continue;
    const scores: Record<string, number> = {};
    for (const [k, v] of Object.entries(m)) {
      if (k.endsWith("_score") && typeof v === "number" && Number.isFinite(v)) {
        const norm = v > 1 && v <= 100 ? v / 100 : v;
        if (norm > 0 && norm <= 1) scores[normBenchKey(k.replace(/_score$/, ""))] = Number(norm.toFixed(4));
      }
    }
    const org = String(m.organization || "unknown");
    const extra = inferCachingAndThinking(id, org);
    modelsMap[id] = {
      id,
      name: String(m.name || id),
      organization: org,
      contextWindow: typeof m.context === "number" && m.context > 0 ? m.context : 128_000,
      inputPricePerMTok: typeof m.input_price === "number" && m.input_price > 0 ? m.input_price : 0,
      outputPricePerMTok: typeof m.output_price === "number" && m.output_price > 0 ? m.output_price : 0,
      multimodal: Boolean(m.multimodal),
      supportsTools: true,
      ...extra,
      scores,
    };
  }

  const lowerToKey = new Map(Object.keys(modelsMap).map(k => [k.toLowerCase(), k]));

  // Merge paginated benchmark entries
  for (const [bid, entries] of Object.entries(benchEntries)) {
    const bk = normBenchKey(bid);
    for (const e of entries) {
      const k = lowerToKey.get(e.model_id.toLowerCase());
      if (!k) continue;
      const norm = e.score > 1 && e.score <= 100 ? e.score / 100 : e.score;
      if (norm > 0 && norm <= 1) modelsMap[k].scores[bk] = Number(norm.toFixed(4));
    }
  }

  // Merge detailed per-model records (e.g. gemini-4-neon, gemini-4-hydrogen, gpt-6-sol)
  for (const [mid, detail] of Object.entries(ocxDetailed)) {
    const k = lowerToKey.get(mid.toLowerCase());
    if (!k) continue;
    for (const b of detail.benchmarks ?? []) {
      const bk = normBenchKey(b.benchmark_id || b.name || "");
      const raw = b.normalized_score ?? b.score;
      if (!bk || typeof raw !== "number" || !Number.isFinite(raw)) continue;
      const norm = raw > 1 && raw <= 100 ? raw / 100 : raw;
      if (norm > 0 && norm <= 1) modelsMap[k].scores[bk] = Number(norm.toFixed(4));
    }
  }

  // Keep only models that have at least one benchmark score OR price (keeps bundled snapshot compact)
  const keptModels: Record<string, SnapshotModelEntry> = {};
  for (const [id, entry] of Object.entries(modelsMap)) {
    if (Object.keys(entry.scores).length > 0 || entry.inputPricePerMTok > 0) {
      keptModels[id] = entry;
    }
  }

  // Precompute pairwise Logit-OLS bridges for every domain chain
  const domains: DomainDefinition[] = JSON.parse(readFileSync(DOMAINS_PATH, "utf8"));
  const allEntries = Object.values(keptModels);
  const bridges: Record<string, BridgeStats> = {};
  for (const d of domains) {
    const target = d.benchmarkChain[0];
    for (let i = 1; i < d.benchmarkChain.length; i++) {
      const fb = d.benchmarkChain[i];
      const key = `${target}|${fb}`;
      if (key in bridges) continue;
      const br = computeBridge(allEntries, target, fb);
      if (br) bridges[key] = br;
    }
  }

  const snapshot: BenchmarkSnapshot = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    source: "https://llm-stats.com",
    attribution: "Benchmark scores and model pricing sourced from llm-stats.com (ZeroEval).",
    models: keptModels,
    bridges,
  };

  writeFileSync(outPath, JSON.stringify(snapshot, null, 2));
  console.log(
    `Built snapshot: ${Object.keys(keptModels).length} models, ${Object.keys(bridges).length} Logit-OLS bridges -> ${outPath}`,
  );
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});

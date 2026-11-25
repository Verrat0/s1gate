import type {
  BackendCalibration,
  ClassifierAdapter,
  ClassifierRequest,
  DomainDefinition,
  SystemOneClientConfig,
  SystemOneSignals,
} from "../types.js";

export const BUILTIN_BACKEND_CALIBRATIONS: Record<string, BackendCalibration> = {
  "tev1:0.8b": { complexityOffset: -0.15, noulOrchestratorThreshold: 0.65, noulEscalateThreshold: 0.70 },
  "tev1:4b": { complexityOffset: 0.0, noulOrchestratorThreshold: 0.65, noulEscalateThreshold: 0.70 },
  "clef:27b": { complexityOffset: 0.0, noulOrchestratorThreshold: 0.65, noulEscalateThreshold: 0.70 },
  "jev-latest": { complexityOffset: 0.0, noulOrchestratorThreshold: 0.65, noulEscalateThreshold: 0.70 },
};

const COMPLEXITY_LEVELS = ["trivial", "simple", "moderate", "complex", "expert"] as const;
const MAX_CHOICE_OPTIONS = 24;
const MIN_CHOICE_OPTIONS = 2;
const DEFAULT_TIMEOUT_MS = 1500;
const DEFAULT_CB_THRESHOLD = 3;
const DEFAULT_CB_COOLDOWN_MS = 30_000;

const DOMAIN_KEYWORDS: Record<string, readonly string[]> = {
  software_engineering: [
    "refactor", "bug", "backend", "orm", "database", "sql", "migration", "api",
    "endpoint", "schema", "typescript", "python", "odoo", "laravel", "model",
    "controller", "transaction", "unit test", "vitest", "pytest", "class", "function",
  ],
  terminal_devops: [
    "docker", "bash", "shell", "linux", "systemd", "ci/cd", "github actions",
    "kubernetes", "nginx", "ssh", "chmod", "curl", "terminal", "cli", "container",
    "deploy", "pipeline", "cron", "firewall", "proxy", "compose",
  ],
  frontend_web: [
    "react", "vue", "next.js", "nextjs", "tailwind", "css", "html", "navbar",
    "component", "responsive", "dom", "browser", "layout", "ui", "ux", "scss",
    "animation", "modal", "button", "flexbox", "grid", "svelte", "astro",
  ],
  agentic_tools: [
    "mcp", "tool calling", "subagent", "workflow", "orchestrate", "autonomous",
    "multi-step", "agentic", "playwright", "browser_click", "automation",
  ],
  research_architecture: [
    "architecture", "trade-off", "tradeoff", "system design", "distributed",
    "sharding", "multi-tenant", "rfc", "scalability", "microservices", "research",
    "comparison", "blueprint", "migration plan", "consensus",
  ],
  math_science: [
    "algorithm", "theorem", "proof", "matrix", "eigenvalue", "calculus", "integral",
    "probability", "statistics", "regression", "optimization", "bayesian", "physics",
    "differential", "complexity o(", "linear algebra",
  ],
  healthcare: [
    "clinical", "medical", "hospital", "emr", "simrs", "patient", "triage",
    "icd-10", "icd10", "hl7", "fhir", "diagnosis", "bpjs", "pharmacy", "nursing",
    "radiology", "lab result", "rekam medis",
  ],
  finance_data: [
    "accounting", "invoice", "invoicing", "ledger", "tax", "billing", "journal",
    "balance sheet", "spreadsheet", "revenue", "payroll", "p&l", "financial",
    "reconciliation", "budget", "vat", "ppn",
  ],
  legal: [
    "contract", "clause", "compliance", "regulation", "gdpr", "hipaa", "liability",
    "indemnity", "policy", "terms of service", "legal", "statute", "nda", "license",
  ],
  writing_docs: [
    "documentation", "readme", "changelog", "commit message", "translate",
    "translation", "summarize", "summary", "blog", "article", "prose", "explain",
    "docstring", "release notes", "copywriting",
  ],
  general_chat: [
    "hello", "hi", "thanks", "thank you", "capital of", "what is", "who is",
    "quick question", "brainstorm", "joke", "weather",
  ],
};

const HIGH_COMPLEXITY_SIGNALS = [
  "architecture", "distributed", "concurrency", "deadlock", "race condition",
  "multi-tenant", "sharding", "zero-downtime", "migration", "refactor", "security",
  "cryptography", "protocol", "compiler", "transaction", "orchestrator", "benchmark",
];

const LOW_COMPLEXITY_SIGNALS = [
  "typo", "rename", "what is", "capital of", "hello", "hi,", "quick question",
  "simple", "trivial", "format", "one-liner",
];

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function softmax(scores: Record<string, number>): Record<string, number> {
  const entries = Object.entries(scores);
  if (entries.length === 0) return { general_chat: 1 };

  const maxScore = Math.max(...entries.map(([, v]) => v));
  const exps = entries.map(([k, v]) => [k, Math.exp(v - maxScore)] as const);
  const sum = exps.reduce((acc, [, v]) => acc + v, 0) || 1;

  const probs: Record<string, number> = {};
  for (const [k, v] of exps) {
    probs[k] = v / sum;
  }
  return probs;
}

function scoreDomainText(lowerText: string, domain: DomainDefinition): number {
  let score = 0;
  const keywords = DOMAIN_KEYWORDS[domain.id] ?? [];
  for (const kw of keywords) {
    if (lowerText.includes(kw)) score += 1.2;
  }

  const descTokens = domain.description
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((tok) => tok.length >= 4);
  for (const tok of descTokens) {
    if (lowerText.includes(tok)) score += 0.35;
  }

  return score;
}

function estimateHeuristicComplexity(text: string, estimatedTokens: number): number {
  const lower = text.toLowerCase();
  let score = 0.25;

  if (estimatedTokens > 120) score += 0.1;
  if (estimatedTokens > 400) score += 0.1;
  if (text.includes("[CODE:")) score += 0.12;
  if (text.includes("recent_error:")) score += 0.12;
  if (text.includes("stuck_loop:")) score += 0.18;

  const toolsMatch = /^tools:\s*(.+)$/m.exec(text);
  if (toolsMatch) {
    const toolCount = toolsMatch[1].split(",").filter(Boolean).length;
    if (toolCount >= 4) score += 0.1;
  }

  const turnsMatch = /^turns:\s*(\d+)/m.exec(text);
  if (turnsMatch && Number(turnsMatch[1]) >= 4) {
    score += 0.08;
  }

  for (const kw of HIGH_COMPLEXITY_SIGNALS) {
    if (lower.includes(kw)) score += 0.08;
  }
  for (const kw of LOW_COMPLEXITY_SIGNALS) {
    if (lower.includes(kw)) score -= 0.12;
  }

  return clamp01(score);
}

export class HeuristicFallbackClassifier implements ClassifierAdapter {
  readonly name = "heuristic-fallback";

  async classify(request: ClassifierRequest): Promise<SystemOneSignals> {
    const startMs = Date.now();
    const text = request.envelope.text;
    const lower = text.toLowerCase();

    const activeDomains: DomainDefinition[] =
      request.domains.length > 0
        ? request.domains
        : [{ id: "general_chat", description: "General chat", benchmarkChain: ["gpqa"] }];

    const rawScores: Record<string, number> = {};
    let bestDomain = activeDomains[0].id;
    let bestScore = -Infinity;

    for (const d of activeDomains) {
      const s = scoreDomainText(lower, d);
      rawScores[d.id] = s;
      if (s > bestScore) {
        bestScore = s;
        bestDomain = d.id;
      }
    }

    // If zero keywords matched, prefer general_chat or software_engineering when present
    if (bestScore === 0) {
      const fallbackId = activeDomains.find((d) => d.id === "general_chat")?.id ?? activeDomains[0].id;
      rawScores[fallbackId] = 0.5;
      bestDomain = fallbackId;
    }

    const domainProbabilities = softmax(rawScores);
    const complexity = estimateHeuristicComplexity(text, request.envelope.estimatedTokens);

    let needsFlagshipOrchestrator = 0;
    if (request.askOrchestrator) {
      let orch = complexity * 0.85;
      if (lower.includes("role: orchestrator")) orch += 0.1;
      if (bestDomain === "research_architecture" || bestDomain === "agentic_tools") orch += 0.1;
      if (text.includes("stuck_loop:")) orch += 0.15;
      needsFlagshipOrchestrator = clamp01(orch);
    }

    let shouldEscalate: number | undefined;
    if (request.askEscalate) {
      let esc = 0.15;
      if (text.includes("stuck_loop: [level=hard]")) esc += 0.7;
      else if (text.includes("stuck_loop: [level=soft]")) esc += 0.45;
      else if (text.includes("stuck_loop:")) esc += 0.5;
      if (text.includes("recent_error:")) esc += 0.2;
      shouldEscalate = clamp01(esc);
    }

    const signals: SystemOneSignals = {
      domain: bestDomain,
      domainProbabilities,
      domainConfidence: clamp01(domainProbabilities[bestDomain] ?? 0),
      complexity,
      needsFlagshipOrchestrator,
      ...(shouldEscalate !== undefined ? { shouldEscalate } : {}),
      backend: this.name,
      latencyMs: Math.max(0, Date.now() - startMs),
      fromFallback: true,
    };

    return signals;
  }
}

function resolveCalibration(model: string, override?: BackendCalibration): Required<BackendCalibration> {
  const exact = BUILTIN_BACKEND_CALIBRATIONS[model];
  const lower = model.toLowerCase();
  const matchedBuiltin =
    exact ??
    Object.entries(BUILTIN_BACKEND_CALIBRATIONS).find(([key]) =>
      lower === key.toLowerCase() || lower.endsWith(`/${key.toLowerCase()}`),
    )?.[1];

  return {
    complexityOffset: override?.complexityOffset ?? matchedBuiltin?.complexityOffset ?? 0.0,
    noulOrchestratorThreshold:
      override?.noulOrchestratorThreshold ?? matchedBuiltin?.noulOrchestratorThreshold ?? 0.65,
    noulEscalateThreshold:
      override?.noulEscalateThreshold ?? matchedBuiltin?.noulEscalateThreshold ?? 0.70,
  };
}

function buildDomainCriteria(domains: readonly DomainDefinition[]): Record<string, string> {
  const criteria: Record<string, string> = {};
  const capped = domains.slice(0, MAX_CHOICE_OPTIONS);

  for (const d of capped) {
    if (Object.keys(criteria).length >= MAX_CHOICE_OPTIONS) break;
    criteria[d.id] = String(d.description);
  }

  const fallbacks: readonly [string, string][] = [
    ["general_chat", "Simple Q&A, quick lookup, casual brainstorming, short clarifications"],
    ["software_engineering", "General software engineering, backend logic, refactoring, bug fixing"],
  ];

  for (const [id, desc] of fallbacks) {
    if (Object.keys(criteria).length >= MIN_CHOICE_OPTIONS) break;
    if (!(id in criteria)) criteria[id] = desc;
  }

  return criteria;
}

function buildSystemOnePayload(model: string, request: ClassifierRequest): Record<string, unknown> {
  const questions: Record<string, unknown> = {
    domain: {
      type: "choice",
      instructions: "Classify the primary technical or subject domain of this request.",
      criteria: buildDomainCriteria(request.domains),
    },
    complexity: {
      type: "score",
      instructions: "Rate the task complexity from trivial to expert.",
      criteria: [...COMPLEXITY_LEVELS],
    },
  };

  if (request.askOrchestrator) {
    questions.needs_flagship_orchestrator = {
      type: "noul",
      instructions:
        "Does this task require a flagship orchestrator model for multi-step planning, architecture, or complex coordination?",
      criteria: {
        true: "Multi-file architecture, deep trade-off planning, complex agentic orchestration, or high-risk debugging",
        false: "Single-step lookup, routine edit, simple Q&A, or straightforward implementation",
      },
    };
  }

  if (request.askEscalate) {
    questions.should_escalate = {
      type: "noul",
      instructions:
        "Is the agent stuck in an error loop or failing repeatedly such that escalating to a stronger model is needed?",
      criteria: {
        true: "Repeated errors, oscillation, stuck loop, or inability to make progress",
        false: "Making steady progress or error is minor and easily resolved",
      },
    };
  }

  return {
    model,
    state: request.envelope.text,
    questions,
  };
}

function parseNoulAnswer(raw: unknown): number | null {
  if (typeof raw === "number" && Number.isFinite(raw)) return clamp01(raw);
  const rec = asRecord(raw);
  if (!rec) return null;
  const val = rec.noul ?? rec.probability ?? rec.value;
  return typeof val === "number" && Number.isFinite(val) ? clamp01(val) : null;
}

function parseScoreAnswer(raw: unknown): number | null {
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return normalizeScoreValue(raw);
  }
  const rec = asRecord(raw);
  if (!rec) return null;

  const direct = rec.score ?? rec.value;
  if (typeof direct === "number" && Number.isFinite(direct)) {
    return normalizeScoreValue(direct);
  }

  if (Array.isArray(rec.probabilities) && rec.probabilities.length > 1) {
    const probs = rec.probabilities.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    if (probs.length === rec.probabilities.length) {
      const denom = probs.length - 1;
      const weighted = probs.reduce((acc, p, idx) => acc + p * (idx / denom), 0);
      return clamp01(weighted);
    }
  }

  return null;
}

function normalizeScoreValue(raw: number): number {
  if (raw <= 1) return clamp01(raw);
  const maxIdx = COMPLEXITY_LEVELS.length - 1;
  if (raw <= maxIdx) return clamp01(raw / maxIdx);
  return clamp01((raw - 1) / maxIdx);
}

function parseChoiceAnswer(
  raw: unknown,
): { domain: string; probabilities: Record<string, number>; confidence: number } | null {
  const rec = asRecord(raw);
  if (!rec) return null;

  const probs: Record<string, number> = {};
  const rawProbs = asRecord(rec.probabilities);
  if (rawProbs) {
    for (const [k, v] of Object.entries(rawProbs)) {
      if (typeof v === "number" && Number.isFinite(v)) {
        probs[k] = clamp01(v);
      }
    }
  }

  let domain =
    typeof rec.choice === "string" && rec.choice.trim()
      ? rec.choice.trim()
      : typeof rec.value === "string" && rec.value.trim()
        ? rec.value.trim()
        : "";

  if (!domain) {
    const top = Object.entries(probs).sort((a, b) => b[1] - a[1])[0];
    if (top) domain = top[0];
  }
  if (!domain) return null;

  const rawConf = typeof rec.confidence === "number" && Number.isFinite(rec.confidence)
    ? clamp01(rec.confidence)
    : clamp01(probs[domain] ?? 1);

  if (Object.keys(probs).length === 0) {
    probs[domain] = rawConf;
  }

  return { domain, probabilities: probs, confidence: rawConf };
}

function parseSystemOneResponse(
  payload: unknown,
  request: ClassifierRequest,
  model: string,
  calibration: Required<BackendCalibration>,
  latencyMs: number,
): SystemOneSignals | null {
  const top = asRecord(payload);
  if (!top) return null;

  const resultRec = asRecord(top.result);
  const root = resultRec?.answers && !top.answers ? resultRec : top;
  const answers = asRecord(root.answers);
  if (!answers) return null;

  const domainParsed = parseChoiceAnswer(answers.domain);
  const rawComplexity = parseScoreAnswer(answers.complexity);
  if (!domainParsed || rawComplexity === null) return null;

  let needsFlagshipOrchestrator = 0;
  if (request.askOrchestrator) {
    const parsedOrch = parseNoulAnswer(answers.needs_flagship_orchestrator);
    if (parsedOrch === null) return null;
    needsFlagshipOrchestrator = parsedOrch;
  }

  let shouldEscalate: number | undefined;
  if (request.askEscalate) {
    const parsedEsc = parseNoulAnswer(answers.should_escalate);
    if (parsedEsc === null) return null;
    shouldEscalate = parsedEsc;
  }

  const calibratedComplexity = clamp01(rawComplexity + calibration.complexityOffset);

  return {
    domain: domainParsed.domain,
    domainProbabilities: domainParsed.probabilities,
    domainConfidence: domainParsed.confidence,
    complexity: calibratedComplexity,
    needsFlagshipOrchestrator,
    ...(shouldEscalate !== undefined ? { shouldEscalate } : {}),
    backend: model,
    latencyMs,
    fromFallback: false,
  };
}

export class SystemOneClient implements ClassifierAdapter {
  readonly name: string;
  private readonly config: SystemOneClientConfig;
  private readonly calibration: Required<BackendCalibration>;
  private readonly threshold: number;
  private readonly cooldownMs: number;
  private readonly fetchImpl: typeof fetch;
  private readonly fallback = new HeuristicFallbackClassifier();
  private consecutiveFailures = 0;
  private openedAtMs: number | null = null;
  private probeInFlight = false;

  constructor(config: SystemOneClientConfig) {
    this.config = config;
    this.name = `systemone:${config.model}`;
    this.calibration = resolveCalibration(config.model, config.calibration);
    this.threshold = Math.max(1, config.circuitBreakerThreshold ?? DEFAULT_CB_THRESHOLD);
    this.cooldownMs = Math.max(0, config.circuitBreakerCooldownMs ?? DEFAULT_CB_COOLDOWN_MS);
    this.fetchImpl = config.fetchImpl ?? globalThis.fetch;
  }

  getCircuitState(): "closed" | "open" | "half_open" {
    if (this.openedAtMs === null) return "closed";
    return Date.now() - this.openedAtMs < this.cooldownMs ? "open" : "half_open";
  }

  private recordSuccess(): void {
    this.consecutiveFailures = 0;
    this.openedAtMs = null;
  }

  private recordFailure(wasHalfOpen: boolean): void {
    this.consecutiveFailures += 1;
    if (wasHalfOpen || this.consecutiveFailures >= this.threshold) {
      this.openedAtMs = Date.now();
    }
  }

  private async failOpen(
    request: ClassifierRequest,
    fallbackReason: string,
    startMs: number,
  ): Promise<SystemOneSignals> {
    const fallbackSignals = await this.fallback.classify(request);
    return {
      ...fallbackSignals,
      latencyMs: Math.max(0, Date.now() - startMs),
      fromFallback: true,
      fallbackReason,
    };
  }

  async classify(request: ClassifierRequest): Promise<SystemOneSignals> {
    const startMs = Date.now();
    const state = this.getCircuitState();

    if (state === "open") {
      return this.failOpen(request, "circuit_open", startMs);
    }

    const wasHalfOpen = state === "half_open";
    if (wasHalfOpen) {
      if (this.probeInFlight) {
        return this.failOpen(request, "circuit_open", startMs);
      }
      this.probeInFlight = true;
    }

    try {
      const timeoutMs =
        Number.isFinite(request.timeoutMs) && request.timeoutMs > 0
          ? request.timeoutMs
          : this.config.timeoutMs;
      const effectiveTimeout =
        typeof timeoutMs === "number" && Number.isFinite(timeoutMs) && timeoutMs > 0
          ? timeoutMs
          : DEFAULT_TIMEOUT_MS;
      const payload = buildSystemOnePayload(this.config.model, request);
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...(this.config.apiKey ? { Authorization: `Bearer ${this.config.apiKey}` } : {}),
        ...(this.config.headers ?? {}),
      };

      let response: Response;
      try {
        response = await this.fetchImpl(this.config.endpointUrl, {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(effectiveTimeout),
        });
      } catch (err: unknown) {
        this.recordFailure(wasHalfOpen);
        const isTimeout =
          err instanceof Error && (err.name === "TimeoutError" || err.name === "AbortError");
        return this.failOpen(request, isTimeout ? "timeout" : "network", startMs);
      }

      if (!response.ok) {
        void response.body?.cancel().catch(() => undefined);
        this.recordFailure(wasHalfOpen);
        return this.failOpen(request, `http_${response.status}`, startMs);
      }

      let json: unknown;
      try {
        json = await response.json();
      } catch {
        this.recordFailure(wasHalfOpen);
        return this.failOpen(request, "malformed", startMs);
      }

      const parsed = parseSystemOneResponse(
        json,
        request,
        this.config.model,
        this.calibration,
        Math.max(0, Date.now() - startMs),
      );
      if (!parsed) {
        this.recordFailure(wasHalfOpen);
        return this.failOpen(request, "malformed", startMs);
      }

      this.recordSuccess();
      return parsed;
    } finally {
      if (wasHalfOpen) {
        this.probeInFlight = false;
      }
    }
  }
}

export type InboundProtocol = "anthropic_messages" | "openai_responses" | "openai_chat";
export type BudgetProfile = "frugal" | "balanced" | "quality";
export type ExecutionRole = "orchestrator" | "subagent" | "specialist" | "auxiliary";
export type RoutingAction = "route" | "pin_reuse" | "escalate" | "auxiliary" | "compact" | "passthrough";
export type ReasoningEffort = "none" | "low" | "medium" | "high" | "max";
export type HarnessId = "claude_code" | "codex" | "aider" | "opencode" | "cursor" | "generic";
export type CompatTier = "native" | "translated_verified" | "degraded" | "unsupported";
export type PrivacyMode = "standard" | "local_only";

export interface DomainDefinition {
  id: string;
  description: string;
  benchmarkChain: string[];
  /** Sibling benchmarks that flag severe single-benchmark cherry-picking if much lower */
  redFlagSiblings?: string[];
}

export interface BridgeStats {
  n: number;
  muX: number;
  sdX: number;
  muY: number;
  sdY: number;
  rAdj: number;
}

export interface SnapshotModelEntry {
  id: string;
  name: string;
  organization: string;
  contextWindow: number;
  inputPricePerMTok: number;
  outputPricePerMTok: number;
  cacheReadMultiplier?: number;
  cacheWriteMultiplier?: number;
  thinkingOutputRatio?: number;
  multimodal: boolean;
  supportsTools?: boolean;
  requiresThinkingSignature?: boolean;
  scores: Record<string, number>;
}

export interface BenchmarkSnapshot {
  schemaVersion: 1;
  generatedAt: string;
  source: string;
  attribution: string;
  models: Record<string, SnapshotModelEntry>;
  /** Precomputed Logit-OLS overlap bridges keyed by `${targetBench}|${fallbackBench}` */
  bridges: Record<string, BridgeStats>;
}

export interface ModelOverride {
  canonicalId?: string;
  inputPricePerMTok?: number;
  outputPricePerMTok?: number;
  cacheReadMultiplier?: number;
  cacheWriteMultiplier?: number;
  proxyStripsCacheControl?: boolean;
  quotaWeightMultiplier?: number;
  harnessCompat?: Partial<Record<HarnessId, CompatTier>>;
  supportsTools?: boolean;
  contextWindow?: number;
}

export interface CandidateModel {
  /** Exact ID as provided by the host proxy (returned in S1GateDecision.selectedModel) */
  proxyId: string;
  /** Resolved canonical ID in llm-stats.com snapshot (or null if unmatched) */
  canonicalId: string | null;
  name: string;
  organization: string;
  contextWindow: number;
  inputPricePerMTok: number;
  outputPricePerMTok: number;
  cacheReadMultiplier: number;
  cacheWriteMultiplier: number;
  proxyStripsCacheControl: boolean;
  thinkingOutputRatio: number;
  quotaWeightMultiplier: number;
  multimodal: boolean;
  supportsTools: boolean;
  requiresThinkingSignature: boolean;
  harnessCompat: Partial<Record<HarnessId, CompatTier>>;
  scores: Record<string, number>;
}

export interface S1GateRequest {
  protocol: InboundProtocol;
  path?: string;
  headers: Record<string, string | undefined>;
  body: Record<string, unknown>;
  availableModels: string[];
  /** Optional real-time quota utilization in [0, 1) keyed by proxyId or provider prefix */
  quotaUtilization?: Record<string, number>;
  /** Optional caller-provided session token count when proxy trims body.messages or already computed tokens */
  estimatedContextTokens?: number;
  /** Explicit profile override (e.g. from 9Router combo UI) */
  profile?: BudgetProfile;
  /** Explicit role override */
  roleOverride?: ExecutionRole;
  /** Target coding harness for compatibility gating */
  harness?: HarnessId;
  /** Current timestamp override (for deterministic tests / replay) */
  nowMs?: number;
}

export interface SystemOneSignals {
  domain: string;
  domainProbabilities: Record<string, number>;
  domainConfidence: number;
  /** Normalized task complexity in [0, 1] */
  complexity: number;
  /** Raw noul probability [0, 1] that task needs a flagship orchestrator */
  needsFlagshipOrchestrator: number;
  /** Optional raw noul probability [0, 1] that a soft-stuck session should escalate */
  shouldEscalate?: number;
  backend: string;
  latencyMs: number;
  fromFallback: boolean;
  fallbackReason?: string;
}

export interface RoutingEnvelope {
  text: string;
  byteLength: number;
  estimatedTokens: number;
  redactionCounts: {
    codeBlocks: number;
    secrets: number;
    pii: number;
  };
}

export type StuckLevel = "none" | "soft" | "hard";

export interface StuckLoopReport {
  level: StuckLevel;
  reasons: string[];
  identicalErrorCount: number;
  identicalToolFailCount: number;
  revertOscillation: boolean;
  tddEvasion: boolean;
  apologyStreak: number;
  hotspotEdits: number;
  errorCountTrendingDown: boolean;
  telemetrySummary: string;
}

export interface SessionPin {
  sessionKey: string;
  model: string;
  role: ExecutionRole;
  profile: BudgetProfile;
  effort?: ReasoningEffort;
  domain: string;
  complexity: number;
  lastEstimatedTokens: number;
  turnCount: number;
  escalatedAtTurn: number | null;
  escalationCount: number;
  pinnedAtMs: number;
  lastUsedAtMs: number;
}

export interface ScoringCandidateSummary {
  proxyId: string;
  canonicalId: string | null;
  qualityScore: number;
  effectiveCost: number;
  verified: boolean;
  redFlagPenalty: number;
  benchmarkRefs: string[];
}

export interface ScoringResult {
  selected: ScoringCandidateSummary;
  rankedCandidates: ScoringCandidateSummary[];
  qMax: number;
  qualityFloor: number;
  deltaQ: number;
  baselineModel: string;
  baselineEffectiveCost: number;
  estimatedSavingsPct: number;
  excludedReasons: Record<string, string>;
}

export interface S1GateDecision {
  selectedModel: string;
  rankedFallbacks: string[];
  action: RoutingAction;
  role: ExecutionRole;
  profile: BudgetProfile;
  recommendedEffort?: ReasoningEffort;
  sessionKey: string | null;
  cachePinned: boolean;
  signals?: SystemOneSignals;
  stuckLoop?: StuckLoopReport;
  scoring?: {
    qualityScore: number;
    qMax: number;
    qualityFloor: number;
    effectiveCost: number;
    benchmarkRefs: string[];
    baselineModel: string;
    estimatedSavingsPct: number;
  };
  latencyMs: {
    total: number;
    fastPath: number;
    classifier: number;
    scoring: number;
  };
  fallbackReason?: string;
  headers: Record<string, string>;
}

export interface ClassifierRequest {
  envelope: RoutingEnvelope;
  domains: DomainDefinition[];
  askOrchestrator: boolean;
  askEscalate: boolean;
  timeoutMs: number;
}

export interface ClassifierAdapter {
  readonly name: string;
  classify(request: ClassifierRequest): Promise<SystemOneSignals>;
}

export interface BackendCalibration {
  /** Added to raw [0,1] complexity score (e.g. -0.15 for tev1:0.8b over-scoring bias) */
  complexityOffset?: number;
  noulOrchestratorThreshold?: number;
  noulEscalateThreshold?: number;
}

export interface SystemOneClientConfig {
  endpointUrl: string;
  model: string;
  apiKey?: string;
  headers?: Record<string, string>;
  timeoutMs?: number;
  calibration?: BackendCalibration;
  /** Consecutive errors before opening circuit breaker (default: 3) */
  circuitBreakerThreshold?: number;
  /** Cooldown before half-open probe in ms (default: 30_000) */
  circuitBreakerCooldownMs?: number;
  fetchImpl?: typeof fetch;
}

export interface S1GateConfig {
  defaultProfile?: BudgetProfile;
  privacyMode?: PrivacyMode;
  /** Optional allowlist of virtual model triggers (if omitted, every route() call is routed) */
  triggerModels?: string[];
  /** Auxiliary fast model preference list (matched against availableModels) */
  auxiliaryModelHints?: string[];
  /** Explicit custom domains (merged with / overriding built-in domains.json) */
  domains?: DomainDefinition[];
  /** Custom snapshot (defaults to bundled data/snapshot.json) */
  snapshot?: BenchmarkSnapshot;
  /** Per-model capability/price/compat overrides */
  modelOverrides?: Record<string, ModelOverride>;
  /** Custom classifier adapter (overrides systemOne config if provided) */
  classifier?: ClassifierAdapter;
  /** System One endpoint config (TypeSafe Jev, Ollama Cloud clef:27b, local Ollama tev1, Cloudflare clef) */
  systemOne?: SystemOneClientConfig;
  /** Classifier timeout in ms (default: 1500) */
  classifierTimeoutMs?: number;
  /** Sticky session TTL in ms (default: 2 hours) */
  sessionTtlMs?: number;
  /** Max sticky sessions in LRU cache (default: 5000) */
  maxSessions?: number;
  /** Enable stuck-loop escalation ratchet (default: true) */
  enableEscalationRatchet?: boolean;
}

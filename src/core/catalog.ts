import type {
  BenchmarkSnapshot,
  CandidateModel,
  CompatTier,
  HarnessId,
  ModelOverride,
  SnapshotModelEntry,
} from "../types.js";

/** Known explicit alias fixes where proxy slugs differ non-trivially from llm-stats canonical IDs */
const BUILTIN_ALIASES: Record<string, string> = {
  "beam": "beam-1.0-501b-a23b",
  "hy4": "hy4-flash",
  "nemotron-3-super": "nemotron-3-super-120b-a12b",
  "nemotron-3-ultra": "nemotron-3-ultra-550b-a55b",
  "gemma4-31b": "gemma-4-31b-it",
  "gemma4:31b": "gemma-4-31b-it",
};

/** Non-chat utility/media models that must never be selected for LLM coding/chat routing */
const EXCLUDED_MODEL_PATTERN =
  /(?:^|[-_/:.])(flash-image|imagen|dall-e|whisper|tts|embedding|embed|clef|tev1|jev|clairic|moderation)(?:$|[-_/:.])/i;

/**
 * Normalize a proxy model ID (OpenCodex, 9Router, OpenRouter, LiteLLM) into a canonical lookup candidate.
 */
export function normalizeProxyModelId(rawId: string): string {
  let id = rawId.trim().toLowerCase();
  // Strip OpenCodex Claude-slot prefix: ocx-claude-<provider>--<model>
  id = id.replace(/^ocx-claude-[a-z0-9._-]+--/, "");
  // Strip provider prefix (anthropic/, openai/, google-antigravity/, ollama-cloud/, cc/, cx/, glm/, etc.)
  const slashIdx = id.indexOf("/");
  if (slashIdx > 0) id = id.slice(slashIdx + 1);
  // Strip OpenRouter or OCX variant suffixes: :free, :exacto, :0813, --fast, -tiered-high, -high/-medium/-low
  id = id.replace(/--(fast|high|medium|low|xhigh|max|ultra)$/, "");
  id = id.replace(/-(tiered-)?(high|medium|low|xhigh)$/, "");
  id = id.replace(/:(free|extended|nitro|floor|exacto|thinking|\d{4})$/, "");
  // Convert remaining colon (e.g. gpt-oss:120b, deepseek-v4-pro:0813) to hyphen
  id = id.replace(/:/g, "-");
  return id;
}

export function isNonChatUtilityModel(rawId: string): boolean {
  return EXCLUDED_MODEL_PATTERN.test(rawId);
}

/**
 * Resolve a proxy-specific model ID to its canonical `llm-stats.com` snapshot key.
 */
export function resolveCanonicalModelId(
  rawId: string,
  snapshot: BenchmarkSnapshot,
  overrides?: Record<string, ModelOverride>,
): string | null {
  const overrideCanonical = overrides?.[rawId]?.canonicalId;
  if (overrideCanonical && snapshot.models[overrideCanonical]) {
    return overrideCanonical;
  }

  const keys = Object.keys(snapshot.models);
  const lowerMap = new Map<string, string>(keys.map(k => [k.toLowerCase(), k]));

  const exact = lowerMap.get(rawId.trim().toLowerCase());
  if (exact) return exact;

  const norm = normalizeProxyModelId(rawId);
  const direct = lowerMap.get(norm);
  if (direct) return direct;

  const alias = BUILTIN_ALIASES[norm];
  if (alias) {
    const aliased = lowerMap.get(alias.toLowerCase());
    if (aliased) return aliased;
  }

  // Match versioned snapshots like claude-haiku-4-5 -> claude-haiku-4-5-20251001 (prefer shortest prefix match)
  let bestPrefix: string | null = null;
  for (const k of keys) {
    const kl = k.toLowerCase();
    if (kl.startsWith(`${norm}-`) || kl.startsWith(`${norm}_`)) {
      if (!bestPrefix || k.length < bestPrefix.length) bestPrefix = k;
    }
  }
  if (bestPrefix) return bestPrefix;

  // Handle compact family names like gemma4-31b -> gemma-4-31b*
  const spaced = norm.replace(/([a-z])(\d)/g, "$1-$2");
  if (spaced !== norm) {
    const spacedExact = lowerMap.get(spaced);
    if (spacedExact) return spacedExact;
    for (const k of keys) {
      if (k.toLowerCase().startsWith(spaced)) return k;
    }
  }

  return null;
}

function inferHarnessCompat(entry: SnapshotModelEntry | null, rawId: string): Partial<Record<HarnessId, CompatTier>> {
  const lower = `${rawId} ${entry?.id ?? ""} ${entry?.organization ?? ""}`.toLowerCase();
  if (lower.includes("claude") || lower.includes("anthropic")) {
    return {
      claude_code: "native",
      codex: "translated_verified",
      aider: "native",
      opencode: "native",
      cursor: "native",
      generic: "native",
    };
  }
  if (lower.includes("gpt") || lower.includes("openai")) {
    return {
      claude_code: "translated_verified",
      codex: "native",
      aider: "native",
      opencode: "native",
      cursor: "native",
      generic: "native",
    };
  }
  return {
    claude_code: "translated_verified",
    codex: "translated_verified",
    aider: "native",
    opencode: "native",
    cursor: "translated_verified",
    generic: "native",
  };
}

/**
 * Build an immutable list of CandidateModel objects for the proxy's currently active models.
 */
export function buildCandidatePool(
  availableModels: readonly string[],
  snapshot: BenchmarkSnapshot,
  overrides?: Record<string, ModelOverride>,
): CandidateModel[] {
  const result: CandidateModel[] = [];
  const seenProxyIds = new Set<string>();

  for (const rawId of availableModels) {
    const proxyId = rawId.trim();
    if (!proxyId || seenProxyIds.has(proxyId) || isNonChatUtilityModel(proxyId)) continue;
    seenProxyIds.add(proxyId);

    const canonicalId = resolveCanonicalModelId(proxyId, snapshot, overrides);
    const entry = canonicalId ? snapshot.models[canonicalId] : null;
    const ov = overrides?.[proxyId] ?? (canonicalId ? overrides?.[canonicalId] : undefined);

    const baseCompat = inferHarnessCompat(entry, proxyId);
    const harnessCompat: Partial<Record<HarnessId, CompatTier>> = {
      ...baseCompat,
      ...(ov?.harnessCompat ?? {}),
    };

    result.push({
      proxyId,
      canonicalId,
      name: entry?.name ?? proxyId,
      organization: entry?.organization ?? "unknown",
      contextWindow: ov?.contextWindow ?? entry?.contextWindow ?? 128_000,
      inputPricePerMTok: ov?.inputPricePerMTok ?? entry?.inputPricePerMTok ?? 0,
      outputPricePerMTok: ov?.outputPricePerMTok ?? entry?.outputPricePerMTok ?? 0,
      cacheReadMultiplier: ov?.cacheReadMultiplier ?? entry?.cacheReadMultiplier ?? 0.25,
      cacheWriteMultiplier: ov?.cacheWriteMultiplier ?? entry?.cacheWriteMultiplier ?? 1.0,
      proxyStripsCacheControl: ov?.proxyStripsCacheControl ?? false,
      thinkingOutputRatio: entry?.thinkingOutputRatio ?? 0.3,
      quotaWeightMultiplier: ov?.quotaWeightMultiplier ?? 1.0,
      multimodal: entry?.multimodal ?? false,
      supportsTools: ov?.supportsTools ?? entry?.supportsTools ?? true,
      requiresThinkingSignature: entry?.requiresThinkingSignature ?? false,
      harnessCompat,
      scores: entry ? { ...entry.scores } : {},
    });
  }

  return result;
}

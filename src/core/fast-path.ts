import type {
  BudgetProfile,
  ExecutionRole,
  S1GateConfig,
  S1GateRequest,
} from "../types.js";
import { isNonChatUtilityModel } from "./catalog.js";

export interface FastPathVerdict {
  kind: "passthrough" | "auxiliary" | "compact" | "proceed";
  reason?: string;
  role: ExecutionRole;
  roleOverride?: ExecutionRole;
  profileOverride?: BudgetProfile;
  hasTools: boolean;
  canSpawnSubagent: boolean;
  requiresMultimodal: boolean;
  estimatedInputTokens: number;
  hasThinkingSignature: boolean;
  isSubagentHeader: boolean;
}

const VALID_PROFILES = new Set<BudgetProfile>(["frugal", "balanced", "quality"]);
const VALID_ROLES = new Set<ExecutionRole>([
  "orchestrator",
  "subagent",
  "specialist",
  "auxiliary",
]);

const DEFAULT_AUXILIARY_PATTERNS: readonly string[] = [
  "haiku",
  "flash-lite",
  "flash_lite",
  "mini",
  "nano",
  "luna",
];

const SUBAGENT_TOOL_RE = /^(agent|task|subagent|invoke_subagent|spawn_agent)$/i;

const MULTIMODAL_BLOCK_RE =
  /^(image|image_url|input_image|document|file|input_file|pdf|audio|input_audio|video|input_video)$/i;

const MULTIMODAL_MIME_RE = /^(image\/|audio\/|video\/|application\/pdf)/i;

const COMPACTION_PROMPT_RE =
  /(compact(ing|ion)?\s+(the\s+)?(conversation|context|history)|(conversation|context|history)\s+compaction|summariz(e|ing)\s+(the\s+)?(conversation|prior\s+messages|context)\s+(so\s+far|for\s+continuation|to\s+reduce|to\s+save)|create\s+a\s+detailed\s+summary\s+of\s+the\s+conversation|your\s+task\s+is\s+to\s+create\s+a\s+detailed\s+summary)/i;

const UTILITY_PROMPT_RE =
  /(generate\s+(a\s+)?(short\s+|concise\s+|brief\s+)?(\d+[-\s]\d+\s+word\s+)?(title|headline|subject|branch\s+name|commit\s+message)|write\s+a\s+(concise\s+|short\s+|brief\s+)?(title|git\s+commit\s+message|branch\s+name)|summarize\s+this\s+(conversation|chat|session)\s+in\s+\d+\s+words|title\s+for\s+this\s+(conversation|chat|thread))/i;

const STACK_TRACE_RE =
  /(Traceback \(most recent call last\)|\bat\s+\S+\s+\([^)]+:\d+:\d+\)|^\s*at\s+\S+:\d+:\d+|(?:Type|Reference|Syntax|Range|Runtime)?Error:\s+.+\n\s+at\s|\bpanic:|\bSEGFAULT\b)/m;

const COMPLEX_KEYWORD_RE =
  /\b(architect|architecture|security\s+audit|race\s+condition|prove|proof|deadlock|distributed|consensus|refactor|migration|vulnerability|exploit|concurrency)\b/i;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeHeaders(
  headers: Record<string, string | undefined>,
): Map<string, string> {
  const map = new Map<string, string>();
  for (const [k, v] of Object.entries(headers)) {
    if (typeof v === "string") {
      const trimmed = v.trim();
      if (trimmed.length > 0) {
        map.set(k.toLowerCase(), trimmed);
      }
    }
  }
  return map;
}

function extractTextFromContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === "string") {
      parts.push(block);
    } else if (isRecord(block)) {
      if (typeof block.text === "string") {
        parts.push(block.text);
      } else if (typeof block.content === "string") {
        parts.push(block.content);
      }
    }
  }
  return parts.join("\n");
}

function extractMessages(body: Record<string, unknown>): Record<string, unknown>[] {
  if (Array.isArray(body.messages)) {
    return body.messages.filter(isRecord);
  }
  if (Array.isArray(body.input)) {
    return body.input.filter(isRecord);
  }
  if (typeof body.input === "string") {
    return [{ role: "user", content: body.input }];
  }
  return [];
}

function extractSystemText(
  body: Record<string, unknown>,
  messages: readonly Record<string, unknown>[],
): string {
  const parts: string[] = [];
  if (body.system !== undefined) {
    const sys = extractTextFromContent(body.system);
    if (sys) parts.push(sys);
  }
  if (typeof body.instructions === "string" && body.instructions.trim().length > 0) {
    parts.push(body.instructions);
  }
  for (const msg of messages) {
    const role = typeof msg.role === "string" ? msg.role.toLowerCase() : "";
    if (role === "system" || role === "developer") {
      const text = extractTextFromContent(msg.content);
      if (text) parts.push(text);
    }
  }
  return parts.join("\n");
}

function detectTools(body: Record<string, unknown>): {
  hasTools: boolean;
  canSpawnSubagent: boolean;
} {
  if (!Array.isArray(body.tools) || body.tools.length === 0) {
    return { hasTools: false, canSpawnSubagent: false };
  }
  let canSpawnSubagent = false;
  for (const tool of body.tools) {
    if (!isRecord(tool)) continue;
    let name: string | undefined;
    if (typeof tool.name === "string") {
      name = tool.name;
    } else if (isRecord(tool.function) && typeof tool.function.name === "string") {
      name = tool.function.name;
    }
    if (name && SUBAGENT_TOOL_RE.test(name.trim())) {
      canSpawnSubagent = true;
      break;
    }
  }
  return { hasTools: true, canSpawnSubagent };
}

function isMultimodalBlock(block: unknown): boolean {
  if (!isRecord(block)) return false;
  if (typeof block.type === "string" && MULTIMODAL_BLOCK_RE.test(block.type.trim())) {
    return true;
  }
  if (
    isRecord(block.source) &&
    typeof block.source.media_type === "string" &&
    MULTIMODAL_MIME_RE.test(block.source.media_type.trim())
  ) {
    return true;
  }
  return false;
}

function detectMultimodalCurrentTurn(
  messages: readonly Record<string, unknown>[],
): boolean {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const role = typeof msg.role === "string" ? msg.role.toLowerCase() : "";
    if (role === "user") {
      if (!Array.isArray(msg.content)) return false;
      return msg.content.some(isMultimodalBlock);
    }
  }
  return false;
}

function detectThinkingSignature(
  messages: readonly Record<string, unknown>[],
): boolean {
  for (const msg of messages) {
    const role = typeof msg.role === "string" ? msg.role.toLowerCase() : "";
    if (role !== "assistant" || !Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (
        isRecord(block) &&
        block.type === "thinking" &&
        typeof block.signature === "string" &&
        block.signature.trim().length > 0
      ) {
        return true;
      }
    }
  }
  return false;
}

function detectSubagentHeader(headers: Map<string, string>): boolean {
  if (headers.has("x-claude-code-agent-id")) {
    return true;
  }
  const openaiSubagent = headers.get("x-openai-subagent");
  if (
    openaiSubagent !== undefined &&
    openaiSubagent.toLowerCase() !== "false" &&
    openaiSubagent !== "0"
  ) {
    return true;
  }
  const parentThread = headers.get("x-codex-parent-thread-id");
  const threadId = headers.get("thread-id");
  if (parentThread && threadId && parentThread !== threadId) {
    return true;
  }
  return false;
}

function estimateInputTokens(body: Record<string, unknown>): number {
  try {
    return Math.ceil(JSON.stringify(body).length / 4);
  } catch {
    return 0;
  }
}

function extractMaxTokens(body: Record<string, unknown>): number | undefined {
  for (const key of ["max_tokens", "max_completion_tokens", "max_output_tokens"]) {
    const val = body[key];
    if (typeof val === "number" && Number.isFinite(val)) {
      return val;
    }
  }
  return undefined;
}

function isShortAuxiliaryCandidate(
  body: Record<string, unknown>,
  messages: readonly Record<string, unknown>[],
  hasTools: boolean,
  requiresMultimodal: boolean,
): boolean {
  if (hasTools || requiresMultimodal) return false;
  const nonSystem = messages.filter((m) => {
    const role = typeof m.role === "string" ? m.role.toLowerCase() : "";
    return role !== "system" && role !== "developer";
  });
  if (nonSystem.length === 0 || nonSystem.length > 2) return false;

  const maxTokens = extractMaxTokens(body);
  if (maxTokens === undefined || maxTokens <= 0 || maxTokens > 512) return false;

  const promptText = nonSystem
    .map((m) => extractTextFromContent(m.content))
    .join("\n")
    .trim();

  if (promptText.length >= 250) return false;
  if (promptText.includes("```")) return false;
  if (STACK_TRACE_RE.test(promptText)) return false;
  if (COMPLEX_KEYWORD_RE.test(promptText)) return false;
  return true;
}

function matchesTriggerAllowlist(
  rawModel: string,
  hintBase: string | undefined,
  triggerModels: readonly string[],
): boolean {
  const normalizedRaw = rawModel.trim().toLowerCase();
  const normalizedBase = hintBase?.trim().toLowerCase();
  for (const trigger of triggerModels) {
    const normalizedTrigger = trigger.trim().toLowerCase();
    if (!normalizedTrigger) continue;
    if (normalizedRaw === normalizedTrigger) return true;
    if (normalizedBase && normalizedBase === normalizedTrigger) return true;
  }
  return false;
}

export function parseVirtualModelHint(
  model: string,
): { base: string; profile?: BudgetProfile; role?: ExecutionRole } | null {
  const trimmed = model.trim();
  if (!trimmed || !trimmed.includes(":")) return null;

  const segments = trimmed.split(":");
  let profile: BudgetProfile | undefined;
  let role: ExecutionRole | undefined;
  let endIndex = segments.length;

  while (endIndex > 1) {
    const token = segments[endIndex - 1].trim().toLowerCase();
    if (VALID_PROFILES.has(token as BudgetProfile) && profile === undefined) {
      profile = token as BudgetProfile;
      endIndex -= 1;
    } else if (token === "fast" && role === undefined) {
      role = "auxiliary";
      endIndex -= 1;
    } else if (VALID_ROLES.has(token as ExecutionRole) && role === undefined) {
      role = token as ExecutionRole;
      endIndex -= 1;
    } else {
      break;
    }
  }

  if (profile === undefined && role === undefined) return null;

  const base = segments.slice(0, endIndex).join(":").trim();
  if (!base) return null;

  const result: { base: string; profile?: BudgetProfile; role?: ExecutionRole } = {
    base,
  };
  if (profile !== undefined) result.profile = profile;
  if (role !== undefined) result.role = role;
  return result;
}

export function pickAuxiliaryModel(
  availableModels: readonly string[],
  hints?: readonly string[],
): string {
  const chatModels = availableModels.filter((m) => !isNonChatUtilityModel(m));
  if (chatModels.length === 0) return "";

  const lowerModels = chatModels.map((m) => m.toLowerCase());

  if (hints && hints.length > 0) {
    for (const hint of hints) {
      const target = hint.trim().toLowerCase();
      if (!target) continue;
      const exactIdx = lowerModels.findIndex((m) => m === target);
      if (exactIdx !== -1) return chatModels[exactIdx];
      const partialIdx = lowerModels.findIndex((m) => m.includes(target));
      if (partialIdx !== -1) return chatModels[partialIdx];
    }
  }

  for (const pattern of DEFAULT_AUXILIARY_PATTERNS) {
    const idx = lowerModels.findIndex((m) => m.includes(pattern));
    if (idx !== -1) return chatModels[idx];
  }

  return chatModels[0];
}

export function inspectFastPath(
  req: S1GateRequest,
  config?: Pick<S1GateConfig, "triggerModels" | "defaultProfile">,
): FastPathVerdict {
  const headers = normalizeHeaders(req.headers);
  const body = isRecord(req.body) ? req.body : {};
  const rawModel = typeof body.model === "string" ? body.model : "";
  const hint = parseVirtualModelHint(rawModel);
  const messages = extractMessages(body);
  const systemText = extractSystemText(body, messages);
  const { hasTools, canSpawnSubagent } = detectTools(body);
  const requiresMultimodal = detectMultimodalCurrentTurn(messages);
  const hasThinkingSignature = detectThinkingSignature(messages);
  const estimatedInputTokens = estimateInputTokens(body);
  const isSubagentHeader = detectSubagentHeader(headers);

  const profileOverride = req.profile ?? hint?.profile;
  const roleOverride = req.roleOverride ?? hint?.role;
  const resolvedRole: ExecutionRole =
    roleOverride ??
    (isSubagentHeader
      ? "subagent"
      : canSpawnSubagent
        ? "orchestrator"
        : "specialist");

  const baseVerdict = {
    role: resolvedRole,
    ...(roleOverride !== undefined ? { roleOverride } : {}),
    ...(profileOverride !== undefined ? { profileOverride } : {}),
    hasTools,
    canSpawnSubagent,
    requiresMultimodal,
    estimatedInputTokens,
    hasThinkingSignature,
    isSubagentHeader,
  };

  if (config?.triggerModels && config.triggerModels.length > 0) {
    if (!matchesTriggerAllowlist(rawModel, hint?.base, config.triggerModels)) {
      return {
        ...baseVerdict,
        kind: "passthrough",
        reason: "model_not_in_trigger_list",
      };
    }
  }

  const cleanPath = (req.path ?? "").split("?")[0];
  if (cleanPath.includes("/v1/messages/count_tokens")) {
    return {
      ...baseVerdict,
      kind: "passthrough",
      reason: "count_tokens",
    };
  }

  const anthropicBeta = headers.get("anthropic-beta") ?? "";
  const isCompactPath = cleanPath.endsWith("/compact");
  const isCompactBeta = anthropicBeta.toLowerCase().includes("compact-");
  const isCompactSystemPrompt =
    !hasTools && systemText.length > 0 && COMPACTION_PROMPT_RE.test(systemText);
  if (isCompactPath || isCompactBeta || isCompactSystemPrompt) {
    return {
      ...baseVerdict,
      kind: "compact",
      reason: isCompactPath
        ? "compact_endpoint"
        : isCompactBeta
          ? "compact_beta_header"
          : "compact_system_prompt",
    };
  }

  if (hint?.role === "auxiliary" || req.roleOverride === "auxiliary") {
    return {
      ...baseVerdict,
      kind: "auxiliary",
      role: "auxiliary",
      reason: req.roleOverride === "auxiliary" ? "role_override" : "virtual_hint_fast",
    };
  }

  if (
    !hasTools &&
    messages.length <= 4 &&
    systemText.length > 0 &&
    UTILITY_PROMPT_RE.test(systemText)
  ) {
    return {
      ...baseVerdict,
      kind: "auxiliary",
      role: "auxiliary",
      reason: "utility_system_prompt",
    };
  }

  if (isShortAuxiliaryCandidate(body, messages, hasTools, requiresMultimodal)) {
    return {
      ...baseVerdict,
      kind: "auxiliary",
      role: "auxiliary",
      reason: "short_non_tool_prompt",
    };
  }

  return {
    ...baseVerdict,
    kind: "proceed",
  };
}

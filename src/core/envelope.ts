import type {
  ExecutionRole,
  RoutingEnvelope,
  S1GateRequest,
  StuckLoopReport,
} from "../types.js";

const UTF8_ENCODER = new TextEncoder();
const DEFAULT_MAX_BYTES = 8192;
const MAX_TOOL_NAMES = 12;
const MAX_ERROR_CHARS = 600;

const CLOSED_FENCED_CODE_REGEX = /(`{3,}|~{3,})([a-zA-Z0-9_+#.-]*)[ \t]*\r?\n([\s\S]*?)\1/g;
const UNCLOSED_FENCED_CODE_REGEX = /(`{3,}|~{3,})([a-zA-Z0-9_+#.-]*)[ \t]*\r?\n([\s\S]*)$/g;

const SECRET_PATTERNS: readonly RegExp[] = [
  /-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----/g,
  /\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|rediss?|amqps?|mssql):\/\/[^\s"'`<>)\]]+/gi,
  /\bAuthorization:\s*Basic\s+[A-Za-z0-9+/=]+/gi,
  /\bBearer\s+[A-Za-z0-9._~+/=-]{8,}/gi,
  /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g,
  /\bsk-[A-Za-z0-9_-]{10,}\b/g,
  /\bsk_(?:live|test)_[A-Za-z0-9]+\b/g,
  /\b(?:gh[pousr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,})\b/g,
  /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g,
  /\bAIza[0-9A-Za-z_-]{35}\b/g,
  /\bxox[baprs]-[A-Za-z0-9-]+\b/g,
  /\bnpm_[A-Za-z0-9]+\b/g,
  /\bhf_[A-Za-z0-9]+\b/g,
  /\b(?:PASSWORD|PASSWD|SECRET|API_KEY|APIKEY|ACCESS_TOKEN|PRIVATE_KEY|AWS_SECRET_ACCESS_KEY)\s*[:=]\s*(?!['"]?\[REDACTED:)['"]?[^\s'",;]{6,}['"]?/gi,
];

const PII_PATTERNS: readonly RegExp[] = [
  /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
  /\b(?:https?|ftp):\/\/[^\s"'`<>)\]]+/gi,
  /(?<![0-9A-Za-z:])(?:(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|::(?:ffff(?::0{1,4})?:)?(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)|(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,5}(?::[0-9A-Fa-f]{1,4}){1,2}|(?:[0-9A-Fa-f]{1,4}:){1,4}(?::[0-9A-Fa-f]{1,4}){1,3}|(?:[0-9A-Fa-f]{1,4}:){1,3}(?::[0-9A-Fa-f]{1,4}){1,4}|(?:[0-9A-Fa-f]{1,4}:){1,2}(?::[0-9A-Fa-f]{1,4}){1,5}|[0-9A-Fa-f]{1,4}:(?::[0-9A-Fa-f]{1,4}){1,6}|:(?::[0-9A-Fa-f]{1,4}){1,7}|(?:[0-9A-Fa-f]{1,4}:){1,7}:)(?![0-9A-Za-z:])/gi,
  /\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b/g,
  /\b(?:MRN|No\.?\s*RM|RM|Patient[\s_-]?ID)[:\s#-]*\d+(?:-\d+)*\b/gi,
  /\b\d{3}-\d{2}-\d{4}\b/g,
  /\b(?:\d{4}[-\s]?){3}\d{4}\b/g,
  /(?<!\d)(?:\+\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}|\(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4}|\b08\d{2}[\s.-]?\d{3,4}[\s.-]?\d{3,5}\b|\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b)(?!\d)/g,
];

const ERROR_TEXT_REGEX =
  /\b(?:error|exception|traceback|failed|typeerror|syntaxerror|referenceerror|parseerror|valueerror|enoent|econnrefused|exit code|exited with code|panic:|fatal:)\b/i;

export interface RedactionCounts {
  codeBlocks: number;
  secrets: number;
  pii: number;
}

export function redactSensitiveText(raw: string): {
  text: string;
  counts: RedactionCounts;
} {
  let codeBlocks = 0;
  let secrets = 0;
  let pii = 0;

  const formatCodeStub = (_match: string, _fence: string, rawLang: string, content: string): string => {
    codeBlocks += 1;
    const lang = rawLang.trim() || "text";
    const trimmed = content.replace(/\r?\n$/, "");
    const lines = trimmed.length === 0 ? 0 : trimmed.split(/\r?\n/).length;
    return `[CODE: ${lang}, ${lines} lines]`;
  };

  let text = raw.replace(CLOSED_FENCED_CODE_REGEX, formatCodeStub);
  text = text.replace(UNCLOSED_FENCED_CODE_REGEX, formatCodeStub);

  for (const pattern of SECRET_PATTERNS) {
    text = text.replace(pattern, () => {
      secrets += 1;
      return "[REDACTED:SECRET]";
    });
  }

  for (const pattern of PII_PATTERNS) {
    text = text.replace(pattern, () => {
      pii += 1;
      return "[REDACTED:PII]";
    });
  }

  return { text, counts: { codeBlocks, secrets, pii } };
}

function utf8ByteLength(text: string): number {
  return UTF8_ENCODER.encode(text).byteLength;
}

function truncateUtf8(text: string, maxBytes: number): string {
  if (maxBytes <= 0) return "";
  if (utf8ByteLength(text) <= maxBytes) return text;

  const ellipsis = "...";
  const budget = maxBytes > 3 ? maxBytes - 3 : maxBytes;
  let low = 0;
  let high = text.length;

  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    const slice = safeCodeUnitSlice(text, mid);
    if (utf8ByteLength(slice) <= budget) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }

  const cut = safeCodeUnitSlice(text, low);
  return maxBytes > 3 ? `${cut}${ellipsis}` : cut;
}

function safeCodeUnitSlice(text: string, endIndex: number): string {
  if (endIndex <= 0) return "";
  if (endIndex >= text.length) return text;
  const code = text.charCodeAt(endIndex - 1);
  // Avoid splitting a UTF-16 high surrogate from its low surrogate
  if (code >= 0xd800 && code <= 0xdbff) {
    return text.slice(0, endIndex - 1);
  }
  return text.slice(0, endIndex);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function extractToolNames(body: Record<string, unknown>): string[] {
  const rawTools = body.tools;
  if (!Array.isArray(rawTools)) return [];

  const names: string[] = [];
  const seen = new Set<string>();

  for (const item of rawTools) {
    if (names.length >= MAX_TOOL_NAMES) break;
    const rec = asRecord(item);
    if (!rec) continue;

    const fnRec = asRecord(rec.function);
    const candidate =
      (typeof rec.name === "string" && rec.name.trim()) ||
      (typeof fnRec?.name === "string" && fnRec.name.trim()) ||
      (typeof rec.type === "string" && rec.type !== "function" && rec.type.trim()) ||
      "";

    if (candidate && !seen.has(candidate)) {
      seen.add(candidate);
      names.push(candidate);
    }
  }

  return names;
}

function extractTextFromContent(content: unknown, includeToolResults = false): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  const parts: string[] = [];
  for (const block of content) {
    const rec = asRecord(block);
    if (!rec) continue;
    const type = typeof rec.type === "string" ? rec.type : "";
    if ((type === "text" || type === "input_text" || type === "output_text") && typeof rec.text === "string") {
      parts.push(rec.text);
    } else if (includeToolResults && type === "tool_result") {
      const inner = extractTextFromContent(rec.content, false);
      if (inner) parts.push(inner);
    }
  }
  return parts.join("\n").trim();
}

function extractErrorFromMessage(msg: Record<string, unknown>): string | null {
  const role = typeof msg.role === "string" ? msg.role : "";
  const type = typeof msg.type === "string" ? msg.type : "";

  if (Array.isArray(msg.content)) {
    for (let i = msg.content.length - 1; i >= 0; i -= 1) {
      const block = asRecord(msg.content[i]);
      if (!block) continue;
      if (block.type === "tool_result" && block.is_error === true) {
        const errText = extractTextFromContent(block.content, false);
        if (errText) return errText;
      }
    }
  }

  if (role === "tool" || type === "function_call_output") {
    const raw =
      typeof msg.output === "string"
        ? msg.output
        : extractTextFromContent(msg.content, true);
    if (raw && ERROR_TEXT_REGEX.test(raw)) return raw;
  }

  if (role === "assistant") {
    const raw = extractTextFromContent(msg.content, false);
    if (raw && ERROR_TEXT_REGEX.test(raw)) return raw;
  }

  return null;
}

interface ExtractedConversation {
  turnCount: number;
  latestUserPrompt: string;
  recentError: string | null;
}

function extractConversation(body: Record<string, unknown>): ExtractedConversation {
  if (typeof body.input === "string") {
    return {
      turnCount: body.input.trim().length > 0 ? 1 : 0,
      latestUserPrompt: body.input,
      recentError: null,
    };
  }

  const rawItems = Array.isArray(body.messages)
    ? body.messages
    : Array.isArray(body.input)
      ? body.input
      : [];

  let userTurns = 0;
  let latestUserPrompt = "";
  let recentError: string | null = null;

  for (const item of rawItems) {
    const msg = asRecord(item);
    if (!msg) continue;
    if (msg.role === "user") {
      const text = extractTextFromContent(msg.content, false);
      if (text) {
        userTurns += 1;
        latestUserPrompt = text;
      }
    }
  }

  const tailStart = Math.max(0, rawItems.length - 6);
  for (let i = rawItems.length - 1; i >= tailStart; i -= 1) {
    const msg = asRecord(rawItems[i]);
    if (!msg) continue;
    const err = extractErrorFromMessage(msg);
    if (err) {
      recentError = err;
      break;
    }
  }

  return {
    turnCount: userTurns || rawItems.length,
    latestUserPrompt,
    recentError,
  };
}

export function buildRoutingEnvelope(
  req: S1GateRequest,
  opts?: { role?: ExecutionRole; stuckLoop?: StuckLoopReport; maxBytes?: number },
): RoutingEnvelope {
  const maxBytes = Math.max(0, opts?.maxBytes ?? DEFAULT_MAX_BYTES);
  const role: ExecutionRole = opts?.role ?? req.roleOverride ?? "orchestrator";
  const harness = req.harness ?? "generic";
  const toolNames = extractToolNames(req.body);
  const { turnCount, latestUserPrompt, recentError } = extractConversation(req.body);

  const counts: RedactionCounts = { codeBlocks: 0, secrets: 0, pii: 0 };
  const addCounts = (delta: RedactionCounts): void => {
    counts.codeBlocks += delta.codeBlocks;
    counts.secrets += delta.secrets;
    counts.pii += delta.pii;
  };

  const promptRedacted = redactSensitiveText(latestUserPrompt);
  addCounts(promptRedacted.counts);

  let errorLine: string | null = null;
  if (recentError) {
    const errRedacted = redactSensitiveText(recentError);
    addCounts(errRedacted.counts);
    const compactErr = errRedacted.text.replace(/\s+/g, " ").trim().slice(0, MAX_ERROR_CHARS);
    if (compactErr) errorLine = `recent_error: ${compactErr}`;
  }

  let stuckLine: string | null = null;
  if (opts?.stuckLoop?.telemetrySummary) {
    const stuckRedacted = redactSensitiveText(opts.stuckLoop.telemetrySummary);
    addCounts(stuckRedacted.counts);
    stuckLine = `stuck_loop: [level=${opts.stuckLoop.level}] ${stuckRedacted.text.trim()}`;
  }

  const headerLines = [
    `protocol: ${req.protocol}`,
    `harness: ${harness}`,
    `role: ${role}`,
    `turns: ${turnCount}`,
  ];
  if (toolNames.length > 0) {
    headerLines.push(`tools: ${toolNames.join(", ")}`);
  }
  if (stuckLine) headerLines.push(stuckLine);
  if (errorLine) headerLines.push(errorLine);

  const prefix = `${headerLines.join("\n")}\nprompt: `;
  const prefixBytes = utf8ByteLength(prefix);

  let finalText: string;
  if (prefixBytes >= maxBytes) {
    finalText = truncateUtf8(`${prefix}${promptRedacted.text.trim()}`, maxBytes);
  } else {
    const promptBudget = maxBytes - prefixBytes;
    const cappedPrompt = truncateUtf8(promptRedacted.text.trim(), promptBudget);
    finalText = truncateUtf8(`${prefix}${cappedPrompt}`, maxBytes);
  }

  const byteLength = utf8ByteLength(finalText);
  return {
    text: finalText,
    byteLength,
    estimatedTokens: Math.ceil(byteLength / 4),
    redactionCounts: counts,
  };
}

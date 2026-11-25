import type { InboundProtocol, SessionPin, StuckLevel, StuckLoopReport } from "../types.js";

const DEFAULT_MIN_LOCK_TURNS = 4;
const MAX_ROLLING_TURNS = 20;
const MAX_TELEMETRY_CHARS = 299;
const MAX_NORMALIZED_SIG_CHARS = 400;

const READ_ONLY_TOOL_PATTERN =
  /^(read|read_file|view_file|view_file_outline|view_code_item|cat|glob|find|find_by_name|grep|grep_search|ripgrep|rg|search|search_files|ls|list_dir|list_files|head|tail|stat|tree|web_search|search_web)$/i;

const EDIT_TOOL_PATTERN =
  /^(edit|edit_file|str_replace|str_replace_editor|str_replace_based_edit_tool|replace_file_content|multi_replace_file_content|patch|apply_patch|write|write_file|write_to_file|create_file|notebook_edit|insert_edit)$/i;

const SHELL_TOOL_PATTERN = /^(bash|shell|exec|execute|run_command|terminal|cmd|command)$/i;

const VERIFICATION_TOOL_NAME_PATTERN = /(test|check|verify|lint|build|typecheck)/i;

const VERIFICATION_COMMAND_PATTERN =
  /\b(vitest|jest|pytest|mocha|ava|tap|unittest|phpunit|rspec|minitest|ctest|tsc|typecheck|eslint|biome|ruff|mypy|flake8|pylint|clippy|rubocop|gofmt|golangci-lint|(?:npm|pnpm|yarn|bun)\s+(?:x\s+|run\s+)?(?:vitest|jest|test|check|lint|build|typecheck)|(?:bunx|npx)\s+(?:vitest|jest|tsc|eslint|biome)|cargo\s+(?:test|check|clippy|build)|go\s+(?:test|vet|build)|python3?\s+-m\s+(?:pytest|unittest)|make\s+(?:test|check|build))\b/i;

const TEST_COMMAND_PATTERN =
  /\b(vitest|jest|pytest|mocha|ava|tap|unittest|phpunit|rspec|minitest|ctest|(?:npm|pnpm|yarn|bun)\s+(?:x\s+|run\s+)?(?:vitest|jest|test)|(?:bunx|npx)\s+(?:vitest|jest)|cargo\s+test|go\s+test|python3?\s+-m\s+(?:pytest|unittest)|make\s+test)\b/i;

const TEST_FILE_PATTERN = /(test|spec)\.[a-z]+$/i;
const TEST_DIR_PATTERN = /__tests__\//i;
const PYTHON_TEST_FILE_PATTERN = /(?:^|\/)test_[^/]+\.[a-z]+$/i;

const TEST_SKIP_OR_WEAKEN_PATTERN =
  /(?:\b(?:it|test|describe|context|suite)\.(?:skip|todo)\s*\(|\b(?:xit|xtest|xdescribe)\s*\(|@(?:pytest\.mark\.skip|unittest\.skip)|\bpytest\.skip\s*\(|#\[\s*ignore\b|\bt\.Skip\s*\(|\bexpect\s*\(\s*true\s*\)\s*\.toBe\s*\(\s*true\s*\)|\bassert\s+True\b)/;

const APOLOGY_MARKER_PATTERN =
  /\b(i apologize|apologies|my mistake|my bad|let me try again|try a different approach|sorry about that|i'm sorry|i am sorry|actually,?\s*wait|wait,?\s*let me reconsider|oops,?\s*i)\b/i;

const DIRECT_TOOL_ERROR_PATTERN =
  /^\s*(?:error\s*:|enoent\b|eacces\b|eperm\b|file not found|no such file|permission denied|failed to\b|old_str\b.*not found|target\b.*not found|\{\s*"error"\s*:)/i;

const NON_ZERO_EXIT_PATTERN =
  /\b(?:exit(?:ed)?\s+(?:with\s+)?(?:code|status)|return(?:ed)?\s+non-zero\s+exit\s+status|status)\s*[:=]?\s*([1-9]\d*)\b/i;

const GENERIC_ERROR_PATTERN =
  /(?:\b(?:TypeError|ReferenceError|SyntaxError|RangeError|URIError|EvalError|AssertionError|ImportError|ModuleNotFoundError|AttributeError|KeyError|ValueError|NameError|IndentationError|RuntimeError|OSError|IOError|FileNotFoundError|PermissionError|TimeoutError|Error)\s*:|\berror\s+TS\d+\b|\berror\[E\d+\]\s*:|\bTraceback\s+\(most recent call last\):|\bUnhandledPromiseRejection\b|\bCannot\s+(?:find|resolve)\s+(?:module|import)\b|\bERR_[A-Z0-9_]+\b|\b(?:ENOENT|EACCES|EPERM|ECONNREFUSED|ECONNRESET|ETIMEDOUT|EADDRINUSE|ENOTFOUND|EPIPE|EISDIR|ENOTDIR)\b|\b(?:panic|fatal)\s*:|\bcommand not found\b|\bSegmentation fault\b|\b(?:compilation|build)\s+failed\b|(?:^|\n)\s*(?:FAIL\s+\S+|FAILED\b|✗\s+|×\s+)|Expected:[\s\S]{0,200}Received:)/m;

const TEST_FAILURE_OUTPUT_PATTERN =
  /(?:\b[1-9]\d*\s+(?:tests?\s+)?(?:failed|failing)\b|(?:^|\n)\s*FAIL\s+\S+|\bAssertionError\b|Expected:[\s\S]{0,200}Received:)/;

const SYSTEM_REMINDER_TAG_PATTERN = /<system-reminder>[\s\S]*?<\/system-reminder>/gi;

const PATH_KEYS = ["path", "file_path", "filePath", "target_file", "TargetFile", "Path", "file"] as const;
const OLD_STR_KEYS = ["old_str", "old_string", "oldString", "target", "Target", "TargetContent", "search", "original"] as const;
const NEW_STR_KEYS = ["new_str", "new_string", "newString", "replacement", "Replacement", "ReplacementContent", "replace", "updated"] as const;
const CONTENT_KEYS = ["content", "file_text", "text", "CodeContent"] as const;

interface NormalizedToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  canonicalArgs: string;
}

interface NormalizedToolResult {
  toolCallId: string;
  toolName: string;
  text: string;
  explicitIsError: boolean | undefined;
}

interface AgentTurn {
  assistantText: string;
  toolCalls: NormalizedToolCall[];
  toolResults: NormalizedToolResult[];
}

interface EditOp {
  path: string;
  oldStr: string | null;
  newStr: string | null;
}

interface InspectionState {
  errorSignatures: string[];
  errorSigCounts: Map<string, number>;
  aggregateErrorCounts: number[];
  toolFailCounts: Map<string, number>;
  lastVerificationFailSig: Map<string, string>;
  verificationFailKeyCount: Map<string, number>;
  lastVerificationEditSeq: Map<string, number>;
  editSeq: number;
  priorOldStringsByPath: Map<string, Set<string>>;
  editsSinceLastPass: Map<string, number>;
  revertOscillation: boolean;
  tddEvasion: boolean;
  hasSeenTestFailure: boolean;
  editedSourceSinceTestFailure: boolean;
  hasUnresolvedError: boolean;
  apologyStreak: number;
}

/**
 * Returns true while a session pin is still within its post-escalation ratchet lock window.
 */
export function isRatchetLocked(pin: SessionPin, minLockTurns: number = DEFAULT_MIN_LOCK_TURNS): boolean {
  return pin.escalatedAtTurn !== null && pin.turnCount - pin.escalatedAtTurn < minLockTurns;
}

/**
 * Normalizes a raw error payload into a stable signature by stripping volatile runtime tokens
 * (timestamps, ISO dates, PIDs, hex addresses, temp paths, line/col numbers) while preserving
 * the error class, core message, and normalized file basenames.
 */
export function normalizeErrorSignature(rawError: string): string {
  if (!rawError) return "";

  let normalized = rawError
    .replace(/\x1b\[[0-9;]*m/g, "")
    .replace(/\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b/g, " ")
    .replace(/\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b/g, " ")
    .replace(/[\[(]?\bpid\s*[:=#]?\s*\d+\b[\])]?/gi, " ")
    .replace(/\(\s*0x[0-9a-fA-F]+\s*\)|\b0x[0-9a-fA-F]+\b/g, " ")
    .replace(
      /(?:\/private)?\/tmp\/[^\s:'",)\]]+|\/var\/folders\/[^\s:'",)\]]+|[A-Za-z]:\\[^\s:'",)\]]*\\Temp\\[^\s:'",)\]]+/gi,
      "<tmp>",
    )
    .replace(/:\d+(?::\d+)?(?=$|[\s'"),:\]])/g, "")
    .replace(/\(\d+,\s*\d+\)/g, "")
    .replace(/\bline\s+\d+(?:,\s*(?:col|column)\s+\d+)?\b/gi, "line")
    .replace(/\b\d+(?:\.\d+)?\s*(?:ms|µs|us|ns)\b/g, " ")
    .replace(
      /(^|[\s'"(,\[=])(?:[A-Za-z]:[\\/]|\/|\.\.?\/|(?:[A-Za-z0-9._-]+[\\/])+)+([A-Za-z0-9._-]+\.[A-Za-z0-9]{1,10})(?=$|[\s'"),:\]])/g,
      "$1$2",
    )
    .replace(/\s+/g, " ")
    .trim();

  if (normalized.length > MAX_NORMALIZED_SIG_CHARS) {
    normalized = `${normalized.slice(0, 200)} ... ${normalized.slice(-195)}`;
  }

  return normalized;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function canonicalizeValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(canonicalizeValue);
  }
  if (isRecord(value)) {
    const sorted: Record<string, unknown> = {};
    for (const key of Object.keys(value).sort()) {
      sorted[key] = canonicalizeValue(value[key]);
    }
    return sorted;
  }
  return value;
}

function parseToolArgs(raw: unknown): Record<string, unknown> {
  if (isRecord(raw)) return raw;
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed) return {};
    try {
      const parsed: unknown = JSON.parse(trimmed);
      return isRecord(parsed) ? parsed : { _raw: trimmed };
    } catch {
      return { _raw: trimmed };
    }
  }
  return {};
}

function extractTextContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === "string") {
      parts.push(block);
    } else if (isRecord(block)) {
      const blockType = typeof block.type === "string" ? block.type : "";
      if (
        (blockType === "text" || blockType === "input_text" || blockType === "output_text" || blockType === "") &&
        typeof block.text === "string"
      ) {
        parts.push(block.text);
      }
    }
  }
  return parts.join("\n");
}

function isRealAnthropicHumanMessage(msg: Record<string, unknown>): boolean {
  if (msg.role !== "user") return false;
  const content = msg.content;
  if (typeof content === "string") return content.trim().length > 0;
  if (!Array.isArray(content) || content.length === 0) return false;

  let hasToolResult = false;
  let hasRealHumanBlock = false;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === "tool_result") {
      hasToolResult = true;
    } else if (block.type === "text" && typeof block.text === "string") {
      if (block.text.replace(SYSTEM_REMINDER_TAG_PATTERN, "").trim().length > 0) {
        hasRealHumanBlock = true;
      }
    } else if (block.type === "image" || block.type === "document") {
      hasRealHumanBlock = true;
    }
  }

  return hasRealHumanBlock || !hasToolResult;
}

function parseAnthropicAssistantTurn(
  item: Record<string, unknown>,
  callNameById: Map<string, string>,
): AgentTurn {
  const turn: AgentTurn = { assistantText: "", toolCalls: [], toolResults: [] };
  if (typeof item.content === "string") {
    turn.assistantText = item.content;
    return turn;
  }
  if (!Array.isArray(item.content)) return turn;

  const textParts: string[] = [];
  for (const block of item.content) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      textParts.push(block.text);
    } else if (block.type === "tool_use" && typeof block.name === "string") {
      const id = typeof block.id === "string" ? block.id : `call_${turn.toolCalls.length}`;
      const args = parseToolArgs(block.input);
      callNameById.set(id, block.name);
      turn.toolCalls.push({
        id,
        name: block.name,
        args,
        canonicalArgs: JSON.stringify(canonicalizeValue(args)),
      });
    }
  }
  turn.assistantText = textParts.join("\n");
  return turn;
}

function extractAnthropicTurns(messages: unknown[]): AgentTurn[] {
  let lastHumanIdx = -1;
  for (let i = 0; i < messages.length; i++) {
    const item = messages[i];
    if (isRecord(item) && isRealAnthropicHumanMessage(item)) {
      lastHumanIdx = i;
    }
  }

  const turns: AgentTurn[] = [];
  const callNameById = new Map<string, string>();
  for (const item of messages.slice(lastHumanIdx + 1)) {
    if (!isRecord(item)) continue;
    if (item.role === "assistant") {
      turns.push(parseAnthropicAssistantTurn(item, callNameById));
    } else if (item.role === "user" && Array.isArray(item.content)) {
      const current = turns[turns.length - 1] ?? { assistantText: "", toolCalls: [], toolResults: [] };
      if (turns.length === 0) turns.push(current);
      for (const block of item.content) {
        if (!isRecord(block) || block.type !== "tool_result") continue;
        const toolCallId = typeof block.tool_use_id === "string" ? block.tool_use_id : "";
        current.toolResults.push({
          toolCallId,
          toolName: callNameById.get(toolCallId) ?? "",
          text: extractTextContent(block.content),
          explicitIsError: typeof block.is_error === "boolean" ? block.is_error : undefined,
        });
      }
    }
  }
  return turns.slice(-MAX_ROLLING_TURNS);
}

function extractOpenAiChatTurns(messages: unknown[]): AgentTurn[] {
  let lastHumanIdx = -1;
  for (let i = 0; i < messages.length; i++) {
    const item = messages[i];
    if (isRecord(item) && item.role === "user") lastHumanIdx = i;
  }

  const turns: AgentTurn[] = [];
  const callNameById = new Map<string, string>();
  for (const item of messages.slice(lastHumanIdx + 1)) {
    if (!isRecord(item)) continue;
    if (item.role === "assistant") {
      const turn: AgentTurn = { assistantText: extractTextContent(item.content), toolCalls: [], toolResults: [] };
      if (Array.isArray(item.tool_calls)) {
        for (const tc of item.tool_calls) {
          if (!isRecord(tc)) continue;
          const fn = isRecord(tc.function) ? tc.function : tc;
          const name = typeof fn.name === "string" ? fn.name : "";
          if (!name) continue;
          const id = typeof tc.id === "string" ? tc.id : `call_${turn.toolCalls.length}`;
          const args = parseToolArgs(fn.arguments);
          callNameById.set(id, name);
          turn.toolCalls.push({ id, name, args, canonicalArgs: JSON.stringify(canonicalizeValue(args)) });
        }
      }
      turns.push(turn);
    } else if (item.role === "tool") {
      const current = turns[turns.length - 1] ?? { assistantText: "", toolCalls: [], toolResults: [] };
      if (turns.length === 0) turns.push(current);
      const toolCallId = typeof item.tool_call_id === "string" ? item.tool_call_id : "";
      const nameFromMsg = typeof item.name === "string" ? item.name : "";
      current.toolResults.push({
        toolCallId,
        toolName: nameFromMsg || callNameById.get(toolCallId) || "",
        text: extractTextContent(item.content),
        explicitIsError: undefined,
      });
    }
  }
  return turns.slice(-MAX_ROLLING_TURNS);
}

function extractOpenAiResponsesTurns(input: unknown[]): AgentTurn[] {
  let lastHumanIdx = -1;
  for (let i = 0; i < input.length; i++) {
    const item = input[i];
    if (isRecord(item) && item.role === "user") lastHumanIdx = i;
  }

  const turns: AgentTurn[] = [];
  const callNameById = new Map<string, string>();
  for (const item of input.slice(lastHumanIdx + 1)) {
    if (!isRecord(item)) continue;
    if (item.role === "assistant") {
      turns.push({ assistantText: extractTextContent(item.content), toolCalls: [], toolResults: [] });
    } else if (item.type === "function_call" && typeof item.name === "string") {
      let current = turns[turns.length - 1];
      if (!current || current.toolResults.length > 0) {
        current = { assistantText: "", toolCalls: [], toolResults: [] };
        turns.push(current);
      }
      const id =
        typeof item.call_id === "string"
          ? item.call_id
          : typeof item.id === "string"
            ? item.id
            : `fc_${current.toolCalls.length}`;
      const args = parseToolArgs(item.arguments);
      callNameById.set(id, item.name);
      current.toolCalls.push({ id, name: item.name, args, canonicalArgs: JSON.stringify(canonicalizeValue(args)) });
    } else if (item.type === "function_call_output") {
      const current = turns[turns.length - 1] ?? { assistantText: "", toolCalls: [], toolResults: [] };
      if (turns.length === 0) turns.push(current);
      const toolCallId = typeof item.call_id === "string" ? item.call_id : "";
      const rawOut = item.output;
      const text =
        typeof rawOut === "string" ? rawOut : rawOut !== undefined && rawOut !== null ? JSON.stringify(rawOut) : "";
      current.toolResults.push({
        toolCallId,
        toolName: callNameById.get(toolCallId) ?? "",
        text,
        explicitIsError: undefined,
      });
    }
  }
  return turns.slice(-MAX_ROLLING_TURNS);
}

function extractTurns(body: Record<string, unknown>, protocol: InboundProtocol): AgentTurn[] {
  if (protocol === "anthropic_messages") {
    return Array.isArray(body.messages) ? extractAnthropicTurns(body.messages) : [];
  }
  if (protocol === "openai_chat") {
    return Array.isArray(body.messages) ? extractOpenAiChatTurns(body.messages) : [];
  }
  return Array.isArray(body.input) ? extractOpenAiResponsesTurns(body.input) : [];
}

function getShellCommand(call: NormalizedToolCall | undefined): string {
  if (!call) return "";
  const raw = call.args.command ?? call.args.cmd ?? call.args.CommandLine ?? call.args.script;
  return typeof raw === "string" ? raw : "";
}

function isReadOnlyTool(toolName: string): boolean {
  return READ_ONLY_TOOL_PATTERN.test(toolName.trim());
}

function isEditTool(toolName: string): boolean {
  return EDIT_TOOL_PATTERN.test(toolName.trim());
}

function isVerificationCall(call: NormalizedToolCall | undefined): boolean {
  if (!call) return false;
  if (VERIFICATION_TOOL_NAME_PATTERN.test(call.name)) return true;
  return SHELL_TOOL_PATTERN.test(call.name) && VERIFICATION_COMMAND_PATTERN.test(getShellCommand(call));
}

function isTestCall(call: NormalizedToolCall | undefined, outputText: string): boolean {
  if (call) {
    if (/test/i.test(call.name)) return true;
    if (SHELL_TOOL_PATTERN.test(call.name) && TEST_COMMAND_PATTERN.test(getShellCommand(call))) return true;
  }
  return TEST_FAILURE_OUTPUT_PATTERN.test(outputText);
}

function isTestFilePath(filePath: string): boolean {
  return TEST_FILE_PATTERN.test(filePath) || TEST_DIR_PATTERN.test(filePath) || PYTHON_TEST_FILE_PATTERN.test(filePath);
}

function normalizeFilePath(rawPath: string): string {
  return rawPath.trim().replace(/^\.\//, "");
}

function pickString(obj: Record<string, unknown>, keys: readonly string[]): string | null {
  for (const key of keys) {
    const val = obj[key];
    if (typeof val === "string") return val;
  }
  return null;
}

function extractEditOps(call: NormalizedToolCall): EditOp[] {
  if (!isEditTool(call.name)) return [];
  const rawPath = pickString(call.args, PATH_KEYS);
  if (!rawPath) return [];
  const path = normalizeFilePath(rawPath);
  if (!path) return [];

  const multi = call.args.edits ?? call.args.ReplacementChunks ?? call.args.chunks;
  if (Array.isArray(multi) && multi.length > 0) {
    const ops: EditOp[] = [];
    for (const chunk of multi) {
      if (!isRecord(chunk)) continue;
      ops.push({ path, oldStr: pickString(chunk, OLD_STR_KEYS), newStr: pickString(chunk, NEW_STR_KEYS) });
    }
    return ops.length > 0 ? ops : [{ path, oldStr: null, newStr: null }];
  }

  return [
    {
      path,
      oldStr: pickString(call.args, OLD_STR_KEYS),
      newStr: pickString(call.args, NEW_STR_KEYS) ?? pickString(call.args, CONTENT_KEYS),
    },
  ];
}

function parseAggregateErrorCount(text: string): number | null {
  if (!text) return null;
  const counts: number[] = [];
  const patterns = [
    /\b(\d+)\s+(?:tests?\s+)?(?:failed|failing)\b/gi,
    /\bfound\s+(\d+)\s+errors?\b/gi,
    /\b(\d+)\s+errors?\b/gi,
    /\b(?:errors?|failures?|failed)\s*:\s*(\d+)\b/gi,
  ];
  for (const pattern of patterns) {
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(text)) !== null) {
      const value = Number.parseInt(match[1], 10);
      if (!Number.isNaN(value)) counts.push(value);
    }
  }
  return counts.length === 0 ? null : Math.max(...counts);
}

function parseJsonErrorFlag(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed.startsWith("{")) return false;
  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (!isRecord(parsed)) return false;
    if (parsed.is_error === true) return true;
    if (typeof parsed.error === "string" && parsed.error.trim().length > 0) return true;
    if (isRecord(parsed.error)) return true;
    if (typeof parsed.exit_code === "number" && parsed.exit_code !== 0) return true;
    if (typeof parsed.exitCode === "number" && parsed.exitCode !== 0) return true;
  } catch {
    // Non-JSON payload
  }
  return false;
}

function isToolResultError(result: NormalizedToolResult, call: NormalizedToolCall | undefined): boolean {
  if (result.explicitIsError === true) return true;
  const text = result.text;
  if (!text.trim()) return false;
  if (parseJsonErrorFlag(text)) return true;

  const toolName = call?.name || result.toolName;
  if (isReadOnlyTool(toolName)) {
    return text.length < 500 && DIRECT_TOOL_ERROR_PATTERN.test(text);
  }
  if (isEditTool(toolName)) {
    return DIRECT_TOOL_ERROR_PATTERN.test(text) || NON_ZERO_EXIT_PATTERN.test(text);
  }
  if (NON_ZERO_EXIT_PATTERN.test(text)) return true;

  const aggCount = parseAggregateErrorCount(text);
  if (aggCount !== null && aggCount > 0) return true;
  return GENERIC_ERROR_PATTERN.test(text);
}

function isErrorCountTrendingDown(counts: readonly number[]): boolean {
  if (counts.length < 2) return false;
  const last = counts[counts.length - 1];
  const prev = counts[counts.length - 2];
  if (!(last < prev) || !(last < counts[0])) return false;
  if (counts.length >= 3 && !(counts[counts.length - 2] < counts[counts.length - 3])) return false;
  return true;
}

function hasErrorPingPong(errorSignatures: readonly string[]): boolean {
  if (errorSignatures.length < 3) return false;
  for (let k = 2; k < errorSignatures.length; k++) {
    const target = errorSignatures[k];
    for (let i = 0; i <= k - 2; i++) {
      if (errorSignatures[i] !== target) continue;
      for (let j = i + 1; j < k; j++) {
        if (errorSignatures[j] !== target) return true;
      }
    }
  }
  return false;
}

function countAssertions(code: string): number {
  const matches = code.match(/\b(?:expect|assert|assertEqual|assertEquals|assertTrue|assertFalse)\s*\(/g);
  return matches ? matches.length : 0;
}

function isWeakeningOrSkippingTest(ops: readonly EditOp[]): boolean {
  for (const op of ops) {
    const newStr = op.newStr ?? "";
    const oldStr = op.oldStr ?? "";
    if (!newStr) continue;
    if (TEST_SKIP_OR_WEAKEN_PATTERN.test(newStr) && !TEST_SKIP_OR_WEAKEN_PATTERN.test(oldStr)) {
      return true;
    }
    if (oldStr && countAssertions(oldStr) > countAssertions(newStr)) return true;
  }
  return false;
}

function createInspectionState(): InspectionState {
  return {
    errorSignatures: [],
    errorSigCounts: new Map(),
    aggregateErrorCounts: [],
    toolFailCounts: new Map(),
    lastVerificationFailSig: new Map(),
    verificationFailKeyCount: new Map(),
    lastVerificationEditSeq: new Map(),
    editSeq: 0,
    priorOldStringsByPath: new Map(),
    editsSinceLastPass: new Map(),
    revertOscillation: false,
    tddEvasion: false,
    hasSeenTestFailure: false,
    editedSourceSinceTestFailure: false,
    hasUnresolvedError: false,
    apologyStreak: 0,
  };
}

function processTurnEdits(
  turn: AgentTurn,
  resultByCallId: Map<string, NormalizedToolResult>,
  state: InspectionState,
): void {
  for (const call of turn.toolCalls) {
    const res = resultByCallId.get(call.id);
    if (res && isToolResultError(res, call)) continue;

    const ops = extractEditOps(call);
    if (ops.length === 0) continue;

    const touchedPath = ops[0].path;
    state.editSeq += 1;
    state.editsSinceLastPass.set(touchedPath, (state.editsSinceLastPass.get(touchedPath) ?? 0) + 1);

    if (isTestFilePath(touchedPath)) {
      if (state.hasSeenTestFailure && (isWeakeningOrSkippingTest(ops) || !state.editedSourceSinceTestFailure)) {
        state.tddEvasion = true;
      }
    } else if (state.hasSeenTestFailure) {
      state.editedSourceSinceTestFailure = true;
    }

    let seenOld = state.priorOldStringsByPath.get(touchedPath);
    if (!seenOld) {
      seenOld = new Set<string>();
      state.priorOldStringsByPath.set(touchedPath, seenOld);
    }
    for (const op of ops) {
      if (op.oldStr?.trim() && op.newStr?.trim() && op.oldStr !== op.newStr) {
        if (seenOld.has(op.newStr)) state.revertOscillation = true;
        seenOld.add(op.oldStr);
      }
    }
  }
}

function recordToolFailure(
  res: NormalizedToolResult,
  call: NormalizedToolCall | undefined,
  isVerify: boolean,
  toolKey: string | null,
  state: InspectionState,
): void {
  state.hasUnresolvedError = true;
  const aggCount = parseAggregateErrorCount(res.text);
  if (aggCount !== null && aggCount > 0) state.aggregateErrorCounts.push(aggCount);

  const sig = normalizeErrorSignature(res.text);
  if (sig) {
    state.errorSignatures.push(sig);
    state.errorSigCounts.set(sig, (state.errorSigCounts.get(sig) ?? 0) + 1);
  }

  if (isTestCall(call, res.text)) {
    state.hasSeenTestFailure = true;
    state.editedSourceSinceTestFailure = false;
  }

  if (!toolKey) return;
  if (!isVerify) {
    state.toolFailCounts.set(toolKey, (state.toolFailCounts.get(toolKey) ?? 0) + 1);
    return;
  }

  const prevEditSeq = state.lastVerificationEditSeq.get(toolKey) ?? -1;
  const prevSig = state.lastVerificationFailSig.get(toolKey);
  const progressedAfterEdit = prevEditSeq !== -1 && state.editSeq > prevEditSeq && Boolean(sig && prevSig && sig !== prevSig);
  const nextCount = progressedAfterEdit ? 1 : (state.verificationFailKeyCount.get(toolKey) ?? 0) + 1;
  state.verificationFailKeyCount.set(toolKey, nextCount);
  state.lastVerificationEditSeq.set(toolKey, state.editSeq);
  if (sig) state.lastVerificationFailSig.set(toolKey, sig);
}

function clearPostRecoveryErrorState(state: InspectionState): void {
  state.editsSinceLastPass.clear();
  state.hasSeenTestFailure = false;
  state.editedSourceSinceTestFailure = false;
  state.hasUnresolvedError = false;
  state.apologyStreak = 0;
  state.errorSignatures.length = 0;
  state.errorSigCounts.clear();
  state.aggregateErrorCounts.length = 0;
  state.revertOscillation = false;
  state.tddEvasion = false;
  state.toolFailCounts.clear();
  state.verificationFailKeyCount.clear();
  state.lastVerificationFailSig.clear();
  state.lastVerificationEditSeq.clear();
  state.priorOldStringsByPath.clear();
}

function processTurnResults(
  turn: AgentTurn,
  callById: Map<string, NormalizedToolCall>,
  state: InspectionState,
): { turnHadError: boolean; turnHadPassingVerification: boolean } {
  let turnHadError = false;
  let turnHadPassingVerification = false;

  for (const res of turn.toolResults) {
    const call = res.toolCallId ? callById.get(res.toolCallId) : undefined;
    const isVerify = isVerificationCall(call);
    const toolKey = call ? `${call.name}\0${call.canonicalArgs}` : null;

    if (isToolResultError(res, call)) {
      turnHadError = true;
      recordToolFailure(res, call, isVerify, toolKey, state);
    } else {
      if (toolKey) {
        state.toolFailCounts.delete(toolKey);
        state.verificationFailKeyCount.delete(toolKey);
      }
      if (isVerify) {
        turnHadPassingVerification = true;
      }
    }
  }

  if (turnHadPassingVerification && !turnHadError) {
    clearPostRecoveryErrorState(state);
  }

  return { turnHadError, turnHadPassingVerification };
}

function buildTelemetrySummary(
  level: StuckLevel,
  reasons: readonly string[],
  turnCount: number,
  metrics: Omit<StuckLoopReport, "level" | "reasons" | "telemetrySummary">,
  topErrorSig: string | null,
): string {
  const parts: string[] = [`level=${level}`, `turns=${turnCount}`];
  if (reasons.length > 0) parts.push(`reasons=${reasons.join(",")}`);
  if (metrics.identicalErrorCount > 0) parts.push(`idErr=${metrics.identicalErrorCount}`);
  if (metrics.identicalToolFailCount > 0) parts.push(`idToolFail=${metrics.identicalToolFailCount}`);
  if (metrics.revertOscillation) parts.push("revert=true");
  if (metrics.tddEvasion) parts.push("tddEvasion=true");
  if (metrics.hotspotEdits > 0) parts.push(`hotspot=${metrics.hotspotEdits}`);
  if (metrics.apologyStreak > 0) parts.push(`apology=${metrics.apologyStreak}`);
  if (metrics.errorCountTrendingDown) parts.push("trendingDown=true");

  let summary = parts.join(" ");
  if (topErrorSig && summary.length < MAX_TELEMETRY_CHARS - 20) {
    const budget = MAX_TELEMETRY_CHARS - summary.length - 10;
    const clipped = topErrorSig.length > budget ? `${topErrorSig.slice(0, Math.max(0, budget - 3))}...` : topErrorSig;
    summary = `${summary} topErr="${clipped}"`;
  }
  return summary.length > MAX_TELEMETRY_CHARS ? `${summary.slice(0, MAX_TELEMETRY_CHARS - 3)}...` : summary;
}

function emptyReport(): StuckLoopReport {
  return {
    level: "none",
    reasons: [],
    identicalErrorCount: 0,
    identicalToolFailCount: 0,
    revertOscillation: false,
    tddEvasion: false,
    apologyStreak: 0,
    hotspotEdits: 0,
    errorCountTrendingDown: false,
    telemetrySummary: "level=none turns=0",
  };
}

function finalizeReport(state: InspectionState, turnCount: number): StuckLoopReport {
  const trendingDown = isErrorCountTrendingDown(state.aggregateErrorCounts);

  let identicalErrorCount = 0;
  let topErrorSig: string | null = null;
  for (const [sig, count] of state.errorSigCounts) {
    if (count > identicalErrorCount) {
      identicalErrorCount = count;
      topErrorSig = sig;
    }
  }
  if (!topErrorSig && state.errorSignatures.length > 0) {
    topErrorSig = state.errorSignatures[state.errorSignatures.length - 1];
  }

  let identicalToolFailCount = 0;
  for (const count of state.toolFailCounts.values()) {
    if (count > identicalToolFailCount) identicalToolFailCount = count;
  }
  if (!trendingDown) {
    for (const count of state.verificationFailKeyCount.values()) {
      if (count > identicalToolFailCount) identicalToolFailCount = count;
    }
  }

  let hotspotEdits = 0;
  for (const editCount of state.editsSinceLastPass.values()) {
    if (editCount - 1 > hotspotEdits) hotspotEdits = editCount - 1;
  }

  const hardReasons: string[] = [];
  if (!trendingDown && identicalErrorCount >= 3) hardReasons.push(`identical_error:${identicalErrorCount}`);
  if (identicalToolFailCount >= 3) hardReasons.push(`identical_tool_fail:${identicalToolFailCount}`);
  if (state.revertOscillation) hardReasons.push("revert_oscillation");

  const softReasons: string[] = [];
  if (!trendingDown && hasErrorPingPong(state.errorSignatures)) softReasons.push("error_ping_pong");
  if (hotspotEdits >= 2) softReasons.push(`hotspot_edits:${hotspotEdits}`);
  if (state.apologyStreak >= 2) softReasons.push(`apology_streak:${state.apologyStreak}`);
  if (state.tddEvasion) softReasons.push("tdd_evasion");

  const level: StuckLevel = hardReasons.length > 0 ? "hard" : softReasons.length > 0 ? "soft" : "none";
  const reasons = [...hardReasons, ...softReasons];
  const metrics = {
    identicalErrorCount,
    identicalToolFailCount,
    revertOscillation: state.revertOscillation,
    tddEvasion: state.tddEvasion,
    apologyStreak: state.apologyStreak,
    hotspotEdits,
    errorCountTrendingDown: trendingDown,
  };

  return {
    level,
    reasons,
    ...metrics,
    telemetrySummary: buildTelemetrySummary(level, reasons, turnCount, metrics, level === "none" ? null : topErrorSig),
  };
}

/**
 * Inspects recent agent/tool turns after the last real human user prompt to detect hard or soft
 * stuck-loop patterns across Anthropic Messages, OpenAI Chat, and OpenAI Responses formats.
 */
export function inspectStuckLoop(body: Record<string, unknown>, protocol: InboundProtocol): StuckLoopReport {
  if (!isRecord(body)) return emptyReport();
  const turns = extractTurns(body, protocol);
  if (turns.length === 0) return emptyReport();

  const callById = new Map<string, NormalizedToolCall>();
  for (const turn of turns) {
    for (const call of turn.toolCalls) callById.set(call.id, call);
  }

  const state = createInspectionState();
  for (const turn of turns) {
    const resultByCallId = new Map<string, NormalizedToolResult>();
    for (const res of turn.toolResults) {
      if (res.toolCallId) resultByCallId.set(res.toolCallId, res);
    }

    processTurnEdits(turn, resultByCallId, state);
    const { turnHadError, turnHadPassingVerification } = processTurnResults(turn, callById, state);

    if (turn.assistantText.trim().length > 0) {
      if (APOLOGY_MARKER_PATTERN.test(turn.assistantText)) {
        if (state.hasUnresolvedError || turnHadError) state.apologyStreak += 1;
      } else if (!state.hasUnresolvedError && !turnHadError && turnHadPassingVerification) {
        state.apologyStreak = 0;
      }
    }
  }

  return finalizeReport(state, turns.length);
}

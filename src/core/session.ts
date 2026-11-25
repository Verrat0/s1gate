import type { CandidateModel, S1GateRequest, SessionPin } from "../types.js";

const DEFAULT_TTL_MS = 2 * 60 * 60 * 1000;
const DEFAULT_MAX_SESSIONS = 5000;
const COMPACTION_MIN_TOKENS = 8000;
const COMPACTION_RATIO = 0.5;
const CACHE_GUARD_MIN_TOKENS = 12_000;
const DEFAULT_EXPECTED_REMAINING_TURNS = 4;

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const textEncoder = new TextEncoder();

function rotr(value: number, bits: number): number {
  return (value >>> bits) | (value << (32 - bits));
}

function sha256Hex32(input: string): string {
  const bytes = textEncoder.encode(input);
  const bitLen = bytes.length * 8;
  const paddedLen = Math.ceil((bytes.length + 9) / 64) * 64;
  const buffer = new Uint8Array(paddedLen);
  buffer.set(bytes);
  buffer[bytes.length] = 0x80;

  const view = new DataView(buffer.buffer);
  view.setUint32(paddedLen - 8, Math.floor(bitLen / 0x100000000), false);
  view.setUint32(paddedLen - 4, bitLen >>> 0, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);

  for (let offset = 0; offset < paddedLen; offset += 64) {
    for (let i = 0; i < 16; i++) {
      w[i] = view.getUint32(offset + i * 4, false);
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i++) {
      const s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + s1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (s0 + maj) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return [h0, h1, h2, h3]
    .map((word) => word.toString(16).padStart(8, "0"))
    .join("");
}

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

function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === "string") {
      parts.push(block);
    } else if (isRecord(block)) {
      if (typeof block.text === "string") parts.push(block.text);
      else if (typeof block.content === "string") parts.push(block.content);
    }
  }
  return parts.join("\n");
}

function extractSystemAndFirstUser(body: Record<string, unknown>): {
  systemText: string;
  firstUserText: string;
} {
  const systemParts: string[] = [];
  if (body.system !== undefined) {
    const sys = extractText(body.system);
    if (sys) systemParts.push(sys);
  }
  if (typeof body.instructions === "string" && body.instructions.length > 0) {
    systemParts.push(body.instructions);
  }

  const items: Record<string, unknown>[] = Array.isArray(body.messages)
    ? body.messages.filter(isRecord)
    : Array.isArray(body.input)
      ? body.input.filter(isRecord)
      : typeof body.input === "string"
        ? [{ role: "user", content: body.input }]
        : [];

  let firstUserText = "";
  for (const item of items) {
    const role = typeof item.role === "string" ? item.role.toLowerCase() : "";
    if (role === "system" || role === "developer") {
      const txt = extractText(item.content);
      if (txt) systemParts.push(txt);
    } else if (role === "user" && !firstUserText) {
      firstUserText = extractText(item.content);
    }
  }

  return {
    systemText: systemParts.join("\n"),
    firstUserText,
  };
}

function resolveBaseSeed(
  headers: Map<string, string>,
  body: Record<string, unknown>,
  firstUserText: string,
  systemText: string,
): string {
  const claudeSession = headers.get("x-claude-code-session-id");
  if (claudeSession) return `cc:${claudeSession}`;

  const metadataUserId =
    isRecord(body.metadata) &&
    typeof body.metadata.user_id === "string" &&
    body.metadata.user_id.trim().length > 0
      ? body.metadata.user_id.trim()
      : undefined;

  if (metadataUserId && metadataUserId.includes("_session_")) {
    return `uid:${metadataUserId}`;
  }

  const codexParent = headers.get("x-codex-parent-thread-id");
  const threadId = headers.get("thread-id");
  if (codexParent && threadId) return `codex:${codexParent}:${threadId}`;
  if (threadId) return `thread:${threadId}`;

  const sessionId = headers.get("session_id") ?? headers.get("session-id");
  if (sessionId) return `sid:${sessionId}`;

  if (
    typeof body.prompt_cache_key === "string" &&
    body.prompt_cache_key.trim().length > 0
  ) {
    return `pck:${body.prompt_cache_key.trim()}`;
  }

  const opencodeSession = headers.get("x-opencode-session");
  if (opencodeSession) return `oc:${opencodeSession}`;

  const xSessionId = headers.get("x-session-id");
  if (xSessionId) return `xsid:${xSessionId}`;

  if (metadataUserId) {
    return `uid:${metadataUserId}`;
  }

  return `fallback:${systemText}\0${firstUserText}`;
}

export function resolveSessionKey(
  req: S1GateRequest,
  isSubagent: boolean,
): string {
  const headers = normalizeHeaders(req.headers);
  const body = isRecord(req.body) ? req.body : {};
  const { systemText, firstUserText } = extractSystemAndFirstUser(body);

  const baseSeed = resolveBaseSeed(headers, body, firstUserText, systemText);
  if (!isSubagent) {
    return sha256Hex32(baseSeed);
  }

  const subagentDiscriminator =
    headers.get("x-claude-code-agent-id") ??
    headers.get("thread-id") ??
    sha256Hex32(`first-user:${firstUserText}`);

  return sha256Hex32(`${baseSeed}::subagent::${subagentDiscriminator}`);
}

export class SessionPinStore {
  private readonly ttlMs: number;
  private readonly maxSessions: number;
  private readonly store = new Map<string, SessionPin>();

  constructor(opts?: { ttlMs?: number; maxSessions?: number }) {
    this.ttlMs =
      opts?.ttlMs !== undefined && Number.isFinite(opts.ttlMs) && opts.ttlMs > 0
        ? opts.ttlMs
        : DEFAULT_TTL_MS;
    this.maxSessions =
      opts?.maxSessions !== undefined &&
      Number.isFinite(opts.maxSessions) &&
      opts.maxSessions >= 1
        ? Math.floor(opts.maxSessions)
        : DEFAULT_MAX_SESSIONS;
  }

  get(key: string, nowMs: number = Date.now()): SessionPin | undefined {
    const existing = this.store.get(key);
    if (!existing) return undefined;

    if (nowMs - existing.lastUsedAtMs > this.ttlMs) {
      this.store.delete(key);
      return undefined;
    }

    // Refresh LRU recency by moving entry to the end of Map insertion order.
    this.store.delete(key);
    this.store.set(key, existing);
    return { ...existing };
  }

  set(pin: SessionPin): void {
    const copy: SessionPin = { ...pin };
    if (this.store.has(copy.sessionKey)) {
      this.store.delete(copy.sessionKey);
    }
    this.store.set(copy.sessionKey, copy);

    while (this.store.size > this.maxSessions) {
      const oldestKey = this.store.keys().next().value;
      if (oldestKey === undefined) break;
      this.store.delete(oldestKey);
    }
  }

  delete(key: string): boolean {
    return this.store.delete(key);
  }

  clear(): void {
    this.store.clear();
  }

  size(): number {
    return this.store.size;
  }

  didCompact(pin: SessionPin, currentEstimatedTokens: number): boolean {
    return (
      pin.lastEstimatedTokens >= COMPACTION_MIN_TOKENS &&
      currentEstimatedTokens < pin.lastEstimatedTokens * COMPACTION_RATIO
    );
  }
}

export function shouldAllowMidSessionSwitch(opts: {
  currentModel: CandidateModel;
  nextModel: CandidateModel;
  currentContextTokens: number;
  expectedRemainingTurns?: number;
  hasThinkingSignature: boolean;
  isCompactionBoundary: boolean;
  isHardEscalation: boolean;
}): { allow: boolean; reason: string } {
  if (opts.currentModel.proxyId === opts.nextModel.proxyId) {
    return { allow: true, reason: "same_model" };
  }

  if (opts.isCompactionBoundary) {
    return { allow: true, reason: "compaction_boundary" };
  }

  if (opts.isHardEscalation) {
    return { allow: true, reason: "hard_escalation" };
  }

  if (opts.hasThinkingSignature && opts.currentModel.requiresThinkingSignature) {
    return { allow: false, reason: "thinking_signature_lock" };
  }

  const isCostDowngrade =
    opts.nextModel.inputPricePerMTok < opts.currentModel.inputPricePerMTok;

  if (isCostDowngrade && opts.currentContextTokens >= CACHE_GUARD_MIN_TOKENS) {
    const turns =
      opts.expectedRemainingTurns !== undefined &&
      Number.isFinite(opts.expectedRemainingTurns) &&
      opts.expectedRemainingTurns >= 0
        ? opts.expectedRemainingTurns
        : DEFAULT_EXPECTED_REMAINING_TURNS;

    const ctxMTok = opts.currentContextTokens / 1_000_000;
    const cacheRewriteTax =
      ctxMTok *
      opts.nextModel.inputPricePerMTok *
      opts.nextModel.cacheWriteMultiplier;
    const perTurnSavingsPerMTok = Math.max(
      0,
      opts.currentModel.inputPricePerMTok * opts.currentModel.cacheReadMultiplier -
        opts.nextModel.inputPricePerMTok * opts.nextModel.cacheReadMultiplier,
    );
    const expectedSavings = turns * ctxMTok * perTurnSavingsPerMTok;

    if (cacheRewriteTax > expectedSavings) {
      return { allow: false, reason: "cache_breakeven_guard" };
    }
    return { allow: true, reason: "downgrade_breakeven_passed" };
  }

  return { allow: true, reason: "allowed" };
}

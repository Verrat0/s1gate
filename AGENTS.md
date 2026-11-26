# AGENTS.md — `s1gate` Engineering & Architecture Guidelines

`s1gate` is an open-source, **proxy-agnostic System-1 cognitive model routing engine**. Every change in this repository must preserve multi-router compatibility, pure Web-Standard core portability, deterministic benchmark integrity, and strict security/privacy boundaries.

---

## 1. Proxy-Agnostic Engine First (Non-Negotiable)

- **`s1gate` is an engine, never a single-proxy plugin.** Never hardcode assumptions, required headers, or proprietary payload schemas that only one proxy (e.g., OpenCodex/`ocx`, 9Router, LiteLLM, One-API, New-API, Bifrost, Portkey, or OpenRouter) emits.
- **Additive, optional signal extraction only:**
  - Any convenience parser for a router's wire conventions (e.g., effort suffixes like `:none`/`:high` or `:free` in `src/core/catalog.ts`, structured `state` fields like `signals.estimated_tokens` / `has_image` / `tool_history`, or advisory quota text in `POST /v1/systemone`) MUST be strictly **optional** and degrade gracefully when omitted.
  - Plain string `state` on `POST /v1/systemone`, standard OpenAI/Anthropic payloads on `POST /v1/route`, and bare or `provider/model` slugs must always work without router-specific metadata.
- **Standard, vendor-neutral contracts:**
  - Prefer generic fields and headers (`S1GateRequest.estimatedContextTokens`, `X-S1Gate-Context-Tokens`, `X-Router-*`, `protocol: "anthropic_messages" | "openai_responses" | "openai_chat"`, `harness: "claude_code" | "codex" | "aider" | "opencode" | "cursor" | "generic"`).
  - Never add proxy-specific runtime dependencies or import proxy internals.

---

## 2. Runtime & Packaging Boundaries

- **Pure Web-Standard `s1gate/core` (`src/core/*`):**
  - **ZERO `node:*` imports** and **ZERO Node-only globals** (`Buffer`, `process`, `__dirname`, `require`) in `src/core/*`.
  - Must run unmodified on Node 20+, Bun, Deno, Cloudflare Workers, and Vercel Edge (`fetch`, `crypto.subtle`, `AbortSignal`, `TextEncoder`).
- **Node-only code stays in `src/server/*`, `src/replay/*`, `src/cli/*`, and `scripts/*`.**
- **Zero runtime npm dependencies:** Keep `"dependencies"` empty in `package.json`; keep `bin.s1gate` normalized (`"dist/cli/index.js"`, no leading `./`).

---

## 3. Scoring & Benchmark Integrity (`llm-stats.com`)

- **Attribution:** Always preserve `X-Router-Benchmark-Source: llm-stats.com` and snapshot attribution metadata.
- **No-Trap Safeguards (never weaken):**
  - Multi-benchmark Logit-OLS + LCB pooling strictly requires `rAdj >= 0.30` — never force-pool uncorrelated benchmarks.
  - Keep the Sibling Red-Flag Penalty, No-Evidence Gate (`balanced`/`quality`), and Unpriced `$0` Snapshot Flagship Guard (`hasKnownPricing` / `isExplicitlyFreeModel`).
  - Keep 90% context-window headroom aligned across sticky-pin invalidation (`contextWindow * 9 < effectiveContextTokens * 10`) and Stage 2 hard eligibility (`Math.ceil((effectiveContextTokens * 10) / 9)`).

---

## 4. Security, Privacy & Hygiene Invariants

- **One-way redacted egress (`src/core/envelope.ts`):** Strip code fences (backtick, tilde, unclosed), secrets/tokens/PEM keys/DB URIs, and PII before calling any external System-1 backend. Never bypass redaction with unsanitized caller fields (e.g., cap/sanitize `tool_call.name` before envelope assembly).
- **HTTP boundary hardening (`src/server/http-bridge.ts` & `src/core/engine.ts`):**
  - Strip `authorization`, `cookie`, `x-api-key`, and `proxy-authorization` on both incoming HTTP headers and `payload.headers`.
  - Guard all dynamic object lookups against `__proto__`, `constructor`, and `prototype`.
  - Sanitize every outgoing header value (printable ASCII, <= 256 chars) and parse numeric headers strictly (e.g., base-10 `/^\d+$/` for `X-S1Gate-Context-Tokens`).
- **No personal paths or literal secret patterns in repo:** Never commit `/Users/<name>/...` paths or raw secret-scanner token literals in tests (assemble synthetic test tokens dynamically at runtime).

---

## 5. Verification Before Completing Any Change

Always run and confirm all four checks pass after edits:

```bash
npm run prepublishOnly    # tsc --noEmit + tsup build + vitest run
npm run test:coverage     # >= 80% lines/functions/statements
npm publish --dry-run     # 0 package.json warnings, clean tarball contents
```

Verify static import purity on `dist/core/index.js` (0 reachable `node:*` imports) whenever touching `src/core/*`.

# Security Policy

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for undisclosed security vulnerabilities.

Instead, report vulnerabilities privately via GitHub Security Advisories:

- **Preferred:** Open a private advisory at [github.com/albifhrzq/s1gate/security/advisories/new](https://github.com/albifhrzq/s1gate/security/advisories/new)

Include a clear description of the issue, reproduction steps (or a minimal PoC), affected version(s), and any suggested mitigation. We aim to acknowledge reports within 72 hours and coordinate a fix before public disclosure.

For non-security bugs or feature requests, please use regular [GitHub Issues](https://github.com/albifhrzq/s1gate/issues).

## Security Model

`s1gate` is designed to make routing decisions with minimal exposure of sensitive prompt data and credentials:

### 1. One-Way Redacted Egress (`src/core/envelope.ts`)

When an external System-1 classifier backend (e.g., Jev/tev1/clef) is configured, `buildRoutingEnvelope` transforms the request into a compact, redacted routing summary (capped at 8 KiB UTF-8) before any outbound request is sent:

- **Code blocks:** Closed backtick fences, tilde fences, and unclosed trailing fences are replaced with structural stubs (`[CODE: <lang>, <n> lines]`).
- **Secrets & credentials:** Private key PEM blocks, database connection URIs, `Authorization: Basic`, `Bearer` tokens, JWTs, common provider API keys (OpenAI, Anthropic, GitHub, AWS, Google, Slack, Stripe, npm, Hugging Face), and generic secret assignments are replaced with `[REDACTED:SECRET]`.
- **PII:** Emails, URLs, IPv4/IPv6 addresses, phone numbers, US SSNs, 16-digit card/ID numbers, and medical record identifiers (`MRN`, `RM`, `Patient ID`) are replaced with `[REDACTED:PII]`.
- **Tool metadata:** Only tool names (capped at 12) are included — tool descriptions and parameter schemas are omitted.

### 2. Zero-Network Local Mode (`--local-only`)

Running `s1gate` with `--local-only` (or without configuring a remote System-1 backend) disables all outbound network calls and routes purely using the built-in local heuristic classifier and bundled `llm-stats.com` benchmark snapshot.

### 3. HTTP Bridge Hardening (`src/server/http-bridge.ts`)

- **Credential header stripping:** Incoming `authorization`, `cookie`, `x-api-key`, and `proxy-authorization` headers are stripped by design from both HTTP request headers and `payload.headers` before entering the routing engine.
- **Prototype-pollution guards:** Dynamic object keys (`__proto__`, `constructor`, `prototype`) are ignored across request parsing and response building.
- **Resource & header limits:** Binds to `127.0.0.1` by default, enforces strict JSON payload size limits (1 MiB on `/v1/route`, 128 KiB on `/v1/systemone`) and HTTP timeouts, validates numeric headers strictly, and sanitizes all outgoing `X-Router-*` header values to printable ASCII (<= 256 chars).

## Scope & Non-Goals

- **Not a secret scanner or DLP engine:** Regex-based redaction in `envelope.ts` is defense-in-depth for routing telemetry, not a cryptographic guarantee against arbitrary, custom, or obfuscated secrets. Workloads with strict zero-egress compliance requirements should run `--local-only` or point `s1gate` at a self-hosted System-1 backend.
- **Not an auth/edge proxy:** `s1gate` is a routing decision engine. Because `http-bridge.ts` strips authentication headers by design, callers and upstream proxies remain responsible for authentication, authorization, TLS termination, rate limiting, and network isolation. Do not expose the HTTP bridge directly to untrusted public networks without an authenticated reverse proxy in front of it.

# Contributing to `s1gate`

Thanks for your interest in contributing to `s1gate`! Please read [AGENTS.md](./AGENTS.md) before opening a pull request — it defines the architecture invariants that every change must preserve.

## Prerequisites & Setup

- **Node.js >= 20** (CI verifies Node 20 and 22)
- **npm** (comes with Node.js)

```bash
git clone https://github.com/albifhrzq/s1gate.git
cd s1gate
npm install
```

## Development Commands

```bash
npm run typecheck         # TypeScript check (tsc --noEmit)
npm run build             # Bundle ESM + types with tsup
npm test                  # Run the Vitest test suite
npm run test:coverage     # Run tests with >= 80% coverage thresholds
npm run prepublishOnly    # Full check: typecheck + build + test
npm publish --dry-run     # Verify clean package tarball & zero warnings
```

## Architecture Boundaries

Full engineering rules live in [AGENTS.md](./AGENTS.md). Keep these core constraints in mind:

- **Pure Web-Standard core (`src/core/*`):** Zero `node:*` imports and zero Node-only globals (`Buffer`, `process`, `__dirname`, `require`). `s1gate/core` must run unmodified across Node 20+, Bun, Deno, Cloudflare Workers, and Vercel Edge.
- **Node-only modules:** Filesystem, HTTP server, replay, and CLI code belong strictly under `src/server/*`, `src/replay/*`, `src/cli/*`, and `scripts/*`.
- **Zero runtime dependencies:** Keep `"dependencies"` empty in `package.json`.
- **Proxy-agnostic by design:** Never hardcode assumptions or required headers specific to a single router/proxy. Optional signal parsers must degrade gracefully to standard OpenAI/Anthropic payloads and bare model slugs.
- **Benchmark & security invariants:** Preserve `llm-stats.com` attribution, No-Trap scoring safeguards, one-way redacted egress (`src/core/envelope.ts`), and HTTP header stripping (`src/server/http-bridge.ts`).

## Pull Request Checklist

Before submitting a PR, verify that:

- [ ] `npm run prepublishOnly` passes cleanly.
- [ ] `npm run test:coverage` meets or exceeds the 80% line/function/statement thresholds.
- [ ] `dist/core/index.js` has zero reachable `node:*` imports (verified automatically in CI).
- [ ] `npm publish --dry-run` succeeds with zero `package.json` warnings and no stray files.
- [ ] Tests cover new behavior and edge cases, with no personal paths or raw secret-scanner token literals committed (build synthetic tokens dynamically in tests).
- [ ] Changes keep `s1gate` proxy-agnostic and preserve existing public contracts.

## Ideas for First Contributions

- **SDK & client adapters:** Lightweight zero-dependency wrappers or examples for OpenAI SDK, Anthropic SDK, Vercel AI SDK, Cloudflare Workers, or Deno.
- **Domain presets:** New or refined domain definitions and benchmark chains in `data/domains.json` with accompanying tests.
- **Evaluation prompts:** Public labelled routing prompts and replay fixtures covering diverse coding, debugging, reasoning, and non-coding tasks.
- **CLI & replay coverage:** Additional unit/integration tests for CLI flags, replay reports, and edge-case payloads.

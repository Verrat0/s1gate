# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Demo replay workflow and public labelled evaluation suite.
- `recommendedEffort` output on routing decisions for downstream reasoning-effort tuning.
- `createRoutingFetch` Web-Standard `fetch` wrapper for drop-in SDK integration.
- Repository hygiene documentation: `CONTRIBUTING.md`, `SECURITY.md`, and `CHANGELOG.md`.
- CI enforcement of 80% coverage thresholds (`npm run test:coverage`), `dist/core/index.js` Web-Standard import purity verification, and `npm publish --dry-run` packaging checks.

### Changed
- Genericized the `healthcare` domain description in `data/domains.json` (`EMR/SIMRS` -> `EMR/EHR`).
- Removed redundant `/\b\d{16}\b/g` PII regex in `src/core/envelope.ts` (already covered by `/\b(?:\d{4}[-\s]?){3}\d{4}\b/g`).

### Fixed
- Aligned CLI `--version` output with `package.json`.

## [0.1.1] - 2026-11-26

### Added
- Dynamic context-window hints (`estimatedContextTokens` and `X-S1Gate-Context-Tokens`) with consistent 90% context-window headroom across sticky-pin invalidation and Stage 2 eligibility.
- Optional structured `state` signal extraction (`signals.estimated_tokens`, `has_image`, `tool_history`, and advisory quota text) on `POST /v1/systemone`.
- Model variant suffix normalization in `src/core/catalog.ts`.
- Engineering and architecture guidelines in `AGENTS.md`.
- npm, CI, and MIT license badges in `README.md`.

### Changed
- Published package under the scoped npm name `@albifhrzq/s1gate`.

## [0.1.0] - 2026-11-25

### Added
- Initial release of `s1gate`, a proxy-agnostic System-1 cognitive model routing engine.
- Pure Web-Standard `s1gate/core` (`src/core/*`) with zero runtime dependencies.
- One-way redacted routing envelope (`src/core/envelope.ts`) scrubbing code fences, secrets, and PII before external System-1 classification.
- System-1 classifier client (Jev/tev1/clef) with local heuristic fallback (`--local-only` zero-network mode).
- Deterministic `llm-stats.com` benchmark scoring with multi-benchmark Logit-OLS + LCB pooling and No-Trap safeguards.
- KV-cache sticky session pinning, ratchet locks, and stuck-loop escalation.
- Node HTTP bridge (`src/server/http-bridge.ts`) serving `/health`, `/v1/route`, and `/v1/systemone`.
- CLI (`s1gate`) and offline session replay harness (`src/replay/*`).

[Unreleased]: https://github.com/albifhrzq/s1gate/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/albifhrzq/s1gate/compare/01cd9ab...v0.1.1
[0.1.0]: https://github.com/albifhrzq/s1gate/commits/01cd9ab

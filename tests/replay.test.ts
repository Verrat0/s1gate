import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  loadReplaySessionsFromSignalsFile,
  runCounterfactualReplay,
  type ReplaySessionInput,
} from "../src/replay/replay.js";
import type { SystemOneSignals } from "../src/types.js";

const REAL_OCX_POOL = [
  "anthropic/claude-fable-5-2",
  "anthropic/claude-opus-5-5",
  "anthropic/claude-sonnet-5-5",
  "anthropic/claude-haiku-5-5",
  "openai/gpt-6.2-sol",
  "openai/gpt-6.1-terra",
  "openai/gpt-6-luna",
  "google-antigravity/gemini-4-neon",
  "google-antigravity/gemini-4-hydrogen",
  "ollama-cloud/beam",
  "ollama-cloud/deepseek-v4.1-flash",
  "ollama-cloud/kimi-k3.1-pro",
  "ollama-cloud/nemotron-3-ultra",
  "ollama-cloud/hy4",
  "ollama-cloud/mimo-v2.6-flash",
  "ollama-cloud/minimax-m3.1-flash",
];

function makeSignals(overrides: Partial<SystemOneSignals> = {}): SystemOneSignals {
  const domain = overrides.domain ?? "software_engineering";
  return {
    domain,
    domainProbabilities: overrides.domainProbabilities ?? { [domain]: 0.9, general_chat: 0.1 },
    domainConfidence: 0.9,
    complexity: 0.5,
    needsFlagshipOrchestrator: 0.2,
    backend: "clef:27b",
    latencyMs: 420,
    fromFallback: false,
    ...overrides,
  };
}

function makeCodingSession(
  id: string,
  domain: string,
  complexity: number,
  opts: { isSubagent?: boolean; noul?: number; turnCount?: number } = {},
): ReplaySessionInput {
  const turnCount = opts.turnCount ?? 10;
  const turns: ReplaySessionInput["turns"] = [];
  for (let i = 0; i < turnCount; i++) {
    turns.push({
      actualModel: i % 2 === 0 ? "claude-opus-5-5" : "claude-sonnet-5-5",
      inputTokens: i === 0 ? 1_500 : 500,
      cacheReadTokens: i === 0 ? 0 : 30_000 + i * 3_000,
      cacheWriteTokens: i === 0 ? 30_000 : 3_000,
      outputTokens: 600,
      toolUses: i % 2 === 0 ? 2 : 1,
      toolErrors: i === 4 ? 1 : 0,
    });
  }
  return {
    id,
    project: "test-proj",
    isSubagent: opts.isSubagent ?? false,
    hasTools: true,
    signals: makeSignals({
      domain,
      domainProbabilities: { [domain]: 0.85, software_engineering: 0.15 },
      complexity,
      needsFlagshipOrchestrator: opts.noul ?? 0.15,
    }),
    turns,
  };
}

describe("runCounterfactualReplay", () => {
  it("excludes hy4-flash and mimo-v2.6-flash from winning balanced profile across diverse sessions", () => {
    const sessions: ReplaySessionInput[] = [
      makeCodingSession("s-se-mid", "software_engineering", 0.55),
      makeCodingSession("s-se-sub", "software_engineering", 0.4, { isSubagent: true }),
      makeCodingSession("s-dev-mid", "terminal_devops", 0.5),
      makeCodingSession("s-agent-mid", "agentic_tools", 0.55),
      makeCodingSession("s-arch-hi", "research_architecture", 0.82, { noul: 0.85 }),
      makeCodingSession("s-fe-mid", "frontend_web", 0.45),
      makeCodingSession("s-health", "healthcare", 0.65),
      makeCodingSession("s-fin", "finance_data", 0.6),
    ];

    const report = runCounterfactualReplay({
      sessions,
      availableModels: REAL_OCX_POOL,
      flagshipModel: "anthropic/claude-opus-5-5",
    });

    const balancedWinners = Object.keys(report.profiles.balanced.modelMix);
    expect(balancedWinners.length).toBeGreaterThan(0);

    for (const winner of balancedWinners) {
      expect(winner.toLowerCase()).not.toContain("hy4");
      expect(winner.toLowerCase()).not.toContain("mimo-v2.6-flash");
    }

    const totalBalancedSessions = Object.values(report.profiles.balanced.modelMix).reduce(
      (sum, count) => sum + count,
      0,
    );
    expect(totalBalancedSessions).toBe(sessions.length);
  });

  it("computes accurate cache-read share, savings percentages, and profile cost ordering", () => {
    const sessions: ReplaySessionInput[] = [
      makeCodingSession("s1", "software_engineering", 0.45, { turnCount: 12 }),
      makeCodingSession("s2", "terminal_devops", 0.5, { turnCount: 12 }),
      makeCodingSession("s3", "agentic_tools", 0.4, { isSubagent: true, turnCount: 8 }),
      makeCodingSession("s4", "research_architecture", 0.85, { noul: 0.8, turnCount: 6 }),
    ];

    const report = runCounterfactualReplay({
      sessions,
      availableModels: REAL_OCX_POOL,
      flagshipModel: "anthropic/claude-opus-5-5",
    });

    expect(report.sessionsCount).toBe(4);
    expect(report.totalTurns).toBe(38);
    expect(report.cacheReadSharePct).toBeGreaterThan(75);
    expect(report.cacheReadSharePct).toBeLessThan(100);

    expect(report.flagshipCostUsd).toBeGreaterThan(report.actualCostUsd);
    expect(report.profiles.balanced.totalCostUsd).toBeLessThan(report.flagshipCostUsd);
    expect(report.profiles.frugal.totalCostUsd).toBeLessThanOrEqual(
      report.profiles.balanced.totalCostUsd,
    );
    expect(report.profiles.balanced.totalCostUsd).toBeLessThanOrEqual(
      report.profiles.quality.totalCostUsd,
    );

    const expectedBalSavedVsFlagship =
      ((report.flagshipCostUsd - report.profiles.balanced.totalCostUsd) /
        report.flagshipCostUsd) *
      100;
    expect(report.profiles.balanced.savedVsFlagshipPct).toBeCloseTo(
      expectedBalSavedVsFlagship,
      1,
    );

    const expectedBalSavedVsActual =
      ((report.actualCostUsd - report.profiles.balanced.totalCostUsd) / report.actualCostUsd) *
      100;
    expect(report.profiles.balanced.savedVsActualPct).toBeCloseTo(expectedBalSavedVsActual, 1);

    // Naive per-turn switching incurs cache-miss penalties and costs significantly more than sticky balanced
    expect(report.naivePerTurnCostUsd).toBeGreaterThan(report.profiles.balanced.totalCostUsd);
  });

  it("handles empty session lists cleanly and orders modelMix descending by session count", () => {
    const empty = runCounterfactualReplay({
      sessions: [],
      availableModels: REAL_OCX_POOL,
    });
    expect(empty.sessionsCount).toBe(0);
    expect(empty.totalTurns).toBe(0);
    expect(empty.flagshipCostUsd).toBe(0);
    expect(empty.profiles.balanced.modelMix).toEqual({});

    const sessions = [
      makeCodingSession("a", "software_engineering", 0.5),
      makeCodingSession("b", "software_engineering", 0.5),
      makeCodingSession("c", "software_engineering", 0.5),
      makeCodingSession("d", "research_architecture", 0.9, { noul: 0.9 }),
    ];
    const report = runCounterfactualReplay({
      sessions,
      availableModels: REAL_OCX_POOL,
    });

    const counts = Object.values(report.profiles.balanced.modelMix);
    for (let i = 1; i < counts.length; i++) {
      expect(counts[i - 1]).toBeGreaterThanOrEqual(counts[i]);
    }
  });

  it("routes short non-tool auxiliary turns to auxiliary model without changing session pin", () => {
    const auxSession: ReplaySessionInput = {
      id: "s-aux",
      isSubagent: false,
      hasTools: false,
      signals: makeSignals({ domain: "research_architecture", complexity: 0.95, needsFlagshipOrchestrator: 0.95 }),
      turns: [
        {
          actualModel: "claude-opus-5-5",
          inputTokens: 400,
          cacheReadTokens: 0,
          cacheWriteTokens: 0,
          outputTokens: 50,
          toolUses: 0,
          toolErrors: 0,
        },
      ],
    };

    const report = runCounterfactualReplay({
      sessions: [auxSession],
      availableModels: ["anthropic/claude-opus-5-5", "anthropic/claude-haiku-5-5"],
      flagshipModel: "anthropic/claude-opus-5-5",
    });

    // Even in quality profile, the auxiliary turn uses Haiku instead of Opus
    expect(report.profiles.quality.totalCostUsd).toBeLessThan(report.flagshipCostUsd * 0.2);
  });
});

describe("loadReplaySessionsFromSignalsFile", () => {
  it("loads signals JSON, applies backend calibration, and synthesizes turns when claudeProjectsDir is omitted", () => {
    const tmp = mkdtempSync(join(tmpdir(), "s1gate-replay-test-"));
    try {
      const signalsPath = join(tmp, "signals.json");
      writeFileSync(
        signalsPath,
        JSON.stringify({
          sessions: [
            {
              id: "sess-1",
              project: "demo",
              isSubagent: false,
              turns: 5,
              hasTools: true,
              clef27b: {
                domain: "software_engineering",
                probs: { software_engineering: 0.8, terminal_devops: 0.2 },
                conf: 0.8,
                comp: 0.6,
                noul: 0.2,
                latencyMs: 510,
                ok: true,
              },
              tev1_08b: {
                domain: "software_engineering",
                probs: { software_engineering: 0.9 },
                conf: 0.9,
                comp: 0.75,
                noul: 0.3,
                latencyMs: 320,
                ok: true,
              },
            },
          ],
        }),
      );

      const clefSessions = loadReplaySessionsFromSignalsFile(signalsPath, "clef27b");
      expect(clefSessions).toHaveLength(1);
      expect(clefSessions[0].turns).toHaveLength(5);
      expect(clefSessions[0].signals.backend).toBe("clef:27b");
      expect(clefSessions[0].signals.complexity).toBeCloseTo(0.6, 5);

      const tev08Sessions = loadReplaySessionsFromSignalsFile(signalsPath, "tev1_08b");
      expect(tev08Sessions[0].signals.backend).toBe("tev1:0.8b");
      // tev1:0.8b has -0.15 complexity calibration offset
      expect(tev08Sessions[0].signals.complexity).toBeCloseTo(0.6, 5);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("pairs signals JSON with real Claude Code .jsonl session logs when claudeProjectsDir is provided", () => {
    const tmp = mkdtempSync(join(tmpdir(), "s1gate-replay-jsonl-"));
    try {
      const signalsPath = join(tmp, "signals.json");
      const projectsDir = join(tmp, "projects");
      const projDir = join(projectsDir, "-Users-demo-app");
      mkdirSync(projDir, { recursive: true });

      const jsonlLines = [
        JSON.stringify({
          type: "assistant",
          message: {
            model: "claude-opus-5-5",
            content: [{ type: "tool_use", name: "Read" }],
            usage: {
              input_tokens: 1000,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 20000,
              output_tokens: 400,
            },
          },
        }),
        JSON.stringify({
          type: "user",
          message: {
            content: [{ type: "tool_result", is_error: true, content: "ENOENT" }],
          },
        }),
        JSON.stringify({
          type: "assistant",
          message: {
            model: "claude-opus-5-5",
            content: [{ type: "text", text: "Fixed" }],
            usage: {
              input_tokens: 300,
              cache_read_input_tokens: 21000,
              cache_creation_input_tokens: 1500,
              output_tokens: 250,
            },
          },
        }),
      ];
      writeFileSync(join(projDir, "abc-123.jsonl"), jsonlLines.join("\n"));

      writeFileSync(
        signalsPath,
        JSON.stringify({
          sessions: [
            {
              id: "abc-123",
              project: "app",
              isSubagent: false,
              turns: 99,
              hasTools: true,
              clef27b: {
                domain: "terminal_devops",
                probs: { terminal_devops: 0.9 },
                conf: 0.9,
                comp: 0.5,
                noul: 0.1,
                latencyMs: 400,
                ok: true,
              },
            },
          ],
        }),
      );

      const loaded = loadReplaySessionsFromSignalsFile(signalsPath, "clef27b", projectsDir);
      expect(loaded).toHaveLength(1);
      expect(loaded[0].turns).toHaveLength(2);
      expect(loaded[0].turns[0].cacheWriteTokens).toBe(20000);
      expect(loaded[0].turns[0].toolUses).toBe(1);
      expect(loaded[0].turns[0].toolErrors).toBe(1);
      expect(loaded[0].turns[1].cacheReadTokens).toBe(21000);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("replays the real 257-session benchmark file without hy4 or mimo-v2.6-flash winning balanced", () => {
    const repoParent = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
    const signalsPath =
      process.env.S1GATE_TEST_SIGNALS_PATH ??
      join(repoParent, "data-s1gate", "benchmarks", "systemone-signals-257-sessions.json");
    const ocxModelsPath =
      process.env.S1GATE_TEST_OCX_MODELS_PATH ??
      join(repoParent, "data-s1gate", "ocx", "local", "models.json");
    if (!existsSync(signalsPath) || !existsSync(ocxModelsPath)) return;

    const ocxDoc = JSON.parse(readFileSync(ocxModelsPath, "utf8")) as {
      data: Array<{ id: string }>;
    };
    const ocxIds = ocxDoc.data.map(m => m.id);
    const sessions = loadReplaySessionsFromSignalsFile(signalsPath, "clef27b");
    expect(sessions.length).toBe(257);

    const report = runCounterfactualReplay({
      sessions,
      availableModels: ocxIds,
      flagshipModel: "anthropic/claude-opus-5-5",
    });

    expect(report.sessionsCount).toBe(257);
    expect(report.profiles.balanced.savedVsFlagshipPct).toBeGreaterThan(20);
    for (const winner of Object.keys(report.profiles.balanced.modelMix)) {
      expect(winner.toLowerCase()).not.toContain("hy4");
      expect(winner.toLowerCase()).not.toContain("mimo-v2.6-flash");
    }
  });

  it("clamps synthesized turnCount to [1, 10000], sanitizes negative JSONL token counts, and skips symlinks", () => {
    const tmp = mkdtempSync(join(tmpdir(), "s1gate-replay-sec-"));
    try {
      const outsideFile = join(tmp, "outside.jsonl");
      writeFileSync(
        outsideFile,
        JSON.stringify({
          type: "assistant",
          message: {
            model: "claude-opus-5-5",
            content: [{ type: "text", text: "secret" }],
            usage: {
              input_tokens: 9999,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
              output_tokens: 999,
            },
          },
        }),
      );

      const projectsDir = join(tmp, "projects");
      const projDir = join(projectsDir, "-Users-demo-sec");
      mkdirSync(projDir, { recursive: true });
      symlinkSync(outsideFile, join(projDir, "sym-sess.jsonl"));

      writeFileSync(
        join(projDir, "neg-sess.jsonl"),
        JSON.stringify({
          type: "assistant",
          message: {
            model: "claude-sonnet-5-5",
            content: [{ type: "text", text: "ok" }],
            usage: {
              input_tokens: -500,
              cache_read_input_tokens: -200,
              cache_creation_input_tokens: 0,
              output_tokens: 250,
            },
          },
        }),
      );

      const signalsPath = join(tmp, "signals.json");
      writeFileSync(
        signalsPath,
        JSON.stringify({
          sessions: [
            { id: "huge-turns", turns: 999_999, clef27b: { domain: "software_engineering", comp: 0.5 } },
            { id: "zero-turns", turns: 0, clef27b: { domain: "software_engineering", comp: 0.5 } },
            { id: "neg-turns", turns: -42, clef27b: { domain: "software_engineering", comp: 0.5 } },
            { id: "sym-sess", turns: 2, clef27b: { domain: "software_engineering", comp: 0.5 } },
            { id: "neg-sess", turns: 3, clef27b: { domain: "software_engineering", comp: 0.5 } },
          ],
        }),
      );

      const loaded = loadReplaySessionsFromSignalsFile(signalsPath, "clef27b", projectsDir);
      expect(loaded).toHaveLength(5);
      // Clamped to 10_000 max and 1 min
      expect(loaded[0].turns).toHaveLength(10_000);
      expect(loaded[1].turns).toHaveLength(1);
      expect(loaded[2].turns).toHaveLength(1);
      // Symlinked outside file was NOT read -> fell back to synthesizeTurns(2)
      expect(loaded[3].turns).toHaveLength(2);
      expect(loaded[3].turns[0].inputTokens).not.toBe(9999);
      // Negative JSONL tokens sanitized to 0 while positive outputTokens preserved
      expect(loaded[4].turns).toHaveLength(1);
      expect(loaded[4].turns[0].inputTokens).toBe(0);
      expect(loaded[4].turns[0].cacheReadTokens).toBe(0);
      expect(loaded[4].turns[0].outputTokens).toBe(250);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });
});

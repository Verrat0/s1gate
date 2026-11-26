import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import pkg from "../package.json";
import { parseArgv, runCli, VERSION } from "../src/cli/index.js";
import type { EvalReport } from "../src/replay/eval.js";
import type { ReplayReport } from "../src/replay/replay.js";
import type { ScoringResult } from "../src/types.js";

async function captureCli(argv: readonly string[]): Promise<string> {
  let out = "";
  await runCli(argv, (chunk) => {
    out += chunk;
  });
  return out;
}

describe("s1gate CLI", () => {
  it("reports --version matching package.json version without drift", async () => {
    expect(VERSION).toBe(pkg.version);
    const out = await captureCli(["--version"]);
    expect(out.trim()).toBe(`s1gate ${pkg.version}`);
  });

  it("prints help including eval, replay --demo, and --models-file", async () => {
    const out = await captureCli(["--help"]);
    expect(out).toContain(`s1gate v${pkg.version}`);
    expect(out).toContain("eval");
    expect(out).toContain("--demo");
    expect(out).toContain("--models-file");
    expect(out).toContain("--ocx-models");
  });

  it("runs inspect --json cleanly", async () => {
    const out = await captureCli([
      "inspect",
      "--domain",
      "software_engineering",
      "--complexity",
      "0.5",
      "--profile",
      "balanced",
      "--json",
    ]);
    const parsed = JSON.parse(out) as ScoringResult;
    expect(parsed.selected.proxyId.length).toBeGreaterThan(0);
    expect(parsed.rankedCandidates.length).toBeGreaterThan(0);
    expect(parsed.qMax).toBeGreaterThan(0);
  });

  it("runs replay --demo --json out of the box with valid cost savings", async () => {
    const out = await captureCli(["replay", "--demo", "--json"]);
    const report = JSON.parse(out) as ReplayReport;
    expect(report.sessionsCount).toBe(16);
    expect(report.totalTurns).toBeGreaterThan(100);
    expect(report.profiles.balanced.totalCostUsd).toBeLessThan(report.flagshipCostUsd);
    expect(report.profiles.frugal.totalCostUsd).toBeLessThanOrEqual(
      report.profiles.balanced.totalCostUsd,
    );
  });

  it("supports --models-file and backward-compatible --ocx-models alias on replay", async () => {
    const tmp = mkdtempSync(join(tmpdir(), "s1gate-cli-models-"));
    try {
      const modelsFile = join(tmp, "models.json");
      writeFileSync(
        modelsFile,
        JSON.stringify({
          data: [
            { id: "anthropic/claude-opus-5-5" },
            { id: "anthropic/claude-sonnet-5-5" },
            { id: "anthropic/claude-haiku-5-5" },
            { id: "openai/gpt-6.1-terra" },
          ],
        }),
      );

      const parsedPrimary = parseArgv(["replay", "--demo", "--models-file", modelsFile, "--json"]);
      expect(parsedPrimary.flags["models-file"]).toBe(modelsFile);
      expect(parsedPrimary.flags.demo).toBe(true);

      const outPrimary = await captureCli(["replay", "--demo", "--models-file", modelsFile, "--json"]);
      const outAlias = await captureCli(["replay", "--demo", "--ocx-models", modelsFile, "--json"]);

      const reportPrimary = JSON.parse(outPrimary) as ReplayReport;
      const reportAlias = JSON.parse(outAlias) as ReplayReport;
      expect(reportPrimary).toEqual(reportAlias);
      expect(reportPrimary.sessionsCount).toBe(16);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("runs eval --local-only --json and defaults to local_only when no URL is configured", async () => {
    const outExplicit = await captureCli(["eval", "--local-only", "--json"]);
    const reportExplicit = JSON.parse(outExplicit) as EvalReport;
    expect(reportExplicit.mode).toBe("local_only");
    expect(reportExplicit.totalPrompts).toBe(66);
    expect(reportExplicit.domainAccuracyPct).toBeGreaterThanOrEqual(85);
    expect(reportExplicit.estimatedDecisionCostPer1kUsd).toBe(0);

    const prevUrl = process.env.S1GATE_SYSTEMONE_URL;
    delete process.env.S1GATE_SYSTEMONE_URL;
    try {
      const outDefault = await captureCli(["eval", "--json"]);
      const reportDefault = JSON.parse(outDefault) as EvalReport;
      expect(reportDefault.mode).toBe("local_only");
      expect(reportDefault.totalPrompts).toBe(66);
    } finally {
      if (prevUrl !== undefined) {
        process.env.S1GATE_SYSTEMONE_URL = prevUrl;
      }
    }

    const outMd = await captureCli(["eval", "--local-only", "--markdown"]);
    expect(outMd).toContain("## s1gate Public Evaluation Report");
  });
});

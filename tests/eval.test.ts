import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import bundledDomains from "../data/domains.json";
import {
  formatEvalReport,
  loadEvalDataset,
  matchesComplexityBand,
  runPublicEval,
  type EvalPromptItem,
} from "../src/replay/eval.js";
import type { ClassifierAdapter, ClassifierRequest, SystemOneSignals } from "../src/types.js";

describe("runPublicEval", () => {
  it("loads the bundled 66-prompt dataset covering all 11 domains cleanly", () => {
    const prompts = loadEvalDataset();
    expect(prompts).toHaveLength(66);

    const countsByDomain = new Map<string, number>();
    const roles = new Set<string>();
    const bands = new Set<string>();
    for (const item of prompts) {
      countsByDomain.set(item.expectedDomain, (countsByDomain.get(item.expectedDomain) ?? 0) + 1);
      roles.add(item.expectedRole);
      bands.add(item.expectedComplexityBand);
      // Global English dataset check: no region-specific clinical references
      expect(item.prompt.toLowerCase()).not.toContain("simrs");
      expect(item.prompt.toLowerCase()).not.toContain("bpjs");
    }

    expect(countsByDomain.size).toBe(bundledDomains.length);
    for (const domain of bundledDomains) {
      expect(countsByDomain.get(domain.id)).toBe(6);
    }
    expect(roles.has("auxiliary")).toBe(true);
    expect(roles.has("specialist")).toBe(true);
    expect(roles.has("orchestrator")).toBe(true);
    expect(bands.has("low")).toBe(true);
    expect(bands.has("medium")).toBe(true);
    expect(bands.has("high")).toBe(true);
  });

  it("evaluates data/eval-prompts.json in local_only mode with high accuracy and zero remote cost", async () => {
    const report = await runPublicEval({ localOnly: true, profile: "balanced" });

    expect(report.mode).toBe("local_only");
    expect(report.backend).toBe("heuristic-fallback");
    expect(report.totalPrompts).toBe(66);
    expect(report.auxiliaryBypassedPrompts).toBeGreaterThan(0);
    expect(report.classifiedPrompts + report.auxiliaryBypassedPrompts).toBe(66);
    expect(report.domainAccuracyPct).toBeGreaterThanOrEqual(85);
    expect(report.roleAgreementPct).toBeGreaterThanOrEqual(85);
    expect(report.complexityAgreementPct).toBeGreaterThanOrEqual(85);
    expect(report.latencyP50Ms).toBeGreaterThanOrEqual(0);
    expect(report.latencyP95Ms).toBeGreaterThanOrEqual(report.latencyP50Ms);
    expect(report.estimatedDecisionCostPer1kUsd).toBe(0);
    expect(report.perDomain).toHaveLength(11);
    expect(report.items).toHaveLength(66);
  });

  it("scales estimatedDecisionCostPer1kUsd by actual classifier call rate for remote classifiers", async () => {
    const mockClassifier: ClassifierAdapter = {
      name: "mock-systemone",
      async classify(request: ClassifierRequest): Promise<SystemOneSignals> {
        return {
          domain: request.domains[0]?.id ?? "software_engineering",
          domainProbabilities: { software_engineering: 0.95 },
          domainConfidence: 0.95,
          complexity: 0.45,
          needsFlagshipOrchestrator: 0.15,
          backend: "mock-systemone",
          latencyMs: 12,
          fromFallback: false,
        };
      },
    };

    const samplePrompts: EvalPromptItem[] = [
      {
        id: "aux-1",
        prompt: "Hello! What is the capital of France?",
        expectedDomain: "general_chat",
        expectedRole: "auxiliary",
        expectedComplexityBand: "low",
        hasTools: false,
        maxTokens: 128,
      },
      {
        id: "se-1",
        prompt: "Add an index to the PostgreSQL orders table migration.",
        expectedDomain: "software_engineering",
        expectedRole: "specialist",
        expectedComplexityBand: "medium",
        hasTools: true,
      },
    ];

    const report = await runPublicEval({
      prompts: samplePrompts,
      localOnly: false,
      classifier: mockClassifier,
    });

    expect(report.mode).toBe("systemone");
    expect(report.backend).toBe("mock-systemone");
    expect(report.totalPrompts).toBe(2);
    expect(report.auxiliaryBypassedPrompts).toBe(1);
    expect(report.classifiedPrompts).toBe(1);
    expect(report.classifierCallRatePct).toBe(50);
    expect(report.estimatedDecisionCostPer1kUsd).toBeCloseTo(0.02, 4);
  });

  it("supports custom dataset files and formats table, markdown, and json reports", async () => {
    const tmp = mkdtempSync(join(tmpdir(), "s1gate-eval-test-"));
    try {
      const datasetPath = join(tmp, "custom-eval.json");
      writeFileSync(
        datasetPath,
        JSON.stringify([
          {
            id: "c-1",
            prompt: "Write a simple TypeScript function and unit test for an API schema.",
            expectedDomain: "software_engineering",
            expectedRole: "specialist",
            expectedComplexityBand: "low",
            hasTools: false,
          },
          {
            id: "c-2",
            prompt: "Hi! What is the capital of Japan?",
            expectedDomain: "general_chat",
            expectedRole: "auxiliary",
            expectedComplexityBand: "low",
            hasTools: false,
            maxTokens: 256,
          },
        ]),
      );

      const report = await runPublicEval({ datasetPath, localOnly: true });
      expect(report.totalPrompts).toBe(2);

      const jsonText = formatEvalReport(report, "json");
      const parsed = JSON.parse(jsonText) as typeof report;
      expect(parsed.totalPrompts).toBe(2);
      expect(parsed.domainAccuracyPct).toBe(100);

      const mdText = formatEvalReport(report, "markdown");
      expect(mdText).toContain("## s1gate Public Evaluation Report");
      expect(mdText).toContain("| Domain | Total | Classified | Aux Bypass | Domain Acc | Role Agr | Complexity Agr |");
      expect(mdText).toContain("`software_engineering`");

      const tableText = formatEvalReport(report, "table");
      expect(tableText).toContain("s1gate Public Eval");
      expect(tableText).toContain("software_engineering");
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("checks complexity band boundaries accurately", () => {
    expect(matchesComplexityBand(0.2, "low")).toBe(true);
    expect(matchesComplexityBand(0.35, "low")).toBe(true);
    expect(matchesComplexityBand(0.36, "low")).toBe(false);

    expect(matchesComplexityBand(0.25, "medium")).toBe(true);
    expect(matchesComplexityBand(0.5, "medium")).toBe(true);
    expect(matchesComplexityBand(0.75, "medium")).toBe(true);
    expect(matchesComplexityBand(0.24, "medium")).toBe(false);

    expect(matchesComplexityBand(0.55, "high")).toBe(true);
    expect(matchesComplexityBand(0.9, "high")).toBe(true);
    expect(matchesComplexityBand(0.54, "high")).toBe(false);
  });
});

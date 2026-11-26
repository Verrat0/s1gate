import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildCandidatePool,
  isNonChatUtilityModel,
  normalizeProxyModelId,
  resolveCanonicalModelId,
} from "../src/core/catalog.js";
import type { BenchmarkSnapshot } from "../src/types.js";

const snapshot: BenchmarkSnapshot = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "../data/snapshot.json"), "utf8"),
);

describe("catalog: canonical model ID resolution & candidate pool", () => {
  it("normalizes OpenCodex, 9Router, and OpenRouter model slugs consistently", () => {
    expect(normalizeProxyModelId("ocx-claude-ollama-cloud--kimi-k3.1-pro")).toBe("kimi-k3.1-pro");
    expect(normalizeProxyModelId("google-antigravity/gemini-4-hydrogen-tiered-high")).toBe("gemini-4-hydrogen");
    expect(normalizeProxyModelId("gpt-6.2-sol--fast")).toBe("gpt-6.2-sol");
    expect(normalizeProxyModelId("anthropic/claude-fable-5-2")).toBe("claude-fable-5-2");
    expect(normalizeProxyModelId("openai/gpt-oss:120b:free")).toBe("gpt-oss-120b");
    expect(normalizeProxyModelId("ollama-cloud/deepseek-v4-pro:0813:none")).toBe("deepseek-v4-pro");
    expect(normalizeProxyModelId("ollama-cloud/glm-5.3:high")).toBe("glm-5.3");
    expect(normalizeProxyModelId("ollama-cloud/gemma4:31b:none")).toBe("gemma4-31b");
  });

  it("resolves real OCX and OpenRouter model IDs against the bundled llm-stats snapshot", () => {
    expect(resolveCanonicalModelId("ollama-cloud/deepseek-v4-pro:0813:none", snapshot)).toBe("deepseek-v4-pro-0813");
    expect(resolveCanonicalModelId("ollama-cloud/gemma4:31b:none", snapshot)).toBe("gemma-4-31b-it");
    expect(resolveCanonicalModelId("anthropic/claude-fable-5-2", snapshot)).toBe("claude-fable-5-2");
    expect(resolveCanonicalModelId("anthropic/claude-haiku-4-5", snapshot)).toBe("claude-haiku-4-5-20251001");
    expect(resolveCanonicalModelId("google-antigravity/gemini-4-neon", snapshot)).toBe("gemini-4-neon");
    expect(resolveCanonicalModelId("google-antigravity/gemini-4-hydrogen-tiered-high", snapshot)).toBe("gemini-4-hydrogen");
    expect(resolveCanonicalModelId("ollama-cloud/beam", snapshot)).toBe("beam-1.0-501b-a23b");
    expect(resolveCanonicalModelId("ollama-cloud/hy4", snapshot)).toBe("hy4-flash");
    expect(resolveCanonicalModelId("ollama-cloud/nemotron-3-ultra", snapshot)).toBe("nemotron-3-ultra-550b-a55b");
    expect(resolveCanonicalModelId("gpt-6.2-sol--fast", snapshot)).toBe("gpt-6.2-sol");
  });

  it("filters out non-chat media and decision models like clef:27b and flash-image", () => {
    expect(isNonChatUtilityModel("ollama-cloud/clef:27b")).toBe(true);
    expect(isNonChatUtilityModel("google-antigravity/gemini-3.1-flash-image")).toBe(true);
    expect(isNonChatUtilityModel("anthropic/claude-sonnet-5-5")).toBe(false);

    const pool = buildCandidatePool(
      [
        "ollama-cloud/clef:27b",
        "google-antigravity/gemini-3.1-flash-image",
        "anthropic/claude-sonnet-5-5",
        "anthropic/claude-sonnet-5-5", // duplicate
      ],
      snapshot,
    );
    expect(pool).toHaveLength(1);
    expect(pool[0].proxyId).toBe("anthropic/claude-sonnet-5-5");
    expect(pool[0].canonicalId).toBe("claude-sonnet-5-5");
    expect(pool[0].harnessCompat.claude_code).toBe("native");
  });

  it("applies custom model overrides cleanly without mutating snapshot", () => {
    const pool = buildCandidatePool(["custom/my-fast-model"], snapshot, {
      "custom/my-fast-model": {
        canonicalId: "claude-sonnet-5-5",
        inputPricePerMTok: 0.5,
        proxyStripsCacheControl: true,
        harnessCompat: { claude_code: "unsupported" },
      },
    });
    expect(pool).toHaveLength(1);
    expect(pool[0].inputPricePerMTok).toBe(0.5);
    expect(pool[0].proxyStripsCacheControl).toBe(true);
    expect(pool[0].harnessCompat.claude_code).toBe("unsupported");
    expect(snapshot.models["claude-sonnet-5-5"].inputPricePerMTok).toBe(2);
  });

  it("enforces delimiter boundaries in EXCLUDED_MODEL_PATTERN so normal chat models are not falsely excluded", () => {
    // True positives: utility/media tokens bounded by start/end or delimiters (-, _, /, :, .)
    expect(isNonChatUtilityModel("openai/tts-1")).toBe(true);
    expect(isNonChatUtilityModel("openai/whisper-large-v3")).toBe(true);
    expect(isNonChatUtilityModel("openai/text-embedding-3-large")).toBe(true);
    expect(isNonChatUtilityModel("provider/omni-moderation-latest")).toBe(true);
    expect(isNonChatUtilityModel("provider/dall-e-3")).toBe(true);
    expect(isNonChatUtilityModel("google/imagen-3.0")).toBe(true);
    expect(isNonChatUtilityModel("provider/tev1")).toBe(true);

    // False-positive immunity: excluded tokens appearing as substrings inside ordinary words
    expect(isNonChatUtilityModel("provider/watts-20b")).toBe(false);
    expect(isNonChatUtilityModel("provider/clefairy-7b")).toBe(false);
    expect(isNonChatUtilityModel("provider/embedded-coder-32b")).toBe(false);
    expect(isNonChatUtilityModel("provider/stev1n-8b")).toBe(false);
    expect(isNonChatUtilityModel("provider/jevon-pro")).toBe(false);
    expect(isNonChatUtilityModel("provider/reimagenation-7b")).toBe(false);
  });
});

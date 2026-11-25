import { describe, expect, it } from "vitest";
import {
  inspectFastPath,
  parseVirtualModelHint,
  pickAuxiliaryModel,
} from "../src/core/fast-path.js";
import type { S1GateRequest } from "../src/types.js";

function makeReq(overrides: Partial<S1GateRequest> = {}): S1GateRequest {
  return {
    protocol: "anthropic_messages",
    path: "/v1/messages",
    headers: {},
    body: {
      model: "s1gate",
      max_tokens: 4096,
      messages: [{ role: "user", content: "Refactor the authentication middleware to support JWT rotation." }],
    },
    availableModels: ["anthropic/claude-opus-4", "anthropic/claude-haiku-4-5"],
    ...overrides,
  };
}

describe("parseVirtualModelHint", () => {
  it("parses profile hints", () => {
    expect(parseVirtualModelHint("s1gate:frugal")).toEqual({ base: "s1gate", profile: "frugal" });
    expect(parseVirtualModelHint("s1gate:quality")).toEqual({ base: "s1gate", profile: "quality" });
    expect(parseVirtualModelHint("s1gate:balanced")).toEqual({ base: "s1gate", profile: "balanced" });
    expect(parseVirtualModelHint("auto:FRUGAL")).toEqual({ base: "auto", profile: "frugal" });
  });

  it("parses role hints including :fast alias", () => {
    expect(parseVirtualModelHint("s1gate:orchestrator")).toEqual({ base: "s1gate", role: "orchestrator" });
    expect(parseVirtualModelHint("s1gate:subagent")).toEqual({ base: "s1gate", role: "subagent" });
    expect(parseVirtualModelHint("s1gate:specialist")).toEqual({ base: "s1gate", role: "specialist" });
    expect(parseVirtualModelHint("s1gate:auxiliary")).toEqual({ base: "s1gate", role: "auxiliary" });
    expect(parseVirtualModelHint("s1gate:fast")).toEqual({ base: "s1gate", role: "auxiliary" });
  });

  it("supports combined profile and role suffixes", () => {
    expect(parseVirtualModelHint("s1gate:frugal:subagent")).toEqual({
      base: "s1gate",
      profile: "frugal",
      role: "subagent",
    });
  });

  it("returns null for regular models, parameter tags, or invalid syntax", () => {
    expect(parseVirtualModelHint("anthropic/claude-sonnet-5-5")).toBeNull();
    expect(parseVirtualModelHint("ollama-cloud/clef:27b")).toBeNull();
    expect(parseVirtualModelHint("openai/gpt-oss:120b:free")).toBeNull();
    expect(parseVirtualModelHint(":frugal")).toBeNull();
    expect(parseVirtualModelHint("")).toBeNull();
  });
});

describe("pickAuxiliaryModel", () => {
  it("prefers explicit user hints before built-in auxiliary patterns", () => {
    const models = [
      "anthropic/claude-opus-4",
      "anthropic/claude-haiku-4-5",
      "custom/my-tiny-helper",
    ];
    expect(pickAuxiliaryModel(models, ["my-tiny-helper"])).toBe("custom/my-tiny-helper");
    expect(pickAuxiliaryModel(models, ["nonexistent", "anthropic/claude-haiku-4-5"])).toBe(
      "anthropic/claude-haiku-4-5",
    );
  });

  it("prefers haiku, flash-lite, mini, nano, and luna when no hints match", () => {
    expect(
      pickAuxiliaryModel(["anthropic/claude-opus-4", "anthropic/claude-haiku-4-5", "openai/gpt-5-mini"]),
    ).toBe("anthropic/claude-haiku-4-5");
    expect(
      pickAuxiliaryModel(["google/gemini-pro", "google/gemini-2.5-flash-lite"]),
    ).toBe("google/gemini-2.5-flash-lite");
    expect(
      pickAuxiliaryModel(["openai/gpt-5", "openai/gpt-5-mini"]),
    ).toBe("openai/gpt-5-mini");
    expect(
      pickAuxiliaryModel(["openai/gpt-5", "openai/gpt-5-nano"]),
    ).toBe("openai/gpt-5-nano");
    expect(
      pickAuxiliaryModel(["provider/flagship", "provider/luna-small"]),
    ).toBe("provider/luna-small");
  });

  it("falls back to first available model or empty string if pool is empty", () => {
    expect(pickAuxiliaryModel(["anthropic/claude-opus-4", "openai/gpt-5"])).toBe(
      "anthropic/claude-opus-4",
    );
    expect(pickAuxiliaryModel([])).toBe("");
  });

  it("filters out non-chat utility models before matching hints, patterns, or fallback", () => {
    const models = [
      "google/gemini-2.5-flash-image",
      "openai/whisper-1",
      "openai/text-embedding-3-small",
      "ollama/tev1:0.8b",
      "anthropic/claude-sonnet-5-5",
    ];
    expect(pickAuxiliaryModel(models, ["flash-image", "whisper"])).toBe(
      "anthropic/claude-sonnet-5-5",
    );
    expect(
      pickAuxiliaryModel([
        "google/gemini-2.5-flash-image",
        "openai/whisper-1",
        "ollama/clef:27b",
      ]),
    ).toBe("");
  });
});

describe("inspectFastPath", () => {
  it("passes through when model is not in triggerModels allowlist", () => {
    const req = makeReq({
      body: { model: "anthropic/claude-sonnet-5-5", messages: [{ role: "user", content: "Hello" }] },
    });
    const verdict = inspectFastPath(req, { triggerModels: ["s1gate", "auto"] });
    expect(verdict.kind).toBe("passthrough");
    expect(verdict.reason).toBeDefined();
  });

  it("matches virtual model hint base against triggerModels allowlist", () => {
    const req = makeReq({
      body: {
        model: "s1gate:quality",
        max_tokens: 4096,
        messages: [{ role: "user", content: "Implement binary search tree balancing." }],
      },
    });
    const verdict = inspectFastPath(req, { triggerModels: ["s1gate"] });
    expect(verdict.kind).toBe("proceed");
    expect(verdict.profileOverride).toBe("quality");
  });

  it("passes through /v1/messages/count_tokens requests", () => {
    const req = makeReq({ path: "/v1/messages/count_tokens" });
    const verdict = inspectFastPath(req);
    expect(verdict.kind).toBe("passthrough");
    expect(verdict.reason).toBe("count_tokens");
  });

  it("detects compaction via path, anthropic-beta header, or system prompt", () => {
    const byPath = inspectFastPath(makeReq({ path: "/v1/messages/compact" }));
    expect(byPath.kind).toBe("compact");

    const byHeader = inspectFastPath(
      makeReq({
        headers: { "Anthropic-Beta": "prompt-caching-2024-07-31,compact-2026-01-12" },
      }),
    );
    expect(byHeader.kind).toBe("compact");

    const bySystem = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          system: "Your task is to create a detailed summary of the conversation so far for context compaction.",
          messages: [{ role: "user", content: "Compact please" }],
        },
      }),
    );
    expect(bySystem.kind).toBe("compact");
  });

  it("detects auxiliary calls via :fast hint, roleOverride, utility prompt, and short non-tool heuristic", () => {
    const byHint = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate:fast",
          max_tokens: 4096,
          messages: [{ role: "user", content: "Summarize this commit" }],
        },
      }),
    );
    expect(byHint.kind).toBe("auxiliary");
    expect(byHint.role).toBe("auxiliary");

    const byOverride = inspectFastPath(makeReq({ roleOverride: "auxiliary" }));
    expect(byOverride.kind).toBe("auxiliary");
    expect(byOverride.role).toBe("auxiliary");

    const byUtilityPrompt = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          system: "Generate a concise 3-5 word title for this conversation.",
          max_tokens: 1024,
          messages: [{ role: "user", content: "We discussed database indexing and query plans." }],
        },
      }),
    );
    expect(byUtilityPrompt.kind).toBe("auxiliary");

    const byShortHeuristic = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          max_tokens: 256,
          messages: [{ role: "user", content: "What does HTTP 418 mean?" }],
        },
      }),
    );
    expect(byShortHeuristic.kind).toBe("auxiliary");
  });

  it("does NOT classify short prompt as auxiliary when tools, code blocks, stack traces, or complex keywords are present", () => {
    const withTools = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          max_tokens: 256,
          tools: [{ name: "read_file", input_schema: { type: "object" } }],
          messages: [{ role: "user", content: "Read foo.ts" }],
        },
      }),
    );
    expect(withTools.kind).toBe("proceed");

    const withCodeBlock = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          max_tokens: 256,
          messages: [{ role: "user", content: "Fix this:\n```ts\nconst x = 1;\n```" }],
        },
      }),
    );
    expect(withCodeBlock.kind).toBe("proceed");

    const withStackTrace = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          max_tokens: 256,
          messages: [
            {
              role: "user",
              content: "TypeError: x is undefined\n  at parseToken (src/auth.ts:42:10)",
            },
          ],
        },
      }),
    );
    expect(withStackTrace.kind).toBe("proceed");

    for (const kw of ["architect", "security audit", "race condition", "prove"]) {
      const withComplexKw = inspectFastPath(
        makeReq({
          body: {
            model: "s1gate",
            max_tokens: 256,
            messages: [{ role: "user", content: `Please ${kw} this lock design.` }],
          },
        }),
      );
      expect(withComplexKw.kind).toBe("proceed");
    }

    const withHighMaxTokens = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          max_tokens: 2048,
          messages: [{ role: "user", content: "Say hi" }],
        },
      }),
    );
    expect(withHighMaxTokens.kind).toBe("proceed");
  });

  it("detects subagent headers case-insensitively", () => {
    const claudeSubagent = inspectFastPath(
      makeReq({ headers: { "X-Claude-Code-Agent-Id": "agent-42" } }),
    );
    expect(claudeSubagent.isSubagentHeader).toBe(true);
    expect(claudeSubagent.role).toBe("subagent");

    const openaiSubagent = inspectFastPath(
      makeReq({ headers: { "x-openai-subagent": "true" } }),
    );
    expect(openaiSubagent.isSubagentHeader).toBe(true);
    expect(openaiSubagent.role).toBe("subagent");

    const openaiSubagentFalse = inspectFastPath(
      makeReq({ headers: { "x-openai-subagent": "false" } }),
    );
    expect(openaiSubagentFalse.isSubagentHeader).toBe(false);

    const codexSubagent = inspectFastPath(
      makeReq({
        headers: {
          "X-Codex-Parent-Thread-Id": "thread-parent",
          "Thread-Id": "thread-child",
        },
      }),
    );
    expect(codexSubagent.isSubagentHeader).toBe(true);
    expect(codexSubagent.role).toBe("subagent");

    const codexSameThread = inspectFastPath(
      makeReq({
        headers: {
          "x-codex-parent-thread-id": "thread-1",
          "thread-id": "thread-1",
        },
      }),
    );
    expect(codexSameThread.isSubagentHeader).toBe(false);
  });

  it("detects tools, subagent-spawning tools, multimodal blocks, estimated tokens, and thinking signatures", () => {
    const req = makeReq({
      body: {
        model: "s1gate",
        max_tokens: 4096,
        tools: [
          { name: "bash", input_schema: { type: "object" } },
          { name: "Agent", input_schema: { type: "object" } },
        ],
        messages: [
          { role: "user", content: "Inspect this architecture" },
          {
            role: "assistant",
            content: [
              { type: "thinking", thinking: "Step 1...", signature: "EqoBCgIYAhIM..." },
              { type: "text", text: "Let me check." },
            ],
          },
          {
            role: "user",
            content: [
              { type: "text", text: "Here is the diagram" },
              { type: "image", source: { type: "base64", media_type: "image/png", data: "AAAA" } },
            ],
          },
        ],
      },
    });

    const verdict = inspectFastPath(req);
    expect(verdict.kind).toBe("proceed");
    expect(verdict.hasTools).toBe(true);
    expect(verdict.canSpawnSubagent).toBe(true);
    expect(verdict.role).toBe("orchestrator");
    expect(verdict.requiresMultimodal).toBe(true);
    expect(verdict.hasThinkingSignature).toBe(true);
    expect(verdict.estimatedInputTokens).toBe(Math.ceil(JSON.stringify(req.body).length / 4));
  });

  it("detects OpenAI chat function tools and OpenAI responses multimodal inputs", () => {
    const chatReq = makeReq({
      protocol: "openai_chat",
      path: "/v1/chat/completions",
      body: {
        model: "s1gate",
        tools: [{ type: "function", function: { name: "spawn_agent", parameters: {} } }],
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Old screenshot" },
              { type: "image_url", image_url: { url: "data:image/png;base64,AAA" } },
            ],
          },
          { role: "assistant", content: "Got it." },
          { role: "user", content: "Now write the migration script." },
        ],
      },
    });
    const chatVerdict = inspectFastPath(chatReq);
    expect(chatVerdict.hasTools).toBe(true);
    expect(chatVerdict.canSpawnSubagent).toBe(true);
    expect(chatVerdict.requiresMultimodal).toBe(false);

    const responsesReq = makeReq({
      protocol: "openai_responses",
      path: "/v1/responses",
      body: {
        model: "s1gate",
        instructions: "You are a helpful coding assistant.",
        input: [
          {
            role: "user",
            content: [
              { type: "input_text", text: "Read this PDF" },
              { type: "input_file", filename: "spec.pdf" },
            ],
          },
        ],
      },
    });
    const respVerdict = inspectFastPath(responsesReq);
    expect(respVerdict.requiresMultimodal).toBe(true);
    expect(respVerdict.hasThinkingSignature).toBe(false);
  });

  it("gates system-prompt compaction behind !hasTools while keeping /compact path and compact-* beta unconditional", () => {
    const codingWithTools = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          system: "CLAUDE.md: Before context compaction, summarize the conversation to save memory.",
          tools: [{ name: "bash", input_schema: { type: "object" } }],
          messages: [{ role: "user", content: "Fix the failing build." }],
        },
      }),
    );
    expect(codingWithTools.kind).toBe("proceed");

    const explicitPathWithTools = inspectFastPath(
      makeReq({
        path: "/v1/messages/compact",
        body: {
          model: "s1gate",
          tools: [{ name: "bash", input_schema: { type: "object" } }],
          messages: [{ role: "user", content: "Compact" }],
        },
      }),
    );
    expect(explicitPathWithTools.kind).toBe("compact");
    expect(explicitPathWithTools.reason).toBe("compact_endpoint");

    const explicitBetaWithTools = inspectFastPath(
      makeReq({
        headers: { "anthropic-beta": "compact-2026-01-12" },
        body: {
          model: "s1gate",
          tools: [{ name: "bash", input_schema: { type: "object" } }],
          messages: [{ role: "user", content: "Compact" }],
        },
      }),
    );
    expect(explicitBetaWithTools.kind).toBe("compact");
    expect(explicitBetaWithTools.reason).toBe("compact_beta_header");
  });

  it("gates utility system prompt behind !hasTools and messages.length <= 4", () => {
    const codingWithTools = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          system: "When committing, write a concise git commit message following conventions.",
          tools: [{ name: "bash", input_schema: { type: "object" } }],
          messages: [{ role: "user", content: "Refactor router and run tests." }],
        },
      }),
    );
    expect(codingWithTools.kind).toBe("proceed");

    const longConversation = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          system: "Generate a concise 3-5 word title for this conversation.",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "msg 1" },
            { role: "assistant", content: "reply 1" },
            { role: "user", content: "msg 2" },
            { role: "assistant", content: "reply 2" },
            { role: "user", content: "msg 3" },
          ],
        },
      }),
    );
    expect(longConversation.kind).toBe("proceed");
  });

  it("populates roleOverride only from explicit req.roleOverride or virtual hint", () => {
    const inferred = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate",
          tools: [{ name: "Agent", input_schema: { type: "object" } }],
          messages: [{ role: "user", content: "Plan architecture" }],
        },
      }),
    );
    expect(inferred.role).toBe("orchestrator");
    expect(inferred.roleOverride).toBeUndefined();

    const fromHint = inspectFastPath(
      makeReq({
        body: {
          model: "s1gate:orchestrator",
          messages: [{ role: "user", content: "Plan architecture" }],
        },
      }),
    );
    expect(fromHint.role).toBe("orchestrator");
    expect(fromHint.roleOverride).toBe("orchestrator");

    const fromReq = inspectFastPath(
      makeReq({
        roleOverride: "specialist",
        body: {
          model: "s1gate:orchestrator",
          messages: [{ role: "user", content: "Do task" }],
        },
      }),
    );
    expect(fromReq.role).toBe("specialist");
    expect(fromReq.roleOverride).toBe("specialist");
  });
});

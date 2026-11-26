import { request as httpRequest } from "node:http";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createS1Gate } from "../src/core/engine.js";
import { createHttpBridge, type HttpBridgeServer } from "../src/server/http-bridge.js";
import type {
  ClassifierAdapter,
  S1GateDecision,
  S1GateRequest,
  SystemOneSignals,
} from "../src/types.js";

const TEST_MODELS = [
  "anthropic/claude-opus-5-5",
  "anthropic/claude-sonnet-5-5",
  "anthropic/claude-haiku-5-5",
  "ollama-cloud/mimo-v2.6-flash",
  "gpt-6.2-sol",
];

function createStubClassifier(overrides: Partial<SystemOneSignals> = {}): ClassifierAdapter & { calls: number } {
  const clf = {
    name: "stub-systemone",
    calls: 0,
    async classify(): Promise<SystemOneSignals> {
      clf.calls += 1;
      return {
        domain: "terminal_devops",
        domainProbabilities: { terminal_devops: 0.92, software_engineering: 0.08 },
        domainConfidence: 0.92,
        complexity: 0.45,
        needsFlagshipOrchestrator: 0.15,
        shouldEscalate: 0.2,
        backend: "stub-systemone",
        latencyMs: 5,
        fromFallback: false,
        ...overrides,
      };
    },
  };
  return clf;
}

describe("HTTP Bridge Server", () => {
  let bridge: HttpBridgeServer;
  let baseUrl: string;
  let classifier: ReturnType<typeof createStubClassifier>;
  let engine: ReturnType<typeof createS1Gate>;

  beforeAll(async () => {
    classifier = createStubClassifier();
    engine = createS1Gate({ defaultProfile: "balanced", classifier });
    bridge = createHttpBridge(engine, { port: 0, host: "127.0.0.1" });
    const started = await bridge.start();
    baseUrl = started.url;
  });

  afterAll(async () => {
    await bridge.close();
  });

  describe("lifecycle and GET /health", () => {
    it("binds to an ephemeral port and exposes the underlying node:http Server", () => {
      expect(baseUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
      expect(bridge.server).toBeDefined();
      expect(bridge.server.listening).toBe(true);
    });

    it("returns engine metadata, snapshot count, domains count, and stats on GET /health", async () => {
      const res = await fetch(`${baseUrl}/health`);
      expect(res.status).toBe(200);
      expect(res.headers.get("content-type")).toContain("application/json");

      const data = (await res.json()) as {
        status: string;
        engine: string;
        benchmarkSource: string;
        snapshotModels: number;
        domains: number;
        stats: { totalRequests: number };
      };

      expect(data.status).toBe("ok");
      expect(data.engine).toBe("s1gate");
      expect(data.benchmarkSource).toBe("llm-stats.com");
      expect(data.snapshotModels).toBe(Object.keys(engine.snapshot.models).length);
      expect(data.domains).toBe(engine.domains.length);
      expect(typeof data.stats.totalRequests).toBe("number");
    });
  });

  describe("POST /v1/route (Native Proxy API)", () => {
    it("routes a native request, defaults protocol to openai_chat, and emits decision HTTP headers", async () => {
      const res = await fetch(`${baseUrl}/v1/route`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "http-route-sess-1",
        },
        body: JSON.stringify({
          body: {
            model: "s1gate",
            messages: [{ role: "user", content: "Debug docker compose exit code 137" }],
          },
          availableModels: TEST_MODELS,
        }),
      });

      expect(res.status).toBe(200);
      const decision = (await res.json()) as S1GateDecision;

      expect(decision.action).toBe("route");
      expect(TEST_MODELS).toContain(decision.selectedModel);
      expect(decision.headers["X-Router-Benchmark-Source"]).toBe("llm-stats.com");

      // Every header in decision.headers must be mirrored onto the HTTP response
      for (const [key, value] of Object.entries(decision.headers)) {
        expect(res.headers.get(key.toLowerCase())).toBe(value);
      }
      expect(res.headers.get("x-router-model")).toBe(decision.selectedModel);
      expect(res.headers.get("x-router-benchmark-source")).toBe("llm-stats.com");
      expect(decision.recommendedEffort).toBe("medium");
      expect(res.headers.get("x-router-effort")).toBe("medium");
    });

    it("merges incoming HTTP headers with payload.headers where payload.headers takes precedence", async () => {
      // Turn 1: pin session using payload header overriding HTTP header
      const res1 = await fetch(`${baseUrl}/v1/route`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "http-header-session",
        },
        body: JSON.stringify({
          protocol: "anthropic_messages",
          harness: "claude_code",
          headers: {
            "X-Claude-Code-Session-Id": "payload-override-session",
          },
          body: {
            model: "s1gate",
            messages: [{ role: "user", content: "Check nginx upstream timeout" }],
          },
          availableModels: TEST_MODELS,
        }),
      });

      expect(res1.status).toBe(200);
      const d1 = (await res1.json()) as S1GateDecision;
      expect(d1.action).toBe("route");

      // Turn 2 with the same payload override session -> reuses sticky pin!
      const res2 = await fetch(`${baseUrl}/v1/route`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "different-http-header",
        },
        body: JSON.stringify({
          protocol: "anthropic_messages",
          harness: "claude_code",
          headers: {
            "x-claude-code-session-id": "payload-override-session",
          },
          body: {
            model: "s1gate",
            messages: [
              { role: "user", content: "Check nginx upstream timeout" },
              { role: "assistant", content: "Looking into nginx.conf." },
              { role: "user", content: "Now check keepalive_timeout" },
            ],
          },
          availableModels: TEST_MODELS,
        }),
      });

      expect(res2.status).toBe(200);
      const d2 = (await res2.json()) as S1GateDecision;
      expect(d2.action).toBe("pin_reuse");
      expect(d2.sessionKey).toBe(d1.sessionKey);
      expect(d2.selectedModel).toBe(d1.selectedModel);

      // Sending only the overridden HTTP header should NOT hit the payload-override-session pin
      const res3 = await fetch(`${baseUrl}/v1/route`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "http-header-session",
        },
        body: JSON.stringify({
          protocol: "anthropic_messages",
          body: {
            model: "s1gate",
            messages: [{ role: "user", content: "Check systemd journal logs" }],
          },
          availableModels: TEST_MODELS,
        }),
      });

      expect(res3.status).toBe(200);
      const d3 = (await res3.json()) as S1GateDecision;
      expect(d3.action).toBe("route");
      expect(d3.sessionKey).not.toBe(d1.sessionKey);
    });

    it("rejects invalid /v1/route payloads with 400 Bad Request", async () => {
      const invalidPayloads = [
        {},
        { body: "not-an-object", availableModels: TEST_MODELS },
        { body: null, availableModels: TEST_MODELS },
        { body: [], availableModels: TEST_MODELS },
        { body: { messages: [] }, availableModels: "not-an-array" },
        { body: { messages: [] }, availableModels: [123, "ok"] },
        { body: { messages: [] }, availableModels: TEST_MODELS, protocol: "invalid_proto" },
        { body: { messages: [] }, availableModels: TEST_MODELS, headers: "not-an-object" },
      ];

      for (const payload of invalidPayloads) {
        const res = await fetch(`${baseUrl}/v1/route`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });
        expect(res.status).toBe(400);
        const err = (await res.json()) as { error: string };
        expect(typeof err.error).toBe("string");
        expect(err.error.length).toBeGreaterThan(0);
      }
    });

    it("validates estimatedContextTokens on /v1/route (accepts non-negative numbers, rejects negative/non-number with 400)", async () => {
      const okRes = await fetch(`${baseUrl}/v1/route`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          body: {
            model: "s1gate",
            messages: [{ role: "user", content: "Inspect long session logs" }],
          },
          availableModels: ["ollama-cloud/gpt-oss:120b", "anthropic/claude-sonnet-5-5"],
          estimatedContextTokens: 150_000,
        }),
      });
      expect(okRes.status).toBe(200);
      const okDecision = (await okRes.json()) as S1GateDecision;
      expect(okDecision.selectedModel).toBe("anthropic/claude-sonnet-5-5");
      expect(okRes.headers.get("x-router-context-window")).toBe("1000000");

      const zeroRes = await fetch(`${baseUrl}/v1/route`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          body: {
            model: "s1gate",
            messages: [{ role: "user", content: "Short check" }],
          },
          availableModels: ["anthropic/claude-sonnet-5-5"],
          estimatedContextTokens: 0,
        }),
      });
      expect(zeroRes.status).toBe(200);

      for (const badVal of [-1, -150_000, "NaN", "150000", null, true, {}, []]) {
        const res = await fetch(`${baseUrl}/v1/route`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            body: { messages: [{ role: "user", content: "hi" }] },
            availableModels: TEST_MODELS,
            estimatedContextTokens: badVal,
          }),
        });
        expect(res.status).toBe(400);
        const err = (await res.json()) as { error: string };
        expect(err.error).toContain("estimatedContextTokens");
      }
    });
  });

  describe("POST /v1/systemone (ocx / 9router drop-in bridge)", () => {
    it("returns System One choice format with normalized monotonically decaying probabilities and router headers", async () => {
      const res = await fetch(`${baseUrl}/v1/systemone`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "sys1-sess-1",
        },
        body: JSON.stringify({
          model: "jev-latest",
          state: "Fix failing Docker container healthcheck and Linux systemd service restart loop",
          questions: {
            route: {
              type: "choice",
              instructions: "Select the best model for this request.",
              criteria: {
                "anthropic/claude-opus-5-5": "Flagship reasoning",
                "anthropic/claude-sonnet-5-5": "Balanced coding and terminal",
                "anthropic/claude-haiku-5-5": "Fast lightweight",
                "ollama-cloud/mimo-v2.6-flash": "Low-cost fallback",
                "gpt-6.2-sol": "General flagship",
              },
            },
          },
        }),
      });

      expect(res.status).toBe(200);
      expect(res.headers.get("x-router-benchmark-source")).toBe("llm-stats.com");
      expect(res.headers.get("x-router-engine")).toBe("s1gate");

      const data = (await res.json()) as {
        model: string;
        answers: Record<
          string,
          {
            type: string;
            choice: string;
            probabilities: Record<string, number>;
            confidence: number;
          }
        >;
        usage: { input_tokens: number; output_tokens: number };
        s1gate: {
          action: string;
          domain: string | null;
          role: string;
          cachePinned: boolean;
          recommendedEffort?: string;
        };
      };

      expect(data.model).toBe("s1gate");
      expect(data.usage).toEqual({ input_tokens: 0, output_tokens: 0 });
      expect(data.s1gate.action).toBe("route");
      expect(data.s1gate.domain).toBe("terminal_devops");
      expect(typeof data.s1gate.role).toBe("string");
      expect(data.s1gate.cachePinned).toBe(true);
      expect(data.s1gate.recommendedEffort).toBe("medium");
      expect(res.headers.get("x-router-effort")).toBe("medium");

      const routeAns = data.answers.route;
      expect(routeAns).toBeDefined();
      expect(routeAns.type).toBe("choice");
      expect(TEST_MODELS).toContain(routeAns.choice);
      expect(res.headers.get("x-router-model")).toBe(routeAns.choice);

      // Probabilities cover all candidates, sum to 1.0, winner is highest, and decay monotonically
      const probEntries = Object.entries(routeAns.probabilities);
      expect(probEntries.length).toBe(TEST_MODELS.length);
      const sum = probEntries.reduce((acc, [, p]) => acc + p, 0);
      expect(sum).toBeCloseTo(1.0, 6);

      const values = probEntries.map(([, p]) => p);
      expect(probEntries[0]?.[0]).toBe(routeAns.choice);
      for (let i = 1; i < values.length; i += 1) {
        expect(values[i - 1]).toBeGreaterThan(values[i]);
      }
      expect(routeAns.confidence).toBeGreaterThan(0);
      expect(routeAns.confidence).toBeLessThanOrEqual(1);
    });

    it("supports JSON object state, custom choice question key, and forwards session headers for sticky pinning", async () => {
      const payload = {
        state: {
          task: "Inspect nginx config on Ubuntu server",
          turns: 1,
        },
        questions: {
          model_pick: {
            type: "choice",
            criteria: {
              "anthropic/claude-sonnet-5-5": "Coding",
              "anthropic/claude-haiku-5-5": "Fast",
            },
          },
        },
      };

      const res1 = await fetch(`${baseUrl}/v1/systemone`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "sys1-sticky-obj",
        },
        body: JSON.stringify(payload),
      });
      expect(res1.status).toBe(200);
      const data1 = (await res1.json()) as {
        answers: Record<string, { choice: string; probabilities: Record<string, number> }>;
        s1gate: { action: string; cachePinned: boolean };
      };
      expect(data1.answers.model_pick).toBeDefined();
      expect(data1.s1gate.action).toBe("route");

      // Turn 2 with same session header -> sticky pin reuse!
      const res2 = await fetch(`${baseUrl}/v1/systemone`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "sys1-sticky-obj",
        },
        body: JSON.stringify(payload),
      });
      expect(res2.status).toBe(200);
      const data2 = (await res2.json()) as {
        answers: Record<string, { choice: string; probabilities: Record<string, number> }>;
        s1gate: { action: string; cachePinned: boolean };
      };
      expect(data2.s1gate.action).toBe("pin_reuse");
      expect(data2.s1gate.cachePinned).toBe(true);
      expect(data2.answers.model_pick.choice).toBe(data1.answers.model_pick.choice);
    });

    it("handles single-candidate criteria with probability 1.0", async () => {
      const res = await fetch(`${baseUrl}/v1/systemone`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          state: "Quick check",
          questions: {
            route: {
              type: "choice",
              criteria: {
                "anthropic/claude-sonnet-5-5": "Only candidate",
              },
            },
          },
        }),
      });

      expect(res.status).toBe(200);
      const data = (await res.json()) as {
        answers: { route: { choice: string; probabilities: Record<string, number>; confidence: number } };
      };
      expect(data.answers.route.choice).toBe("anthropic/claude-sonnet-5-5");
      expect(data.answers.route.probabilities).toEqual({ "anthropic/claude-sonnet-5-5": 1 });
      expect(data.answers.route.confidence).toBe(1);
    });

    it("rejects invalid /v1/systemone payloads with 400 Bad Request", async () => {
      const invalidPayloads = [
        { questions: { route: { type: "choice", criteria: { a: "1" } } } }, // missing state
        { state: null, questions: { route: { type: "choice", criteria: { a: "1" } } } }, // null state
        { state: "hello" }, // missing questions
        { state: "hello", questions: [] }, // array questions
        { state: "hello", questions: { scoreQ: { type: "score", criteria: ["low", "high"] } } }, // no choice question
        { state: "hello", questions: { route: { type: "choice", criteria: {} } } }, // empty criteria
      ];

      for (const payload of invalidPayloads) {
        const res = await fetch(`${baseUrl}/v1/systemone`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });
        expect(res.status).toBe(400);
        const err = (await res.json()) as { error: string };
        expect(typeof err.error).toBe("string");
      }
    });

    it("excludes 128K gpt-oss:120b on /v1/systemone when X-S1Gate-Context-Tokens: 150000 is sent and returns X-Router-Context-Window", async () => {
      const frugalEngine = createS1Gate({
        defaultProfile: "frugal",
        classifier: createStubClassifier({
          domain: "healthcare",
          domainProbabilities: { healthcare: 1.0 },
          complexity: 0.3,
        }),
      });
      const ctxBridge = createHttpBridge(frugalEngine, { port: 0, host: "127.0.0.1" });
      const started = await ctxBridge.start();
      try {
        const body = JSON.stringify({
          state: "Continue clinical triage summary",
          questions: {
            route: {
              type: "choice",
              criteria: {
                "ollama-cloud/gpt-oss:120b": "Fast low-cost 128K",
                "anthropic/claude-sonnet-5-5": "Balanced 1M context",
              },
            },
          },
        });

        // Without context header -> 128K gpt-oss:120b wins
        const smallRes = await fetch(`${started.url}/v1/systemone`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-session-id": "sys1-ctx-small",
          },
          body,
        });
        expect(smallRes.status).toBe(200);
        const smallData = (await smallRes.json()) as {
          answers: { route: { choice: string } };
        };
        expect(smallData.answers.route.choice).toBe("ollama-cloud/gpt-oss:120b");
        expect(smallRes.headers.get("x-router-context-window")).toBe("131072");

        // With X-S1Gate-Context-Tokens: 150000 -> 128K excluded, 1M model chosen
        const largeRes = await fetch(`${started.url}/v1/systemone`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-session-id": "sys1-ctx-large",
            "X-S1Gate-Context-Tokens": "150000",
          },
          body,
        });
        expect(largeRes.status).toBe(200);
        const largeData = (await largeRes.json()) as {
          answers: { route: { choice: string } };
        };
        expect(largeData.answers.route.choice).toBe("anthropic/claude-sonnet-5-5");
        expect(largeRes.headers.get("x-router-context-window")).toBe("1000000");
      } finally {
        await ctxBridge.close();
      }
    });

    it("parses structured state signals (estimated_tokens, has_image, tool_history) and criteria quota % to steer away from 128K and high-quota models", async () => {
      const captured: S1GateRequest[] = [];
      const baseEngine = createS1Gate({
        defaultProfile: "balanced",
        classifier: createStubClassifier({
          domain: "healthcare",
          domainProbabilities: { healthcare: 1.0 },
          complexity: 0.5,
        }),
      });
      const spyEngine = {
        ...baseEngine,
        async route(req: S1GateRequest): Promise<S1GateDecision> {
          captured.push(req);
          return baseEngine.route(req);
        },
      };
      const sigBridge = createHttpBridge(spyEngine, { port: 0, host: "127.0.0.1" });
      const started = await sigBridge.start();
      try {
        const res = await fetch(`${started.url}/v1/systemone`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-session-id": "sys1-structured-signals",
          },
          body: JSON.stringify({
            state: {
              task: "Analyze radiology screenshot and patch EMR endpoint",
              previous_assistant: "Checking the uploaded chart and server logs.",
              step: {
                type: "tool_step",
                tool_call: { name: "bash" },
                last_tool_output_tail: "pytest: 2 failed",
              },
              signals: {
                estimated_tokens: 150_000,
                has_image: true,
                tool_history: true,
              },
            },
            questions: {
              route: {
                type: "choice",
                criteria: {
                  "ollama-cloud/gpt-oss:120b": "Low-cost 128K. Quota low (5% of 5h used).",
                  "anthropic/claude-sonnet-5-5": "Balanced coding. Quota high (92% of 5h used).",
                  "openai/gpt-6.2-sol": "General flagship. Quota low (12% of 5h used).",
                },
              },
            },
          }),
        });

        expect(res.status).toBe(200);
        const data = (await res.json()) as {
          answers: { route: { choice: string } };
        };
        // Excludes 128K non-multimodal gpt-oss:120b AND steers away from 92%-quota claude-sonnet-5-5
        expect(data.answers.route.choice).toBe("openai/gpt-6.2-sol");
        expect(res.headers.get("x-router-context-window")).toBeTruthy();

        expect(captured).toHaveLength(1);
        const forwarded = captured[0];
        expect(forwarded.estimatedContextTokens).toBe(150_000);
        expect(forwarded.quotaUtilization).toEqual({
          "ollama-cloud/gpt-oss:120b": 0.05,
          "anthropic/claude-sonnet-5-5": 0.92,
          "openai/gpt-6.2-sol": 0.12,
        });
        expect(Array.isArray(forwarded.body.tools)).toBe(true);
        const firstMsg = (forwarded.body.messages as Array<{ content: unknown }>)[0];
        expect(Array.isArray(firstMsg.content)).toBe(true);
        const blocks = firstMsg.content as Array<{ type: string; text?: string }>;
        expect(blocks[0].type).toBe("text");
        expect(blocks[0].text).toContain("Analyze radiology screenshot");
        expect(blocks[0].text).toContain("pytest: 2 failed");
        expect(blocks.some(b => b.type === "image_url")).toBe(true);
      } finally {
        await sigBridge.close();
      }
    });

    it("sanitizes oversized/newline-injected tool_call.name, falls through state.estimated_tokens: 0 to signals, and takes max multi-window quota %", async () => {
      const captured: S1GateRequest[] = [];
      const baseEngine = createS1Gate({
        defaultProfile: "balanced",
        classifier: createStubClassifier({
          domain: "terminal_devops",
          domainProbabilities: { terminal_devops: 1.0 },
          complexity: 0.4,
        }),
      });
      const spyEngine = {
        ...baseEngine,
        async route(req: S1GateRequest): Promise<S1GateDecision> {
          captured.push(req);
          return baseEngine.route(req);
        },
      };
      const sigBridge = createHttpBridge(spyEngine, { port: 0, host: "127.0.0.1" });
      const started = await sigBridge.start();
      try {
        const res = await fetch(`${started.url}/v1/systemone`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            state: {
              task: "Run diagnostics",
              estimated_tokens: 0,
              step: {
                type: "tool_step",
                tool_call: { name: `bash\r\nX-Injected: evil !@# ${"z".repeat(120)}` },
              },
              signals: {
                estimated_tokens: 150_000,
              },
            },
            questions: {
              route: {
                type: "choice",
                criteria: {
                  "anthropic/claude-sonnet-5-5":
                    "Quota 5h (15% of 5h used). Quota weekly cap (91% of 7d used).",
                  "openai/gpt-6.2-sol": "Quota (12% of 5h used).",
                },
              },
            },
          }),
        });

        expect(res.status).toBe(200);
        expect(captured).toHaveLength(1);
        const forwarded = captured[0];
        expect(forwarded.estimatedContextTokens).toBe(150_000);
        expect(forwarded.quotaUtilization).toEqual({
          "anthropic/claude-sonnet-5-5": 0.91,
          "openai/gpt-6.2-sol": 0.12,
        });
        const forwardedTools = forwarded.body.tools as Array<{ function: { name: string } }>;
        expect(forwardedTools).toHaveLength(1);
        const sanitizedName = forwardedTools[0].function.name;
        expect(sanitizedName.length).toBeLessThanOrEqual(64);
        expect(sanitizedName).toMatch(/^[\w.:-]+$/);
        expect(sanitizedName.startsWith("bashX-Injected:evil")).toBe(true);
      } finally {
        await sigBridge.close();
      }
    });

    it("excludes a non-multimodal large-context candidate when signals.has_image is true (isolated)", async () => {
      const mmEngine = createS1Gate({
        defaultProfile: "frugal",
        classifier: createStubClassifier({
          domain: "healthcare",
          domainProbabilities: { healthcare: 1.0 },
          complexity: 0.3,
        }),
        modelOverrides: {
          "ollama-cloud/gpt-oss:120b": {
            contextWindow: 1_000_000,
          },
        },
      });
      const mmBridge = createHttpBridge(mmEngine, { port: 0, host: "127.0.0.1" });
      const started = await mmBridge.start();
      try {
        const criteria = {
          "ollama-cloud/gpt-oss:120b": "Ultra-cheap 1M text-only",
          "anthropic/claude-sonnet-5-5": "Balanced 1M multimodal",
        };

        // Baseline without image -> ultra-cheap text-only 1M model wins
        const textRes = await fetch(`${started.url}/v1/systemone`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-session-id": "sys1-mm-text",
          },
          body: JSON.stringify({
            state: {
              task: "Summarize patient chart notes",
              signals: { has_image: false },
            },
            questions: { route: { type: "choice", criteria } },
          }),
        });
        expect(textRes.status).toBe(200);
        const textData = (await textRes.json()) as { answers: { route: { choice: string } } };
        expect(textData.answers.route.choice).toBe("ollama-cloud/gpt-oss:120b");

        // Isolated signals.has_image: true -> excludes non-multimodal 1M candidate
        const imgRes = await fetch(`${started.url}/v1/systemone`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-session-id": "sys1-mm-image",
          },
          body: JSON.stringify({
            state: {
              task: "Summarize patient chart notes",
              signals: { has_image: true },
            },
            questions: { route: { type: "choice", criteria } },
          }),
        });
        expect(imgRes.status).toBe(200);
        const imgData = (await imgRes.json()) as { answers: { route: { choice: string } } };
        expect(imgData.answers.route.choice).toBe("anthropic/claude-sonnet-5-5");
      } finally {
        await mmBridge.close();
      }
    });
  });

  describe("error handling: 400 malformed JSON, 404 unknown routes, 413 body size limits", () => {
    it("returns 400 Bad Request for malformed JSON and non-object JSON bodies", async () => {
      for (const path of ["/v1/route", "/v1/systemone"]) {
        const malformedRes = await fetch(`${baseUrl}${path}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: "{not-valid-json",
        });
        expect(malformedRes.status).toBe(400);

        const arrayRes = await fetch(`${baseUrl}${path}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(["unexpected", "array"]),
        });
        expect(arrayRes.status).toBe(400);
      }
    });

    it("returns 404 Not Found for unknown paths or unsupported methods", async () => {
      const checks = [
        fetch(`${baseUrl}/does-not-exist`),
        fetch(`${baseUrl}/v1/route`, { method: "GET" }),
        fetch(`${baseUrl}/v1/systemone`, { method: "GET" }),
        fetch(`${baseUrl}/health`, { method: "POST" }),
      ];

      for (const promise of checks) {
        const res = await promise;
        expect(res.status).toBe(404);
        const data = (await res.json()) as { error: string };
        expect(typeof data.error).toBe("string");
      }
    });

    it("enforces maxRouteBodyBytes and maxSystemOneBodyBytes with 413 Payload Too Large and respects defaultProfile", async () => {
      const customEngine = createS1Gate({ classifier: createStubClassifier() });
      const limitedBridge = createHttpBridge(customEngine, {
        port: 0,
        host: "127.0.0.1",
        defaultProfile: "frugal",
        maxRouteBodyBytes: 512,
        maxSystemOneBodyBytes: 256,
      });

      const started = await limitedBridge.start();
      try {
        // Verify defaultProfile override works on /v1/route and /v1/systemone
        const okRoute = await fetch(`${started.url}/v1/route`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            body: { messages: [{ role: "user", content: "hi" }] },
            availableModels: ["anthropic/claude-sonnet-5-5"],
          }),
        });
        expect(okRoute.status).toBe(200);
        const okDecision = (await okRoute.json()) as S1GateDecision;
        expect(okDecision.profile).toBe("frugal");

        // Exceed maxRouteBodyBytes (512 bytes) -> 413
        const bigRouteBody = JSON.stringify({
          body: { messages: [{ role: "user", content: "x".repeat(800) }] },
          availableModels: ["anthropic/claude-sonnet-5-5"],
        });
        const tooBigRoute = await fetch(`${started.url}/v1/route`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: bigRouteBody,
        });
        expect(tooBigRoute.status).toBe(413);
        const routeErr = (await tooBigRoute.json()) as { error: string };
        expect(routeErr.error.toLowerCase()).toContain("too large");

        // Exceed maxSystemOneBodyBytes (256 bytes) -> 413
        const bigSysOneBody = JSON.stringify({
          state: "y".repeat(400),
          questions: {
            route: {
              type: "choice",
              criteria: { "anthropic/claude-sonnet-5-5": "a" },
            },
          },
        });
        const tooBigSysOne = await fetch(`${started.url}/v1/systemone`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: bigSysOneBody,
        });
        expect(tooBigSysOne.status).toBe(413);
        const sysOneErr = (await tooBigSysOne.json()) as { error: string };
        expect(sysOneErr.error.toLowerCase()).toContain("too large");

        // Streaming chunked body without Content-Length exceeding maxSystemOneBodyBytes -> 413
        const chunkedStatus = await new Promise<number>((resolve, reject) => {
          const req = httpRequest(
            {
              hostname: started.host,
              port: started.port,
              path: "/v1/systemone",
              method: "POST",
              headers: {
                "content-type": "application/json",
                "transfer-encoding": "chunked",
              },
            },
            (res) => {
              res.resume();
              resolve(res.statusCode ?? 0);
            },
          );
          req.on("error", reject);
          req.write(`{"state":"${"z".repeat(350)}","questions":{}}`);
          req.end();
        });
        expect(chunkedStatus).toBe(413);
      } finally {
        await limitedBridge.close();
      }
    });

    it("enforces default 128 KiB limit on /v1/systemone and returns 400 on empty body", async () => {
      const emptyRes = await fetch(`${baseUrl}/v1/route`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "",
      });
      expect(emptyRes.status).toBe(400);

      const oversizedSysOne = await fetch(`${baseUrl}/v1/systemone`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          state: "a".repeat(140_000),
          questions: {
            route: { type: "choice", criteria: { "anthropic/claude-sonnet-5-5": "a" } },
          },
        }),
      });
      expect(oversizedSysOne.status).toBe(413);
    });
  });

  describe("security, resilience, and sanitization regressions", () => {
    it("configures strict HTTP server timeouts", () => {
      expect(bridge.server.headersTimeout).toBe(10_000);
      expect(bridge.server.requestTimeout).toBe(15_000);
      expect(bridge.server.keepAliveTimeout).toBe(5_000);
    });

    it("sanitizes outgoing decision headers against CRLF/control chars and caps length to 256", async () => {
      const baseEngine = createS1Gate({ classifier: createStubClassifier() });
      const wrappedEngine = {
        ...baseEngine,
        async route(req: S1GateRequest): Promise<S1GateDecision> {
          const decision = await baseEngine.route(req);
          return {
            ...decision,
            headers: {
              ...decision.headers,
              "X-Test-Crlf": "safe\r\nX-Injected: evil\x00\u2603",
              "X-Test-Long": "a".repeat(320),
            },
          };
        },
      };
      const secBridge = createHttpBridge(wrappedEngine, { port: 0, host: "127.0.0.1" });
      const started = await secBridge.start();
      try {
        const res = await fetch(`${started.url}/v1/route`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            body: { messages: [{ role: "user", content: "hi" }] },
            availableModels: ["anthropic/claude-sonnet-5-5"],
          }),
        });
        expect(res.status).toBe(200);
        expect(res.headers.get("x-injected")).toBeNull();
        expect(res.headers.get("x-test-crlf")).toBe("safeX-Injected: evil");
        expect(res.headers.get("x-test-long")).toBe("a".repeat(256));
      } finally {
        await secBridge.close();
      }
    });

    it("strips sensitive auth headers from incoming HTTP requests on /v1/route and /v1/systemone", async () => {
      const captured: S1GateRequest[] = [];
      const baseEngine = createS1Gate({ classifier: createStubClassifier() });
      const spyEngine = {
        ...baseEngine,
        async route(req: S1GateRequest): Promise<S1GateDecision> {
          captured.push(req);
          return baseEngine.route(req);
        },
      };
      const secBridge = createHttpBridge(spyEngine, { port: 0, host: "127.0.0.1" });
      const started = await secBridge.start();
      try {
        const sensitiveHeaders = {
          "content-type": "application/json",
          authorization: "Bearer sk-super-secret",
          cookie: "session=secret-cookie",
          "x-api-key": "key-secret-123",
          "proxy-authorization": "Basic dXNlcjpwYXNz",
          "x-claude-code-session-id": "keep-session-id",
        };

        const r1 = await fetch(`${started.url}/v1/route`, {
          method: "POST",
          headers: sensitiveHeaders,
          body: JSON.stringify({
            body: { messages: [{ role: "user", content: "hello" }] },
            availableModels: ["anthropic/claude-sonnet-5-5"],
          }),
        });
        expect(r1.status).toBe(200);

        const r2 = await fetch(`${started.url}/v1/systemone`, {
          method: "POST",
          headers: sensitiveHeaders,
          body: JSON.stringify({
            state: "check docker logs",
            questions: {
              route: {
                type: "choice",
                criteria: { "anthropic/claude-sonnet-5-5": "Coding" },
              },
            },
          }),
        });
        expect(r2.status).toBe(200);

        const r3 = await fetch(`${started.url}/v1/route`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "keep-session-id",
          },
          body: JSON.stringify({
            headers: {
              Authorization: "Bearer payload-secret",
              "X-API-Key": "payload-api-key",
              Cookie: "session=payload-cookie",
              "Proxy-Authorization": "Basic payload-proxy",
              "X-Safe-Header": "kept",
            },
            body: { messages: [{ role: "user", content: "hello" }] },
            availableModels: ["anthropic/claude-sonnet-5-5"],
          }),
        });
        expect(r3.status).toBe(200);

        expect(captured).toHaveLength(3);
        for (const req of captured) {
          expect(req.headers.authorization).toBeUndefined();
          expect(req.headers.cookie).toBeUndefined();
          expect(req.headers["x-api-key"]).toBeUndefined();
          expect(req.headers["proxy-authorization"]).toBeUndefined();
          expect(req.headers["x-claude-code-session-id"]).toBe("keep-session-id");
        }
        expect(captured[2].headers["x-safe-header"]).toBe("kept");
      } finally {
        await secBridge.close();
      }
    });

    it("blocks prototype pollution keys in headers, quotaUtilization, and systemone criteria", async () => {
      const captured: S1GateRequest[] = [];
      const baseEngine = createS1Gate({ classifier: createStubClassifier() });
      const spyEngine = {
        ...baseEngine,
        async route(req: S1GateRequest): Promise<S1GateDecision> {
          captured.push(req);
          return baseEngine.route(req);
        },
      };
      const secBridge = createHttpBridge(spyEngine, { port: 0, host: "127.0.0.1" });
      const started = await secBridge.start();
      try {
        const routeRes = await fetch(`${started.url}/v1/route`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body:
            '{"body":{"messages":[{"role":"user","content":"hi"}]},"availableModels":["anthropic/claude-sonnet-5-5"],"headers":{"__proto__":"evil","constructor":"evil","prototype":"evil","x-safe":"ok"},"quotaUtilization":{"__proto__":0.9,"constructor":0.8,"prototype":0.7,"openai":0.5}}',
        });
        expect(routeRes.status).toBe(200);
        expect(({} as Record<string, unknown>).polluted).toBeUndefined();
        expect(captured).toHaveLength(1);
        expect(Object.hasOwn(captured[0].headers, "__proto__")).toBe(false);
        expect(Object.hasOwn(captured[0].headers, "constructor")).toBe(false);
        expect(Object.hasOwn(captured[0].headers, "prototype")).toBe(false);
        expect(captured[0].headers["x-safe"]).toBe("ok");
        expect(captured[0].quotaUtilization).toEqual({ openai: 0.5 });

        const sysOneRes = await fetch(`${started.url}/v1/systemone`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body:
            '{"state":"test","questions":{"route":{"type":"choice","criteria":{"__proto__":"bad","constructor":"bad","prototype":"bad","anthropic/claude-sonnet-5-5":"good"}}}}',
        });
        expect(sysOneRes.status).toBe(200);
        const sysData = (await sysOneRes.json()) as {
          answers: { route: { choice: string; probabilities: Record<string, number> } };
        };
        expect(sysData.answers.route.choice).toBe("anthropic/claude-sonnet-5-5");
        expect(Object.keys(sysData.answers.route.probabilities)).toEqual([
          "anthropic/claude-sonnet-5-5",
        ]);
      } finally {
        await secBridge.close();
      }
    });

    it("closes the connection on 413 overflow for both declared Content-Length and chunked streams", async () => {
      const customEngine = createS1Gate({ classifier: createStubClassifier() });
      const limitedBridge = createHttpBridge(customEngine, {
        port: 0,
        host: "127.0.0.1",
        maxSystemOneBodyBytes: 128,
      });
      const started = await limitedBridge.start();
      try {
        const declared = await new Promise<{ status: number; connection: string | undefined }>(
          (resolve, reject) => {
            const body = JSON.stringify({ state: "x".repeat(256), questions: {} });
            const req = httpRequest(
              {
                hostname: started.host,
                port: started.port,
                path: "/v1/systemone",
                method: "POST",
                headers: {
                  "content-type": "application/json",
                  "content-length": String(Buffer.byteLength(body)),
                },
              },
              (res) => {
                res.resume();
                resolve({
                  status: res.statusCode ?? 0,
                  connection: res.headers.connection,
                });
              },
            );
            req.on("error", (err) => {
              // Socket destruction after 413 response is expected on write races
              if ((err as NodeJS.ErrnoException).code === "ECONNRESET") return;
              reject(err);
            });
            req.end(body);
          },
        );
        expect(declared.status).toBe(413);
        expect(declared.connection?.toLowerCase()).toBe("close");
      } finally {
        await limitedBridge.close();
      }
    });

    it("guarantees /v1/systemone choice belongs to candidateIds even when triggerModels causes passthrough", async () => {
      const restrictedEngine = createS1Gate({
        classifier: createStubClassifier(),
        triggerModels: ["custom-trigger-only"],
      });
      const triggerBridge = createHttpBridge(restrictedEngine, { port: 0, host: "127.0.0.1" });
      const started = await triggerBridge.start();
      try {
        const res = await fetch(`${started.url}/v1/systemone`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            state: "Debug nginx upstream timeout",
            questions: {
              route: {
                type: "choice",
                criteria: {
                  "anthropic/claude-sonnet-5-5": "Balanced coding",
                  "anthropic/claude-haiku-5-5": "Fast",
                },
              },
            },
          }),
        });
        expect(res.status).toBe(200);
        const data = (await res.json()) as {
          answers: { route: { choice: string; probabilities: Record<string, number> } };
          s1gate: { action: string; recommendedEffort?: string };
        };
        expect(data.answers.route.choice).toBe("anthropic/claude-sonnet-5-5");
        expect(Object.keys(data.answers.route.probabilities)).toEqual([
          "anthropic/claude-sonnet-5-5",
          "anthropic/claude-haiku-5-5",
        ]);
        expect(res.headers.get("x-router-model")).toBe("anthropic/claude-sonnet-5-5");
        expect(data.s1gate.action).toBe("passthrough");
        expect(data.s1gate.recommendedEffort).toBeUndefined();
        expect(res.headers.get("x-router-effort")).toBeNull();
      } finally {
        await triggerBridge.close();
      }
    });
  });
});

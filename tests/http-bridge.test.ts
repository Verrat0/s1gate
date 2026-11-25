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
        };
      };

      expect(data.model).toBe("s1gate");
      expect(data.usage).toEqual({ input_tokens: 0, output_tokens: 0 });
      expect(data.s1gate.action).toBe("route");
      expect(data.s1gate.domain).toBe("terminal_devops");
      expect(typeof data.s1gate.role).toBe("string");
      expect(data.s1gate.cachePinned).toBe(true);

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

        expect(captured).toHaveLength(2);
        for (const req of captured) {
          expect(req.headers.authorization).toBeUndefined();
          expect(req.headers.cookie).toBeUndefined();
          expect(req.headers["x-api-key"]).toBeUndefined();
          expect(req.headers["proxy-authorization"]).toBeUndefined();
          expect(req.headers["x-claude-code-session-id"]).toBe("keep-session-id");
        }
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
        };
        expect(data.answers.route.choice).toBe("anthropic/claude-sonnet-5-5");
        expect(Object.keys(data.answers.route.probabilities)).toEqual([
          "anthropic/claude-sonnet-5-5",
          "anthropic/claude-haiku-5-5",
        ]);
        expect(res.headers.get("x-router-model")).toBe("anthropic/claude-sonnet-5-5");
      } finally {
        await triggerBridge.close();
      }
    });
  });
});

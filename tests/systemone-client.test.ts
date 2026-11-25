import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  BUILTIN_BACKEND_CALIBRATIONS,
  HeuristicFallbackClassifier,
  SystemOneClient,
} from "../src/core/systemone-client.js";
import type { ClassifierRequest, DomainDefinition, RoutingEnvelope } from "../src/types.js";

const domains: DomainDefinition[] = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "../data/domains.json"), "utf8"),
);

function makeEnvelope(text: string): RoutingEnvelope {
  const byteLength = new TextEncoder().encode(text).byteLength;
  return {
    text,
    byteLength,
    estimatedTokens: Math.ceil(byteLength / 4),
    redactionCounts: { codeBlocks: 0, secrets: 0, pii: 0 },
  };
}

function makeRequest(
  text: string,
  overrides?: Partial<ClassifierRequest>,
): ClassifierRequest {
  return {
    envelope: makeEnvelope(text),
    domains,
    askOrchestrator: true,
    askEscalate: false,
    timeoutMs: 1500,
    ...overrides,
  };
}

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.useRealTimers();
});

describe("systemone-client: BUILTIN_BACKEND_CALIBRATIONS", () => {
  it("contains required built-in calibrations for tev1, clef, and jev", () => {
    expect(BUILTIN_BACKEND_CALIBRATIONS["tev1:0.8b"]).toEqual({
      complexityOffset: -0.15,
      noulOrchestratorThreshold: 0.65,
      noulEscalateThreshold: 0.70,
    });
    expect(BUILTIN_BACKEND_CALIBRATIONS["tev1:4b"]).toEqual({
      complexityOffset: 0.0,
      noulOrchestratorThreshold: 0.65,
      noulEscalateThreshold: 0.70,
    });
    expect(BUILTIN_BACKEND_CALIBRATIONS["clef:27b"]).toEqual({
      complexityOffset: 0.0,
      noulOrchestratorThreshold: 0.65,
      noulEscalateThreshold: 0.70,
    });
    expect(BUILTIN_BACKEND_CALIBRATIONS["jev-latest"]).toEqual({
      complexityOffset: 0.0,
      noulOrchestratorThreshold: 0.65,
      noulEscalateThreshold: 0.70,
    });
  });
});

describe("systemone-client: HeuristicFallbackClassifier", () => {
  it("deterministically classifies domains and outputs normalized probabilities", async () => {
    const clf = new HeuristicFallbackClassifier();

    const hc = await clf.classify(
      makeRequest("role: specialist\nprompt: Implement hospital SIMRS EMR triage and patient clinical coding ICD-10"),
    );
    expect(hc.domain).toBe("healthcare");
    expect(hc.fromFallback).toBe(true);
    const sum = Object.values(hc.domainProbabilities).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
    expect(hc.domainConfidence).toBeCloseTo(hc.domainProbabilities[hc.domain], 5);

    const devops = await clf.classify(
      makeRequest("role: specialist\nprompt: Debug Docker container bash script CI/CD pipeline on Linux systemd"),
    );
    expect(devops.domain).toBe("terminal_devops");

    const fe = await clf.classify(
      makeRequest("role: specialist\nprompt: Build responsive React Next.js Tailwind CSS navbar component"),
    );
    expect(fe.domain).toBe("frontend_web");
  });

  it("scores complex architecture/stuck prompts higher than trivial Q&A and handles askEscalate", async () => {
    const clf = new HeuristicFallbackClassifier();

    const simple = await clf.classify(
      makeRequest("role: auxiliary\nturns: 1\nprompt: Hi, what is the capital of France?"),
    );
    const complex = await clf.classify(
      makeRequest(
        [
          "role: orchestrator",
          "tools: read_file, edit_file, run_command, search_graph, grep, test_runner",
          "turns: 8",
          "prompt: Design a distributed multi-tenant architecture migration with concurrency safety and zero-downtime database sharding [CODE: ts, 85 lines]",
          "recent_error: RaceConditionError: deadlock detected during transaction replay",
          "stuck_loop: [level=hard] 3 identical failures and revert oscillation",
        ].join("\n"),
        { askOrchestrator: true, askEscalate: true },
      ),
    );

    expect(complex.complexity).toBeGreaterThan(simple.complexity);
    expect(complex.needsFlagshipOrchestrator).toBeGreaterThan(simple.needsFlagshipOrchestrator);
    expect(simple.shouldEscalate).toBeUndefined();
    expect(complex.shouldEscalate).toBeDefined();
    expect(complex.shouldEscalate!).toBeGreaterThan(0.5);
  });
});

describe("systemone-client: SystemOneClient", () => {
  it("sends a single batched /v1/systemone payload with custom Zero Trust headers and parses answers", async () => {
    let capturedUrl = "";
    let capturedInit: RequestInit | undefined;

    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      capturedUrl = String(input);
      capturedInit = init;
      return mockJsonResponse({
        answers: {
          domain: {
            type: "choice",
            choice: "software_engineering",
            confidence: 0.88,
            probabilities: { software_engineering: 0.88, terminal_devops: 0.12 },
          },
          complexity: {
            type: "score",
            score: 0.72,
          },
          needs_flagship_orchestrator: {
            type: "noul",
            noul: 0.81,
          },
          should_escalate: {
            type: "noul",
            noul: 0.76,
          },
        },
      });
    }) as unknown as typeof fetch;

    const client = new SystemOneClient({
      endpointUrl: "https://s1.example.com/v1/systemone",
      model: "clef:27b",
      apiKey: "secret-api-key",
      headers: {
        "CF-Access-Client-Id": "cf-id-123",
        "CF-Access-Client-Secret": "cf-secret-456",
      },
      fetchImpl,
    });

    const res = await client.classify(
      makeRequest("Refactor ORM transaction handling across 5 modules", {
        askOrchestrator: true,
        askEscalate: true,
      }),
    );

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(capturedUrl).toBe("https://s1.example.com/v1/systemone");
    expect(capturedInit?.method).toBe("POST");

    const hdrs = capturedInit?.headers as Record<string, string>;
    expect(hdrs["Content-Type"]).toBe("application/json");
    expect(hdrs["Authorization"]).toBe("Bearer secret-api-key");
    expect(hdrs["CF-Access-Client-Id"]).toBe("cf-id-123");
    expect(hdrs["CF-Access-Client-Secret"]).toBe("cf-secret-456");

    const body = JSON.parse(String(capturedInit?.body));
    expect(body.model).toBe("clef:27b");
    expect(body.state).toContain("Refactor ORM transaction handling");
    expect(body.questions.domain.type).toBe("choice");
    expect(typeof body.questions.domain.criteria.software_engineering).toBe("string");
    expect(body.questions.complexity).toMatchObject({
      type: "score",
      criteria: ["trivial", "simple", "moderate", "complex", "expert"],
    });
    expect(body.questions.needs_flagship_orchestrator.type).toBe("noul");
    expect(body.questions.should_escalate.type).toBe("noul");

    expect(res.fromFallback).toBe(false);
    expect(res.backend).toBe("clef:27b");
    expect(res.domain).toBe("software_engineering");
    expect(res.domainConfidence).toBeCloseTo(0.88, 5);
    expect(res.complexity).toBeCloseTo(0.72, 5);
    expect(res.needsFlagshipOrchestrator).toBeCloseTo(0.81, 5);
    expect(res.shouldEscalate).toBeCloseTo(0.76, 5);
    expect(client.getCircuitState()).toBe("closed");
  });

  it("omits optional noul questions when askOrchestrator/askEscalate are false", async () => {
    let sentBody: Record<string, unknown> = {};
    const fetchImpl = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      sentBody = JSON.parse(String(init?.body));
      return mockJsonResponse({
        answers: {
          domain: { choice: "general_chat", confidence: 0.9, probabilities: { general_chat: 0.9 } },
          complexity: { score: 0.15 },
        },
      });
    }) as unknown as typeof fetch;

    const client = new SystemOneClient({
      endpointUrl: "http://localhost:11434/v1/systemone",
      model: "jev-latest",
      fetchImpl,
    });

    const res = await client.classify(
      makeRequest("Quick syntax question", { askOrchestrator: false, askEscalate: false }),
    );
    const questions = sentBody.questions as Record<string, unknown>;
    expect(questions.needs_flagship_orchestrator).toBeUndefined();
    expect(questions.should_escalate).toBeUndefined();
    expect(res.needsFlagshipOrchestrator).toBe(0);
    expect(res.shouldEscalate).toBeUndefined();
  });

  it("unwraps Cloudflare { result: { answers } } envelope automatically", async () => {
    const fetchImpl = vi.fn(async () =>
      mockJsonResponse({
        success: true,
        result: {
          answers: {
            domain: {
              type: "choice",
              choice: "terminal_devops",
              confidence: 0.91,
              probabilities: { terminal_devops: 0.91, software_engineering: 0.09 },
            },
            complexity: { type: "score", score: 0.55 },
            needs_flagship_orchestrator: { type: "noul", noul: 0.42 },
          },
        },
      }),
    ) as unknown as typeof fetch;

    const client = new SystemOneClient({
      endpointUrl: "https://api.cloudflare.com/client/v4/accounts/abc/ai/run/@cf/typesafe/clef",
      model: "clef:27b",
      fetchImpl,
    });

    const res = await client.classify(makeRequest("Fix nginx reverse proxy config"));
    expect(res.fromFallback).toBe(false);
    expect(res.domain).toBe("terminal_devops");
    expect(res.complexity).toBeCloseTo(0.55, 5);
    expect(res.needsFlagshipOrchestrator).toBeCloseTo(0.42, 5);
  });

  it("applies built-in tev1:0.8b complexityOffset (-0.15) clamped to [0, 1]", async () => {
    const scores = [0.65, 0.05];
    let idx = 0;
    const fetchImpl = vi.fn(async () =>
      mockJsonResponse({
        answers: {
          domain: { choice: "software_engineering", confidence: 0.8, probabilities: { software_engineering: 0.8 } },
          complexity: { score: scores[idx++] },
          needs_flagship_orchestrator: { noul: 0.3 },
        },
      }),
    ) as unknown as typeof fetch;

    const client = new SystemOneClient({
      endpointUrl: "http://127.0.0.1:11434/v1/systemone",
      model: "tev1:0.8b",
      fetchImpl,
    });

    const first = await client.classify(makeRequest("Moderate task"));
    expect(first.complexity).toBeCloseTo(0.50, 5);

    const second = await client.classify(makeRequest("Tiny task"));
    expect(second.complexity).toBe(0);

    // Explicit calibration override takes precedence over built-in offset
    const overridden = new SystemOneClient({
      endpointUrl: "http://127.0.0.1:11434/v1/systemone",
      model: "ollama/tev1:0.8b",
      calibration: { complexityOffset: 0.1 },
      fetchImpl: vi.fn(async () =>
        mockJsonResponse({
          answers: {
            domain: { choice: "software_engineering", probabilities: { software_engineering: 0.9 } },
            complexity: { score: 0.95 },
            needs_flagship_orchestrator: { noul: 0.5 },
          },
        }),
      ) as unknown as typeof fetch,
    });
    const clampedHigh = await overridden.classify(makeRequest("Expert task"));
    expect(clampedHigh.complexity).toBe(1);
  });

  it("enforces 2..24 choice options by trimming >24 domains to first 24 and padding <2 domains", async () => {
    const capturedCriteriaCounts: number[] = [];
    const fetchImpl = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body));
      const criteria = body.questions.domain.criteria as Record<string, string>;
      capturedCriteriaCounts.push(Object.keys(criteria).length);
      const firstKey = Object.keys(criteria)[0];
      return mockJsonResponse({
        answers: {
          domain: { choice: firstKey, confidence: 0.9, probabilities: { [firstKey]: 0.9 } },
          complexity: { score: 0.4 },
          needs_flagship_orchestrator: { noul: 0.2 },
        },
      });
    }) as unknown as typeof fetch;

    const client = new SystemOneClient({
      endpointUrl: "http://localhost:11434/v1/systemone",
      model: "tev1:4b",
      fetchImpl,
    });

    const thirtyDomains: DomainDefinition[] = Array.from({ length: 30 }, (_, i) => ({
      id: `domain_${i + 1}`,
      description: `Description for domain ${i + 1}`,
      benchmarkChain: ["gpqa"],
    }));

    await client.classify(makeRequest("Test many domains", { domains: thirtyDomains }));
    await client.classify(makeRequest("Test single domain", { domains: [thirtyDomains[0]] }));

    expect(capturedCriteriaCounts[0]).toBe(24);
    expect(capturedCriteriaCounts[1]).toBe(2);
  });

  it("fails open without throwing on timeout, HTTP 500, network error, and malformed JSON", async () => {
    const scenarios: Array<{ impl: typeof fetch; expectedReason: string }> = [
      {
        impl: vi.fn(async () => {
          const err = new Error("signal timed out");
          err.name = "TimeoutError";
          throw err;
        }) as unknown as typeof fetch,
        expectedReason: "timeout",
      },
      {
        impl: vi.fn(async () => new Response("Internal Server Error", { status: 500 })) as unknown as typeof fetch,
        expectedReason: "http_500",
      },
      {
        impl: vi.fn(async () => {
          throw new TypeError("fetch failed: ECONNREFUSED");
        }) as unknown as typeof fetch,
        expectedReason: "network",
      },
      {
        impl: vi.fn(async () => new Response("{not valid json", { status: 200 })) as unknown as typeof fetch,
        expectedReason: "malformed",
      },
      {
        impl: vi.fn(async () => mockJsonResponse({ unexpected: true })) as unknown as typeof fetch,
        expectedReason: "malformed",
      },
    ];

    for (const { impl, expectedReason } of scenarios) {
      const client = new SystemOneClient({
        endpointUrl: "http://localhost:11434/v1/systemone",
        model: "tev1:4b",
        fetchImpl: impl,
      });

      const res = await client.classify(makeRequest("Fix SQL migration bug in backend"));
      expect(res.fromFallback).toBe(true);
      expect(res.fallbackReason).toBe(expectedReason);
      expect(res.domain).toBeTruthy();
    }
  });

  it("opens circuit breaker after threshold consecutive failures, short-circuits with zero fetch, then recovers via half_open", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-11-25T10:00:00.000Z"));

    let shouldFail = true;
    const fetchImpl = vi.fn(async () => {
      if (shouldFail) {
        return new Response("Service Unavailable", { status: 503 });
      }
      return mockJsonResponse({
        answers: {
          domain: { choice: "software_engineering", confidence: 0.92, probabilities: { software_engineering: 0.92 } },
          complexity: { score: 0.5 },
          needs_flagship_orchestrator: { noul: 0.3 },
        },
      });
    }) as unknown as typeof fetch;

    const client = new SystemOneClient({
      endpointUrl: "http://localhost:11434/v1/systemone",
      model: "clef:27b",
      circuitBreakerThreshold: 3,
      circuitBreakerCooldownMs: 30_000,
      fetchImpl,
    });

    expect(client.getCircuitState()).toBe("closed");

    // 3 consecutive failures trip the breaker
    for (let i = 0; i < 3; i++) {
      const res = await client.classify(makeRequest("Debug backend API"));
      expect(res.fromFallback).toBe(true);
      expect(res.fallbackReason).toBe("http_503");
    }
    expect(fetchImpl).toHaveBeenCalledTimes(3);
    expect(client.getCircuitState()).toBe("open");

    // While open, no fetch is called at all
    const openRes = await client.classify(makeRequest("Debug backend API"));
    expect(openRes.fromFallback).toBe(true);
    expect(openRes.fallbackReason).toBe("circuit_open");
    expect(fetchImpl).toHaveBeenCalledTimes(3);

    // Advance past cooldown -> half_open
    vi.advanceTimersByTime(30_000);
    expect(client.getCircuitState()).toBe("half_open");

    // Failed half_open probe re-opens circuit immediately
    const probeFail = await client.classify(makeRequest("Debug backend API"));
    expect(probeFail.fromFallback).toBe(true);
    expect(probeFail.fallbackReason).toBe("http_503");
    expect(fetchImpl).toHaveBeenCalledTimes(4);
    expect(client.getCircuitState()).toBe("open");

    // Advance past cooldown again -> half_open -> probe succeeds -> closed
    vi.advanceTimersByTime(30_000);
    expect(client.getCircuitState()).toBe("half_open");
    shouldFail = false;

    const recovered = await client.classify(makeRequest("Debug backend API"));
    expect(recovered.fromFallback).toBe(false);
    expect(recovered.domain).toBe("software_engineering");
    expect(fetchImpl).toHaveBeenCalledTimes(5);
    expect(client.getCircuitState()).toBe("closed");
  });

  it("allows only a single in-flight probe in half_open state, cancels non-OK response bodies, and sanitizes invalid timeouts", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-11-25T10:00:00.000Z"));

    const cancelSpy = vi.fn(async () => undefined);
    let resolveProbe: ((res: Response) => void) | undefined;
    let callNum = 0;

    const fetchImpl = vi.fn(async () => {
      callNum += 1;
      if (callNum === 1) {
        return {
          ok: false,
          status: 502,
          body: { cancel: cancelSpy },
        } as unknown as Response;
      }
      return new Promise<Response>((resolve) => {
        resolveProbe = resolve;
      });
    }) as unknown as typeof fetch;

    const client = new SystemOneClient({
      endpointUrl: "http://localhost:11434/v1/systemone",
      model: "clef:27b",
      timeoutMs: Number.NaN,
      circuitBreakerThreshold: 1,
      circuitBreakerCooldownMs: 10_000,
      fetchImpl,
    });

    // Invalid NaN/Infinity timeouts must not throw and non-OK response cancels body
    const failRes = await client.classify(
      makeRequest("Debug backend API", { timeoutMs: Number.POSITIVE_INFINITY }),
    );
    expect(failRes.fromFallback).toBe(true);
    expect(failRes.fallbackReason).toBe("http_502");
    expect(cancelSpy).toHaveBeenCalledTimes(1);
    expect(client.getCircuitState()).toBe("open");

    // Advance past cooldown -> half_open
    vi.advanceTimersByTime(10_000);
    expect(client.getCircuitState()).toBe("half_open");

    // Launch two concurrent calls in half_open: only first hits fetch, second fails open with circuit_open
    const firstProbe = client.classify(makeRequest("Probe 1", { timeoutMs: Number.NaN }));
    const secondConcurrent = await client.classify(makeRequest("Probe 2"));

    expect(secondConcurrent.fromFallback).toBe(true);
    expect(secondConcurrent.fallbackReason).toBe("circuit_open");
    expect(fetchImpl).toHaveBeenCalledTimes(2);

    // Resolve the single probe -> circuit closes
    resolveProbe?.(
      mockJsonResponse({
        answers: {
          domain: { choice: "software_engineering", confidence: 0.9, probabilities: { software_engineering: 0.9 } },
          complexity: { score: 0.4 },
          needs_flagship_orchestrator: { noul: 0.2 },
        },
      }),
    );
    const probeRes = await firstProbe;
    expect(probeRes.fromFallback).toBe(false);
    expect(client.getCircuitState()).toBe("closed");
  });
});

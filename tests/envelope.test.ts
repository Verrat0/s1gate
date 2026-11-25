import { describe, expect, it } from "vitest";
import { buildRoutingEnvelope, redactSensitiveText } from "../src/core/envelope.js";
import type { S1GateRequest, StuckLoopReport } from "../src/types.js";

const utf8Bytes = (s: string): number => new TextEncoder().encode(s).byteLength;

describe("envelope: redactSensitiveText", () => {
  it("replaces fenced code blocks with structural stubs and counts them", () => {
    const raw = [
      "Please refactor this helper:",
      "```typescript",
      "export function add(a: number, b: number) {",
      "  return a + b;",
      "}",
      "```",
      "And check this config:",
      "```",
      "key=value",
      "```",
    ].join("\n");

    const res = redactSensitiveText(raw);
    expect(res.counts.codeBlocks).toBe(2);
    expect(res.text).toContain("[CODE: typescript, 3 lines]");
    expect(res.text).toContain("[CODE: text, 1 lines]");
    expect(res.text).not.toContain("return a + b");
    expect(res.text).not.toContain("key=value");
  });

  it("scrubs API keys, tokens, JWTs, PEM keys, DB URIs, and AWS keys as [REDACTED:SECRET]", () => {
    const pemBegin = ["-----BEGIN ", "RSA PRIVATE KEY-----"].join("");
    const pemEnd = ["-----END ", "RSA PRIVATE KEY-----"].join("");
    const pem = [pemBegin, "MIIEowIBAAKCAQEAtestkeymaterial1234567890", pemEnd].join("\n");

    const fakeOpenAi = ["sk", "proj", "abc1234567890xyz"].join("-");
    const fakeAnthropic = ["sk", "ant", "api03", "secretkeyvalue99999"].join("-");
    const fakeGhClassic = ["ghp", "1234567890abcdef123456"].join("_");
    const fakeGhPat = ["github", "pat", "11ABCDEF0123456789", "xyz987654"].join("_");
    const fakeAws = ["AKIA", "IOSFODNN7EXAMPLE"].join("");

    const raw = [
      `OpenAI: ${fakeOpenAi}`,
      `Anthropic: ${fakeAnthropic}`,
      `GitHub classic: ${fakeGhClassic}`,
      `GitHub PAT: ${fakeGhPat}`,
      "JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
      "Header: Bearer supersecret_token_987654321",
      "PG: postgres://admin:s3cr3t@db.internal:5432/simrs",
      "MySQL: mysql://root:pass123@127.0.0.1:3306/app",
      "Mongo: mongodb+srv://user:pw@cluster0.example.net/prod",
      `AWS: ${fakeAws}`,
      pem,
    ].join("\n");

    const res = redactSensitiveText(raw);
    expect(res.counts.secrets).toBe(11);
    expect(res.counts.pii).toBe(0);
    expect(res.text).not.toContain(fakeOpenAi);
    expect(res.text).not.toContain(fakeAnthropic);
    expect(res.text).not.toContain(fakeGhClassic);
    expect(res.text).not.toContain(fakeGhPat);
    expect(res.text).not.toContain("eyJhbGci");
    expect(res.text).not.toContain("supersecret_token");
    expect(res.text).not.toContain("postgres://");
    expect(res.text).not.toContain("mysql://");
    expect(res.text).not.toContain("mongodb+srv://");
    expect(res.text).not.toContain(fakeAws);
    expect(res.text).not.toContain("PRIVATE KEY");
    const secretTags = res.text.match(/\[REDACTED:SECRET\]/g) ?? [];
    expect(secretTags).toHaveLength(11);
  });

  it("scrubs emails, IPv4, URLs, phones, Indonesian NIK, US SSN, and MRN/RM patterns as [REDACTED:PII]", () => {
    const raw = [
      "Contact: dr.budi@hospital.co.id",
      "Server IP: 192.168.10.42",
      "Portal: https://emr.example.org/patient/detail?id=99",
      "Phone ID: +62 812-3456-7890 and 0812-9876-5432",
      "Phone US: (555) 867-5309",
      "NIK: 3273011205900001",
      "SSN: 123-45-6789",
      "Patient MRN: 0045821 and RM-992104 and Patient ID: 77120",
    ].join("\n");

    const res = redactSensitiveText(raw);
    expect(res.counts.secrets).toBe(0);
    expect(res.counts.pii).toBe(11);
    expect(res.text).not.toContain("dr.budi@hospital.co.id");
    expect(res.text).not.toContain("192.168.10.42");
    expect(res.text).not.toContain("https://emr.example.org");
    expect(res.text).not.toContain("3456-7890");
    expect(res.text).not.toContain("9876-5432");
    expect(res.text).not.toContain("867-5309");
    expect(res.text).not.toContain("3273011205900001");
    expect(res.text).not.toContain("123-45-6789");
    expect(res.text).not.toContain("0045821");
    expect(res.text).not.toContain("992104");
    expect(res.text).not.toContain("77120");
  });

  it("preserves benign technical text without false positives", () => {
    const raw = "Fix Odoo ORM form view date 2026-11-25 at 12:34:56 for version 18.0 and port 8069";
    const res = redactSensitiveText(raw);
    expect(res.text).toBe(raw);
    expect(res.counts).toEqual({ codeBlocks: 0, secrets: 0, pii: 0 });
  });

  it("replaces tilde code fences and unclosed trailing code fences", () => {
    const raw = [
      "Tilde block:",
      "~~~python",
      "def secret_fn():",
      "    return 42",
      "~~~",
      "Truncated paste:",
      "```json",
      '{ "apiKey": "raw_unclosed_code_body" }',
    ].join("\n");

    const res = redactSensitiveText(raw);
    expect(res.counts.codeBlocks).toBe(2);
    expect(res.text).toContain("[CODE: python, 2 lines]");
    expect(res.text).toContain("[CODE: json, 1 lines]");
    expect(res.text).not.toContain("secret_fn");
    expect(res.text).not.toContain("raw_unclosed_code_body");
  });

  it("scrubs Basic auth, generic secret assignments, Google/Slack/Stripe/npm/HF tokens, credit cards, and IPv6", () => {
    const fakeGoogle = ["AIza", "SyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe"].join("");
    const fakeSlack = ["xoxb", "123456789012", "1234567890123", "abcdef123456"].join("-");
    const fakeStripeLive = ["sk", "live", "51ABCdef1234567890"].join("_");
    const fakeStripeTest = ["sk", "test", "99XYZabc123456"].join("_");
    const fakeNpm = ["npm", "AbCdEfGhIjKlMnOpQrStUvWxYz123456"].join("_");
    const fakeHf = ["hf", "abcdefghijklmnopqrstuvwxyz123456"].join("_");

    const raw = [
      "Authorization: Basic dXNlcjpwYXNzMTIz",
      "PASSWORD=superSecretValue123",
      'API_KEY: "my_custom_api_key_98765"',
      "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
      `Google: ${fakeGoogle}`,
      `Slack: ${fakeSlack}`,
      `Stripe: ${fakeStripeLive} and ${fakeStripeTest}`,
      `NPM: ${fakeNpm}`,
      `HF: ${fakeHf}`,
      "Card1: 4532-1234-5678-9010 Card2: 5412 7512 3412 3456",
      "IPv6 full: 2001:0db8:85a3:0000:0000:8a2e:0370:7334",
      "IPv6 short: 2001:db8::1 and ::1 and ::ffff:192.0.2.128",
    ].join("\n");

    const res = redactSensitiveText(raw);
    expect(res.counts.secrets).toBe(10);
    expect(res.counts.pii).toBe(6);
    expect(res.text).not.toContain("dXNlcjpwYXNzMTIz");
    expect(res.text).not.toContain("superSecretValue123");
    expect(res.text).not.toContain("my_custom_api_key_98765");
    expect(res.text).not.toContain("wJalrXUtnFEMI");
    expect(res.text).not.toContain(fakeGoogle);
    expect(res.text).not.toContain(fakeSlack);
    expect(res.text).not.toContain(fakeStripeLive);
    expect(res.text).not.toContain(fakeStripeTest);
    expect(res.text).not.toContain(fakeNpm);
    expect(res.text).not.toContain(fakeHf);
    expect(res.text).not.toContain("4532-1234-5678-9010");
    expect(res.text).not.toContain("5412 7512 3412 3456");
    expect(res.text).not.toContain("2001:0db8:85a3");
    expect(res.text).not.toContain("2001:db8::1");
    expect(res.text).not.toContain("::1");
    expect(res.text).not.toContain("::ffff:192.0.2.128");
  });
});

describe("envelope: buildRoutingEnvelope", () => {
  it("extracts Anthropic metadata, caps tool names at 12 without schemas, and redacts prompt + tool error", () => {
    const tools = Array.from({ length: 15 }, (_, i) => ({
      name: `tool_${i + 1}`,
      description: `SENSITIVE_DESCRIPTION_SHOULD_NOT_APPEAR_${i + 1}`,
      input_schema: {
        type: "object",
        properties: { secret_schema_field: { type: "string" } },
      },
    }));

    const req: S1GateRequest = {
      protocol: "anthropic_messages",
      harness: "claude_code",
      headers: {},
      availableModels: ["anthropic/claude-sonnet-5-5"],
      body: {
        tools,
        messages: [
          { role: "user", content: "First message" },
          { role: "assistant", content: [{ type: "text", text: "Working on it." }] },
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Debug postgres://u:p@db/app for patient MRN: 123456:\n```py\nraise ValueError('bad')\n```",
              },
            ],
          },
          { role: "assistant", content: [{ type: "tool_use", name: "tool_1", input: {} }] },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                is_error: true,
                content: `TypeError: failed connecting to 10.0.0.5 with ${["sk", "ant", "api03", "abcdefghijklm"].join("-")}`,
              },
            ],
          },
        ],
      },
    };

    const stuckLoop: StuckLoopReport = {
      level: "soft",
      reasons: ["repeated_error"],
      identicalErrorCount: 2,
      identicalToolFailCount: 2,
      revertOscillation: false,
      tddEvasion: false,
      apologyStreak: 0,
      hotspotEdits: 3,
      errorCountTrendingDown: false,
      telemetrySummary: "2 identical tool failures on same file",
    };

    const env = buildRoutingEnvelope(req, { role: "orchestrator", stuckLoop });
    expect(env.text).toContain("protocol: anthropic_messages");
    expect(env.text).toContain("harness: claude_code");
    expect(env.text).toContain("role: orchestrator");
    expect(env.text).toContain("tool_1");
    expect(env.text).toContain("tool_12");
    expect(env.text).not.toContain("tool_13");
    expect(env.text).not.toContain("secret_schema_field");
    expect(env.text).not.toContain("SENSITIVE_DESCRIPTION");
    expect(env.text).toContain("[CODE: py, 1 lines]");
    expect(env.text).toContain("[REDACTED:SECRET]");
    expect(env.text).toContain("[REDACTED:PII]");
    expect(env.text).toContain("2 identical tool failures on same file");
    expect(env.text).toContain("TypeError");
    expect(env.text).not.toContain("10.0.0.5");
    expect(env.text).not.toContain("abcdefghijklm");
    expect(env.redactionCounts.codeBlocks).toBe(1);
    expect(env.redactionCounts.secrets).toBe(2);
    expect(env.redactionCounts.pii).toBe(2);
    expect(env.byteLength).toBe(utf8Bytes(env.text));
    expect(env.byteLength).toBeLessThanOrEqual(8192);
    expect(env.estimatedTokens).toBeGreaterThan(0);
  });

  it("handles openai_chat and openai_responses formats cleanly", () => {
    const chatReq: S1GateRequest = {
      protocol: "openai_chat",
      harness: "aider",
      headers: {},
      availableModels: ["openai/gpt-6.2"],
      body: {
        tools: [{ type: "function", function: { name: "run_shell", parameters: { type: "object" } } }],
        messages: [
          { role: "system", content: "You are helpful." },
          { role: "user", content: "Fix docker compose networking issue" },
          { role: "tool", content: "Error: container exited with code 1" },
        ],
      },
    };
    const chatEnv = buildRoutingEnvelope(chatReq, { role: "specialist" });
    expect(chatEnv.text).toContain("protocol: openai_chat");
    expect(chatEnv.text).toContain("run_shell");
    expect(chatEnv.text).toContain("Fix docker compose networking issue");
    expect(chatEnv.text).toContain("container exited with code 1");

    const respReq: S1GateRequest = {
      protocol: "openai_responses",
      harness: "codex",
      headers: {},
      availableModels: ["openai/gpt-6.2"],
      body: {
        tools: [{ type: "function", name: "apply_patch" }],
        input: "Design a distributed rate limiter for email user@test.com",
      },
    };
    const respEnv = buildRoutingEnvelope(respReq);
    expect(respEnv.text).toContain("protocol: openai_responses");
    expect(respEnv.text).toContain("apply_patch");
    expect(respEnv.text).toContain("Design a distributed rate limiter");
    expect(respEnv.text).not.toContain("user@test.com");
    expect(respEnv.redactionCounts.pii).toBe(1);

    const respArrayReq: S1GateRequest = {
      protocol: "openai_responses",
      harness: "codex",
      headers: {},
      availableModels: ["openai/gpt-6.2"],
      body: {
        tools: [{ type: "web_search" }],
        input: [
          { role: "user", content: [{ type: "input_text", text: "Fix build failure" }] },
          { type: "function_call_output", output: "SyntaxError: unexpected token at line 42" },
        ],
      },
    };
    const respArrayEnv = buildRoutingEnvelope(respArrayReq);
    expect(respArrayEnv.text).toContain("web_search");
    expect(respArrayEnv.text).toContain("Fix build failure");
    expect(respArrayEnv.text).toContain("SyntaxError: unexpected token");
  });

  it("strictly enforces default 8192 byte cap and custom small maxBytes even with multi-byte UTF-8", () => {
    const hugePrompt = "🚀 日本語テスト architecture planning ".repeat(2000);
    const req: S1GateRequest = {
      protocol: "openai_chat",
      headers: {},
      availableModels: ["anthropic/claude-sonnet-5-5"],
      body: {
        messages: [{ role: "user", content: hugePrompt }],
      },
    };

    const defaultEnv = buildRoutingEnvelope(req);
    expect(defaultEnv.byteLength).toBe(utf8Bytes(defaultEnv.text));
    expect(defaultEnv.byteLength).toBeLessThanOrEqual(8192);

    const tinyEnv = buildRoutingEnvelope(req, { maxBytes: 256 });
    expect(tinyEnv.byteLength).toBe(utf8Bytes(tinyEnv.text));
    expect(tinyEnv.byteLength).toBeLessThanOrEqual(256);
    expect(tinyEnv.text).toContain("protocol: openai_chat");
  });
});

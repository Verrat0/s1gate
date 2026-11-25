import { describe, expect, it } from "vitest";
import {
  inspectStuckLoop,
  isRatchetLocked,
  normalizeErrorSignature,
} from "../src/core/stuck-loop.js";
import type { SessionPin } from "../src/types.js";

function makePin(overrides: Partial<SessionPin> = {}): SessionPin {
  return {
    sessionKey: "sess-1",
    model: "anthropic/claude-sonnet-5-5",
    role: "orchestrator",
    profile: "balanced",
    domain: "coding",
    complexity: 0.5,
    lastEstimatedTokens: 4000,
    turnCount: 5,
    escalatedAtTurn: null,
    escalationCount: 0,
    pinnedAtMs: 1_000_000,
    lastUsedAtMs: 1_001_000,
    ...overrides,
  };
}

function anthropicToolTurn(
  id: string,
  name: string,
  input: Record<string, unknown>,
  result: string,
  options: { text?: string; isError?: boolean } = {},
) {
  const assistantContent: unknown[] = [];
  if (options.text) {
    assistantContent.push({ type: "text", text: options.text });
  }
  assistantContent.push({ type: "tool_use", id, name, input });

  return [
    { role: "assistant", content: assistantContent },
    {
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: id,
          content: result,
          ...(options.isError !== undefined ? { is_error: options.isError } : {}),
        },
      ],
    },
  ];
}

function openaiChatToolTurn(
  id: string,
  name: string,
  args: string,
  result: string,
  text?: string,
) {
  return [
    {
      role: "assistant",
      content: text ?? null,
      tool_calls: [
        {
          id,
          type: "function",
          function: { name, arguments: args },
        },
      ],
    },
    {
      role: "tool",
      tool_call_id: id,
      name,
      content: result,
    },
  ];
}

describe("isRatchetLocked", () => {
  it("returns false when session has never been escalated", () => {
    expect(isRatchetLocked(makePin({ turnCount: 10, escalatedAtTurn: null }))).toBe(false);
  });

  it("locks for default 4 turns after escalation and unlocks at turn diff >= 4", () => {
    expect(isRatchetLocked(makePin({ turnCount: 5, escalatedAtTurn: 5 }))).toBe(true);
    expect(isRatchetLocked(makePin({ turnCount: 8, escalatedAtTurn: 5 }))).toBe(true);
    expect(isRatchetLocked(makePin({ turnCount: 9, escalatedAtTurn: 5 }))).toBe(false);
    expect(isRatchetLocked(makePin({ turnCount: 14, escalatedAtTurn: 5 }))).toBe(false);
  });

  it("respects a custom minLockTurns value", () => {
    expect(isRatchetLocked(makePin({ turnCount: 6, escalatedAtTurn: 5 }), 2)).toBe(true);
    expect(isRatchetLocked(makePin({ turnCount: 7, escalatedAtTurn: 5 }), 2)).toBe(false);
    expect(isRatchetLocked(makePin({ turnCount: 9, escalatedAtTurn: 5 }), 6)).toBe(true);
  });
});

describe("normalizeErrorSignature", () => {
  it("strips timestamps, ISO dates, PIDs, hex addresses, temp paths, and volatile line/col numbers while keeping core info", () => {
    const errA =
      "2026-11-25T11:25:39.123Z 11:25:39 [pid 4821] TypeError: Cannot read properties of undefined (reading 'token') at /Users/dev/project/src/core/auth.ts:123:45 (0x7fff5fbff8a0) tmp=/tmp/run-9a8b7c/cache.json";
    const errB =
      "2026-11-26T18:04:11.998Z 18:04:11 [pid 9102] TypeError: Cannot read properties of undefined (reading 'token') at /opt/ci/workspace/src/core/auth.ts:299:12 (0x7fff5fbff9b4) tmp=/tmp/run-1x2y3z/other.json";

    const sigA = normalizeErrorSignature(errA);
    const sigB = normalizeErrorSignature(errB);

    expect(sigA).toBe(sigB);
    expect(sigA).toContain("TypeError");
    expect(sigA).toContain("Cannot read properties of undefined");
    expect(sigA).toContain("auth.ts");
    expect(sigA).not.toContain("2026-11-25");
    expect(sigA).not.toContain("4821");
    expect(sigA).not.toContain("0x7fff5fbff8a0");
    expect(sigA).not.toContain("/tmp/");
    expect(sigA).not.toContain(":123:45");
  });

  it("keeps distinct errors in different files or with different error classes distinct", () => {
    const sigA = normalizeErrorSignature("src/core/auth.ts(10,5): error TS2322: Type 'string' is not assignable to type 'number'");
    const sigB = normalizeErrorSignature("src/core/billing.ts(42,8): error TS2339: Property 'amount' does not exist on type 'Invoice'");

    expect(sigA).not.toBe(sigB);
    expect(sigA).toContain("auth.ts");
    expect(sigB).toContain("billing.ts");
  });

  it("normalizes Python tracebacks and colon-terminated file line numbers consistently", () => {
    const pyA = 'File "/workspace/app/services/calc.py", line 14, in run\nValueError: invalid rate';
    const pyB = 'File "/ci/repo/services/calc.py", line 99, in run\nValueError: invalid rate';
    expect(normalizeErrorSignature(pyA)).toBe(normalizeErrorSignature(pyB));
    expect(normalizeErrorSignature(pyA)).toContain("calc.py");
    expect(normalizeErrorSignature(pyA)).toContain("ValueError");

    const tsA = "src/core/worker.ts:12:5: error TS2322: bad type";
    const tsB = "/abs/path/src/core/worker.ts:88:19: error TS2322: bad type";
    expect(normalizeErrorSignature(tsA)).toBe(normalizeErrorSignature(tsB));
  });
});

describe("inspectStuckLoop: Hard 0ms triggers", () => {
  it("flags level=hard when identical normalized error repeats >= 3 times (Anthropic Messages)", () => {
    const body = {
      messages: [
        { role: "user", content: "Fix the auth middleware bug" },
        ...anthropicToolTurn(
          "t1",
          "bash",
          { command: "bunx vitest run tests/auth.test.ts" },
          "2026-11-25T10:00:01Z pid=101 TypeError: Cannot read properties of undefined at /app/src/auth.ts:41:12 (0x10a0)",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "t2",
          "edit_file",
          { path: "src/auth.ts", old_str: "a", new_str: "b" },
          "Updated src/auth.ts",
        ),
        ...anthropicToolTurn(
          "t3",
          "bash",
          { command: "bunx vitest run tests/auth.test.ts" },
          "2026-11-25T10:00:15Z pid=102 TypeError: Cannot read properties of undefined at /app/src/auth.ts:44:9 (0x20b0)",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "t4",
          "edit_file",
          { path: "src/auth.ts", old_str: "b", new_str: "c" },
          "Updated src/auth.ts",
        ),
        ...anthropicToolTurn(
          "t5",
          "bash",
          { command: "bunx vitest run tests/auth.test.ts" },
          "2026-11-25T10:00:29Z pid=103 TypeError: Cannot read properties of undefined at /app/src/auth.ts:47:3 (0x30c0)",
          { isError: true },
        ),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("hard");
    expect(report.identicalErrorCount).toBeGreaterThanOrEqual(3);
    expect(report.errorCountTrendingDown).toBe(false);
    expect(report.reasons.some((r) => r.includes("identical_error"))).toBe(true);
    expect(report.telemetrySummary.length).toBeLessThan(300);
  });

  it("flags level=hard when identical (tool_name, canonical_sorted_args) fails >= 3 times (OpenAI Chat)", () => {
    const body = {
      messages: [
        { role: "user", content: "Update the config file" },
        ...openaiChatToolTurn(
          "c1",
          "edit_file",
          JSON.stringify({ path: "src/config.ts", old_str: "port: 3000", new_str: "port: 8080" }),
          "Error: old_str not found in src/config.ts (attempt 1)",
        ),
        ...openaiChatToolTurn(
          "c2",
          "edit_file",
          JSON.stringify({ new_str: "port: 8080", path: "src/config.ts", old_str: "port: 3000" }),
          "Error: target block mismatch in src/config.ts (attempt 2)",
        ),
        ...openaiChatToolTurn(
          "c3",
          "edit_file",
          JSON.stringify({ old_str: "port: 3000", new_str: "port: 8080", path: "src/config.ts" }),
          "Error: replacement failed in src/config.ts (attempt 3)",
        ),
      ],
    };

    const report = inspectStuckLoop(body, "openai_chat");
    expect(report.level).toBe("hard");
    expect(report.identicalToolFailCount).toBeGreaterThanOrEqual(3);
    expect(report.reasons.some((r) => r.includes("identical_tool_fail"))).toBe(true);
    expect(report.telemetrySummary.length).toBeLessThan(300);
  });

  it("flags level=hard on revert oscillation (new_str equals earlier old_str on same path)", () => {
    const body = {
      messages: [
        { role: "user", content: "Fix rounding calculation" },
        ...anthropicToolTurn(
          "t1",
          "edit_file",
          { path: "src/math.ts", old_str: "return Math.floor(x);", new_str: "return Math.round(x);" },
          "Applied edit",
        ),
        ...anthropicToolTurn(
          "t2",
          "edit_file",
          { path: "./src/math.ts", old_str: "return Math.round(x);", new_str: "return Math.floor(x);" },
          "Applied edit",
        ),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("hard");
    expect(report.revertOscillation).toBe(true);
    expect(report.reasons.some((r) => r.includes("revert_oscillation"))).toBe(true);
  });

  it("detects revert oscillation using target/replacement keys in OpenAI Responses format", () => {
    const body = {
      input: [
        { role: "user", content: "Fix parser state" },
        {
          type: "function_call",
          call_id: "fc1",
          name: "edit_file",
          arguments: JSON.stringify({ path: "src/parser.ts", target: "mode = 'strict'", replacement: "mode = 'loose'" }),
        },
        { type: "function_call_output", call_id: "fc1", output: "ok" },
        {
          type: "function_call",
          call_id: "fc2",
          name: "edit_file",
          arguments: JSON.stringify({ path: "src/parser.ts", target: "mode = 'loose'", replacement: "mode = 'strict'" }),
        },
        { type: "function_call_output", call_id: "fc2", output: "ok" },
      ],
    };

    const report = inspectStuckLoop(body, "openai_responses");
    expect(report.level).toBe("hard");
    expect(report.revertOscillation).toBe(true);
  });
});

describe("inspectStuckLoop: Soft triggers", () => {
  it("flags level=soft on error ping-pong (A -> B -> A)", () => {
    const body = {
      messages: [
        { role: "user", content: "Fix schema validation" },
        ...anthropicToolTurn(
          "t1",
          "bash",
          { command: "bunx vitest run" },
          "AssertionError: expected status 200 to equal 201 in schema.ts:10:2",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "t2",
          "edit_file",
          { path: "src/schema.ts", old_str: "status = 200", new_str: "status = 201" },
          "ok",
        ),
        ...anthropicToolTurn(
          "t3",
          "bash",
          { command: "bunx vitest run" },
          "TypeError: Cannot read properties of null (reading 'id') at handler.ts:22:4",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "t4",
          "edit_file",
          { path: "src/handler.ts", old_str: "res.id", new_str: "res?.id" },
          "ok",
        ),
        ...anthropicToolTurn(
          "t5",
          "bash",
          { command: "bunx vitest run" },
          "AssertionError: expected status 200 to equal 201 in schema.ts:18:2",
          { isError: true },
        ),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("soft");
    expect(report.identicalErrorCount).toBe(2);
    expect(report.reasons.some((r) => r.includes("error_ping_pong"))).toBe(true);
  });

  it("flags level=soft when hotspotEdits >= 2 (same file edited >= 3 times without passing verification)", () => {
    const body = {
      messages: [
        { role: "user", content: "Refactor router" },
        ...anthropicToolTurn("t1", "edit_file", { path: "src/router.ts", old_str: "v1", new_str: "v2" }, "ok"),
        ...anthropicToolTurn("t2", "edit_file", { path: "src/router.ts", old_str: "v2", new_str: "v3" }, "ok"),
        ...anthropicToolTurn("t3", "edit_file", { path: "src/router.ts", old_str: "v3", new_str: "v4" }, "ok"),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("soft");
    expect(report.hotspotEdits).toBe(2);
    expect(report.reasons.some((r) => r.includes("hotspot_edits"))).toBe(true);
  });

  it("resets hotspotEdits when a passing verification command runs between edits", () => {
    const body = {
      messages: [
        { role: "user", content: "Iteratively refine router" },
        ...anthropicToolTurn("t1", "edit_file", { path: "src/router.ts", old_str: "v1", new_str: "v2" }, "ok"),
        ...anthropicToolTurn("t2", "edit_file", { path: "src/router.ts", old_str: "v2", new_str: "v3" }, "ok"),
        ...anthropicToolTurn("t3", "bash", { command: "bunx vitest run" }, "Tests  5 passed (5)"),
        ...anthropicToolTurn("t4", "edit_file", { path: "src/router.ts", old_str: "v3", new_str: "v4" }, "ok"),
        ...anthropicToolTurn("t5", "edit_file", { path: "src/router.ts", old_str: "v4", new_str: "v5" }, "ok"),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("none");
    expect(report.hotspotEdits).toBe(1);
  });

  it("flags level=soft on apologyStreak >= 2 paired with continuing errors", () => {
    const body = {
      messages: [
        { role: "user", content: "Fix the build" },
        ...openaiChatToolTurn(
          "c1",
          "bash",
          JSON.stringify({ command: "bun run build" }),
          "Error: Cannot resolve import './missing-a'",
        ),
        ...openaiChatToolTurn(
          "c2",
          "edit_file",
          JSON.stringify({ path: "src/a.ts", old_str: "./missing-a", new_str: "./missing-b" }),
          "ok",
          "I apologize, that path was wrong. Let me fix the import.",
        ),
        ...openaiChatToolTurn(
          "c3",
          "bash",
          JSON.stringify({ command: "bun run build" }),
          "Error: Cannot resolve import './missing-b'",
        ),
        ...openaiChatToolTurn(
          "c4",
          "edit_file",
          JSON.stringify({ path: "src/b.ts", old_str: "./missing-b", new_str: "./missing-c" }),
          "ok",
          "My mistake, let me try again with the relative path.",
        ),
        ...openaiChatToolTurn(
          "c5",
          "bash",
          JSON.stringify({ command: "bun run build" }),
          "Error: Cannot resolve import './missing-c'",
        ),
      ],
    };

    const report = inspectStuckLoop(body, "openai_chat");
    expect(report.level).toBe("soft");
    expect(report.apologyStreak).toBeGreaterThanOrEqual(2);
    expect(report.reasons.some((r) => r.includes("apology_streak"))).toBe(true);
  });

  it("does NOT flag apologyStreak when assistant apologizes without errors", () => {
    const body = {
      messages: [
        { role: "user", content: "Inspect the repo" },
        ...anthropicToolTurn("t1", "read_file", { path: "src/a.ts" }, "export const a = 1;", {
          text: "I apologize for the delay, reading a.ts first.",
        }),
        ...anthropicToolTurn("t2", "read_file", { path: "src/b.ts" }, "export const b = 2;", {
          text: "Sorry about that, let me also check b.ts.",
        }),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("none");
    expect(report.apologyStreak).toBe(0);
  });

  it("flags level=soft on tddEvasion when agent modifies a test file to skip/weaken after a test failure", () => {
    const body = {
      messages: [
        { role: "user", content: "Make all auth tests pass" },
        ...anthropicToolTurn(
          "t1",
          "bash",
          { command: "bunx vitest run tests/auth.test.ts" },
          "FAIL tests/auth.test.ts\nAssertionError: expected 401 to be 200\n1 failed",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "t2",
          "edit_file",
          {
            path: "tests/auth.test.ts",
            old_str: 'it("allows valid jwt", () => {',
            new_str: 'it.skip("allows valid jwt", () => {',
          },
          "Updated tests/auth.test.ts",
        ),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("soft");
    expect(report.tddEvasion).toBe(true);
    expect(report.reasons.some((r) => r.includes("tdd_evasion"))).toBe(true);
  });

  it("flags tddEvasion for __tests__/ path modified after test failure without fixing source", () => {
    const body = {
      messages: [
        { role: "user", content: "Fix payment validation" },
        ...openaiChatToolTurn(
          "c1",
          "bash",
          JSON.stringify({ command: "npx jest __tests__/payment.ts" }),
          "FAIL __tests__/payment.ts\nExpected: 100\nReceived: 99\n1 failed",
        ),
        ...openaiChatToolTurn(
          "c2",
          "edit_file",
          JSON.stringify({
            path: "src/__tests__/payment.ts",
            old_str: "expect(total).toBe(100);",
            new_str: "expect(total).toBe(99);",
          }),
          "Updated src/__tests__/payment.ts",
        ),
      ],
    };

    const report = inspectStuckLoop(body, "openai_chat");
    expect(report.level).toBe("soft");
    expect(report.tddEvasion).toBe(true);
  });
});

describe("inspectStuckLoop: False-positive immunity", () => {
  it("returns level=none for normal TDD cycle (edit test -> fail -> edit source -> pass)", () => {
    const body = {
      messages: [
        { role: "user", content: "Implement add() using TDD" },
        ...anthropicToolTurn(
          "t1",
          "edit_file",
          { path: "tests/foo.test.ts", old_str: "// todo", new_str: "expect(add(1, 2)).toBe(3);" },
          "Updated tests/foo.test.ts",
        ),
        ...anthropicToolTurn(
          "t2",
          "bash",
          { command: "bunx vitest run tests/foo.test.ts" },
          "FAIL tests/foo.test.ts\nReferenceError: add is not defined\n1 failed",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "t3",
          "edit_file",
          { path: "src/foo.ts", old_str: "export {};", new_str: "export const add = (a: number, b: number) => a + b;" },
          "Updated src/foo.ts",
        ),
        ...anthropicToolTurn(
          "t4",
          "bash",
          { command: "bunx vitest run tests/foo.test.ts" },
          "PASS tests/foo.test.ts\nTests  1 passed (1)",
          { isError: false },
        ),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("none");
    expect(report.tddEvasion).toBe(false);
    expect(report.reasons).toEqual([]);
  });

  it("returns level=none across multiple consecutive normal TDD red-green cycles in one prompt", () => {
    const messages: unknown[] = [{ role: "user", content: "Build three helpers with TDD" }];
    for (const name of ["add", "sub", "mul"]) {
      messages.push(
        ...anthropicToolTurn(`ed-test-${name}`, "edit_file", { path: `tests/${name}.test.ts`, old_str: "//", new_str: `expect(${name}(1,2)).toBeDefined()` }, "ok"),
        ...anthropicToolTurn(`run-fail-${name}`, "bash", { command: "bunx vitest run" }, `FAIL tests/${name}.test.ts\nReferenceError: ${name} is not defined\n1 failed`, { isError: true }),
        ...anthropicToolTurn(`ed-src-${name}`, "edit_file", { path: `src/${name}.ts`, old_str: "//", new_str: `export const ${name} = () => 1;` }, "ok"),
        ...anthropicToolTurn(`run-pass-${name}`, "bash", { command: "bunx vitest run" }, "Tests  1 passed (1)", { isError: false }),
      );
    }

    const report = inspectStuckLoop({ messages }, "anthropic_messages");
    expect(report.level).toBe("none");
    expect(report.tddEvasion).toBe(false);
    expect(report.hotspotEdits).toBe(0);
    expect(report.reasons).toEqual([]);
  });

  it("returns level=none and errorCountTrendingDown=true for strictly decreasing failure counts (7 -> 4 -> 1)", () => {
    const body = {
      messages: [
        { role: "user", content: "Fix all TypeScript compilation errors" },
        ...anthropicToolTurn(
          "t1",
          "bash",
          { command: "bunx tsc --noEmit" },
          "src/a.ts:12:5 - error TS2322: Type mismatch\nFound 7 errors in 3 files.",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "t2",
          "edit_file",
          { path: "src/a.ts", old_str: "a", new_str: "b" },
          "ok",
        ),
        ...anthropicToolTurn(
          "t3",
          "bash",
          { command: "bunx tsc --noEmit" },
          "src/a.ts:12:5 - error TS2322: Type mismatch\nFound 4 errors in 2 files.",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "t4",
          "edit_file",
          { path: "src/b.ts", old_str: "c", new_str: "d" },
          "ok",
        ),
        ...anthropicToolTurn(
          "t5",
          "bash",
          { command: "bunx tsc --noEmit" },
          "src/a.ts:12:5 - error TS2322: Type mismatch\nFound 1 error in 1 file.",
          { isError: true },
        ),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("none");
    expect(report.errorCountTrendingDown).toBe(true);
    expect(report.reasons).toEqual([]);
  });

  it("does not flag error repetition when aggregate test failures decrease (7 failed -> 4 failed -> 1 failed)", () => {
    const body = {
      messages: [
        { role: "user", content: "Fix failing unit tests" },
        ...anthropicToolTurn(
          "t1",
          "bash",
          { command: "bunx vitest run" },
          "Test Files  1 failed (1)\nTests  7 failed | 3 passed (10)\nAssertionError: mismatch in calc.ts:10:2",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "t2",
          "bash",
          { command: "bunx vitest run" },
          "Test Files  1 failed (1)\nTests  4 failed | 6 passed (10)\nAssertionError: mismatch in calc.ts:10:2",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "t3",
          "bash",
          { command: "bunx vitest run" },
          "Test Files  1 failed (1)\nTests  1 failed | 9 passed (10)\nAssertionError: mismatch in calc.ts:10:2",
          { isError: true },
        ),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("none");
    expect(report.errorCountTrendingDown).toBe(true);
    expect(report.reasons).toEqual([]);
  });

  it("returns level=none for sequential compiler peeling across distinct files/symbols without counts", () => {
    const body = {
      messages: [
        { role: "user", content: "Fix type errors across modules" },
        ...openaiChatToolTurn(
          "c1",
          "bash",
          JSON.stringify({ command: "bunx tsc --noEmit" }),
          "src/alpha.ts:10:4 - error TS2322: Type 'string' is not assignable to type 'number'.",
        ),
        ...openaiChatToolTurn(
          "c2",
          "edit_file",
          JSON.stringify({ path: "src/alpha.ts", old_str: "'1'", new_str: "1" }),
          "ok",
        ),
        ...openaiChatToolTurn(
          "c3",
          "bash",
          JSON.stringify({ command: "bunx tsc --noEmit" }),
          "src/beta.ts:24:9 - error TS2339: Property 'userId' does not exist on type 'Session'.",
        ),
        ...openaiChatToolTurn(
          "c4",
          "edit_file",
          JSON.stringify({ path: "src/beta.ts", old_str: "s.userId", new_str: "s.id" }),
          "ok",
        ),
        ...openaiChatToolTurn(
          "c5",
          "bash",
          JSON.stringify({ command: "bunx tsc --noEmit" }),
          "src/gamma.ts:5:1 - error TS2304: Cannot find name 'Logger'.",
        ),
      ],
    };

    const report = inspectStuckLoop(body, "openai_chat");
    expect(report.level).toBe("none");
    expect(report.reasons).toEqual([]);
  });

  it("returns level=none for read-only exploration of distinct files containing error-like text", () => {
    const body = {
      messages: [
        { role: "user", content: "Understand error handling in the codebase" },
        ...anthropicToolTurn("t1", "read_file", { path: "src/errors.ts" }, "export class AppError extends TypeError {}\n// 7 failed tests"),
        ...anthropicToolTurn("t2", "read_file", { path: "src/parser.ts" }, "throw new SyntaxError('Found 4 errors');"),
        ...anthropicToolTurn("t3", "read_file", { path: "src/runner.ts" }, "if (code !== 0) console.error('Exit code 1');"),
        ...anthropicToolTurn("t4", "grep", { pattern: "TypeError" }, "src/errors.ts:1:TypeError\nsrc/errors.ts:2:TypeError"),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("none");
    expect(report.identicalErrorCount).toBe(0);
    expect(report.identicalToolFailCount).toBe(0);
  });
});

describe("inspectStuckLoop: Windowing, protocol support, and telemetrySummary", () => {
  it("ignores stuck history before the last real human user prompt (while ignoring tool_result-only user messages)", () => {
    const body = {
      messages: [
        { role: "user", content: "Old task that got stuck" },
        ...anthropicToolTurn("old1", "bash", { command: "npm test" }, "TypeError: same boom", { isError: true }),
        ...anthropicToolTurn("old2", "bash", { command: "npm test" }, "TypeError: same boom", { isError: true }),
        ...anthropicToolTurn("old3", "bash", { command: "npm test" }, "TypeError: same boom", { isError: true }),
        { role: "user", content: "Never mind, let's work on a fresh task instead." },
        ...anthropicToolTurn("new1", "read_file", { path: "src/index.ts" }, "export const ok = true;"),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("none");
    expect(report.identicalErrorCount).toBe(0);
  });

  it("caps inspection to a rolling window of the last 20 assistant/tool turns", () => {
    const messages: unknown[] = [{ role: "user", content: "Long running task" }];
    // Turns 1..3: identical hard error (will fall outside last 20 turns)
    for (let i = 1; i <= 3; i++) {
      messages.push(
        ...anthropicToolTurn(`err${i}`, "bash", { command: "npm test" }, "TypeError: ancient failure", {
          isError: true,
        }),
      );
    }
    // Turns 4..24: 21 clean exploration turns
    for (let i = 1; i <= 21; i++) {
      messages.push(
        ...anthropicToolTurn(`ok${i}`, "read_file", { path: `src/file-${i}.ts` }, `export const x${i} = ${i};`),
      );
    }

    const report = inspectStuckLoop({ messages }, "anthropic_messages");
    expect(report.level).toBe("none");
    expect(report.identicalErrorCount).toBe(0);
  });

  it("supports OpenAI Responses API (body.input with function_call/function_call_output)", () => {
    const body = {
      input: [
        { role: "user", content: [{ type: "input_text", text: "Fix the failing worker" }] },
        { type: "function_call", call_id: "fc1", name: "bash", arguments: '{"command":"bun test"}' },
        { type: "function_call_output", call_id: "fc1", output: "2026-11-25T01:00:00Z Error: ECONNREFUSED 0xabc1" },
        { type: "function_call", call_id: "fc2", name: "bash", arguments: '{"command":"bun test"}' },
        { type: "function_call_output", call_id: "fc2", output: "2026-11-25T01:00:05Z Error: ECONNREFUSED 0xdef2" },
        { type: "function_call", call_id: "fc3", name: "bash", arguments: '{"command":"bun test"}' },
        { type: "function_call_output", call_id: "fc3", output: "2026-11-25T01:00:10Z Error: ECONNREFUSED 0x9876" },
      ],
    };

    const report = inspectStuckLoop(body, "openai_responses");
    expect(report.level).toBe("hard");
    expect(report.identicalErrorCount).toBe(3);
  });

  it("handles empty or malformed request bodies cleanly without throwing", () => {
    expect(inspectStuckLoop({}, "anthropic_messages").level).toBe("none");
    expect(inspectStuckLoop({ messages: "invalid" }, "openai_chat").level).toBe("none");
    expect(inspectStuckLoop({ input: "just a string prompt" }, "openai_responses").level).toBe("none");
  });

  it("keeps telemetrySummary concise (< 300 chars) even for huge error payloads", () => {
    const giantError = `TypeError: ${"x".repeat(2000)} at /very/long/path/src/worker.ts:99:1`;
    const body = {
      messages: [
        { role: "user", content: "Fix worker" },
        ...anthropicToolTurn("t1", "bash", { command: "bun test" }, giantError, { isError: true }),
        ...anthropicToolTurn("t2", "bash", { command: "bun test" }, giantError, { isError: true }),
        ...anthropicToolTurn("t3", "bash", { command: "bun test" }, giantError, { isError: true }),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("hard");
    expect(report.telemetrySummary.length).toBeGreaterThan(0);
    expect(report.telemetrySummary.length).toBeLessThan(300);
  });

  it("clears pre-recovery error state after a passing verification so stale errors never re-trigger hard after lock expires", () => {
    const body = {
      messages: [
        { role: "user", content: "Fix the failing auth module" },
        // Pre-escalation stuck state: 3 identical hard errors, non-verify failing tool, revert oscillation, and TDD evasion
        ...anthropicToolTurn(
          "pre-err-1",
          "bash",
          { command: "bunx vitest run tests/auth.test.ts" },
          "FAIL tests/auth.test.ts\nTypeError: Cannot read token at src/auth.ts:12:4\n3 failed",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "pre-tdd",
          "edit_file",
          { path: "tests/auth.test.ts", old_str: 'it("works", () => {', new_str: 'it.skip("works", () => {' },
          "Updated tests/auth.test.ts",
        ),
        ...anthropicToolTurn(
          "pre-rev-1",
          "edit_file",
          { path: "src/auth.ts", old_str: "return a;", new_str: "return b;" },
          "Updated src/auth.ts",
        ),
        ...anthropicToolTurn(
          "pre-rev-2",
          "edit_file",
          { path: "src/auth.ts", old_str: "return b;", new_str: "return a;" },
          "Updated src/auth.ts",
        ),
        ...anthropicToolTurn(
          "pre-tool-1",
          "edit_file",
          { path: "src/missing.ts", old_str: "x", new_str: "y" },
          "Error: old_str not found",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "pre-tool-2",
          "edit_file",
          { path: "src/missing.ts", old_str: "x", new_str: "y" },
          "Error: old_str not found",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "pre-tool-3",
          "edit_file",
          { path: "src/missing.ts", old_str: "x", new_str: "y" },
          "Error: old_str not found",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "pre-err-2",
          "bash",
          { command: "bunx vitest run tests/auth.test.ts" },
          "FAIL tests/auth.test.ts\nTypeError: Cannot read token at src/auth.ts:12:4\n3 failed",
          { isError: true },
        ),
        ...anthropicToolTurn(
          "pre-err-3",
          "bash",
          { command: "bunx vitest run tests/auth.test.ts" },
          "FAIL tests/auth.test.ts\nTypeError: Cannot read token at src/auth.ts:12:4\n3 failed",
          { isError: true },
        ),
        // Recovery: escalated model fixes source and runs passing verification
        ...anthropicToolTurn(
          "fix-1",
          "edit_file",
          { path: "src/auth.ts", old_str: "return a;", new_str: "return validToken(a);" },
          "Updated src/auth.ts",
        ),
        ...anthropicToolTurn(
          "pass-1",
          "bash",
          { command: "bunx vitest run" },
          "PASS tests/auth.test.ts\nTests  3 passed (3)",
          { isError: false },
        ),
        // 5 clean post-recovery turns (past the 4-turn ratchet lock window)
        ...anthropicToolTurn("post-1", "read_file", { path: "src/index.ts" }, "export * from './auth';"),
        ...anthropicToolTurn("post-2", "read_file", { path: "src/routes.ts" }, "export const routes = [];"),
        ...anthropicToolTurn("post-3", "read_file", { path: "src/config.ts" }, "export const cfg = {};"),
        ...anthropicToolTurn("post-4", "read_file", { path: "src/types.ts" }, "export type User = {};"),
        ...anthropicToolTurn("post-5", "read_file", { path: "README.md" }, "# Docs"),
      ],
    };

    const report = inspectStuckLoop(body, "anthropic_messages");
    expect(report.level).toBe("none");
    expect(report.identicalErrorCount).toBe(0);
    expect(report.identicalToolFailCount).toBe(0);
    expect(report.revertOscillation).toBe(false);
    expect(report.tddEvasion).toBe(false);
    expect(report.reasons).toEqual([]);

    // Even a single new error after recovery starts fresh at count=1 and does not re-trigger hard
    const withOnePostError = {
      messages: [
        ...body.messages,
        ...anthropicToolTurn(
          "post-err-1",
          "bash",
          { command: "bunx vitest run tests/auth.test.ts" },
          "FAIL tests/auth.test.ts\nTypeError: Cannot read token at src/auth.ts:12:4\n1 failed",
          { isError: true },
        ),
      ],
    };
    const postErrReport = inspectStuckLoop(withOnePostError, "anthropic_messages");
    expect(postErrReport.level).toBe("none");
    expect(postErrReport.identicalErrorCount).toBe(1);
    expect(postErrReport.revertOscillation).toBe(false);
    expect(postErrReport.tddEvasion).toBe(false);
  });
});

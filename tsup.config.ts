import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "core/index": "src/core/index.ts",
    "server/http-bridge": "src/server/http-bridge.ts",
    "cli/index": "src/cli/index.ts",
  },
  format: ["esm"],
  dts: true,
  clean: true,
  sourcemap: false,
  target: "es2022",
  splitting: true,
  shims: false,
  external: [/^node:/],
});

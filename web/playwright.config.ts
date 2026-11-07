import { defineConfig } from "@playwright/test";

// Browser check of the M1 loop against running servers (make serve-model / serve-api /
// serve-web). Uses the locally installed Google Chrome, so no browser download is needed.
export default defineConfig({
  testDir: "e2e",
  outputDir: "../artifacts/e2e/results",
  timeout: 60_000,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: process.env.RECLOOP_WEB_URL ?? "http://localhost:13000",
    channel: "chrome",
    viewport: { width: 1440, height: 1000 },
    screenshot: "only-on-failure",
  },
});

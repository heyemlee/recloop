import type { NextConfig } from "next";

// The browser talks to the Go api through this app (/api/v1/* -> RECLOOP_API_URL/v1/*), so
// events sent with navigator.sendBeacon stay same-origin and no CORS setup is needed. Only the
// public /v1 routes are proxied: /internal/* (the shopping assistant's ads endpoint) and
// /metrics stay off the browser's reach. The conversation (/api/v1/chat, server-sent events)
// goes to the shopping assistant's own service (agent/server.py), which calls the api itself.
const apiUrl = process.env.RECLOOP_API_URL ?? "http://127.0.0.1:18080";
// An empty RECLOOP_AGENT_URL (the Kubernetes image, deploy/web.Dockerfile): no assistant, so no
// /api/v1/chat rewrite, and the chat page says so (NEXT_PUBLIC_CHAT_BACKEND, fixed at build).
const agentUrl = process.env.RECLOOP_AGENT_URL ?? "http://127.0.0.1:18083";

const nextConfig: NextConfig = {
  // The container image (deploy/web.Dockerfile) builds with NEXT_OUTPUT=standalone: a minimal
  // server.js plus the traced dependencies. `next dev` and `next start` are unaffected.
  output: process.env.NEXT_OUTPUT === "standalone" ? "standalone" : undefined,
  env: { NEXT_PUBLIC_CHAT_BACKEND: agentUrl ? "on" : "off" },
  cacheComponents: true,
  partialPrefetching: true,
  // The dev server only serves its scripts to localhost by default; tests may use 127.0.0.1.
  allowedDevOrigins: ["127.0.0.1"],
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    // In order: the first match wins.
    return [
      ...(agentUrl ? [{ source: "/api/v1/chat", destination: `${agentUrl}/v1/chat` }] : []),
      { source: "/api/v1/:path*", destination: `${apiUrl}/v1/:path*` },
    ];
  },
};

export default nextConfig;

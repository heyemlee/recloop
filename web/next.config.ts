import type { NextConfig } from "next";

// The browser talks to the Go api through this app (/api/v1/* -> RECLOOP_API_URL/v1/*), so
// events sent with navigator.sendBeacon stay same-origin and no CORS setup is needed. Only the
// public /v1 routes are proxied: /internal/* (the shopping assistant's ads endpoint) and
// /metrics stay off the browser's reach.
const apiUrl = process.env.RECLOOP_API_URL ?? "http://127.0.0.1:18080";

const nextConfig: NextConfig = {
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
    return [{ source: "/api/v1/:path*", destination: `${apiUrl}/v1/:path*` }];
  },
};

export default nextConfig;

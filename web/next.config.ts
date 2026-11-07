import type { NextConfig } from "next";

// The browser talks to the Go api through this app (/api/* -> RECLOOP_API_URL/*), so events
// sent with navigator.sendBeacon stay same-origin and no CORS setup is needed.
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
    return [{ source: "/api/:path*", destination: `${apiUrl}/:path*` }];
  },
};

export default nextConfig;

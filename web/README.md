# web

RecLoop's storefront (Next.js 16, App Router, shadcn/ui): the home feed, item pages with
"similar items", a user switcher (dataset users, or any id) and a debug panel that shows, per
request and per item, the recall sources, RRF score, model version, experiment groups and
filters.

The browser only talks to this app; `/api/*` is rewritten to the Go api (`RECLOOP_API_URL`,
default `http://127.0.0.1:18080`). Events follow design §7.1: impressions when a tile is at least
50% visible for 1 s, batched every 2 s and sent with `navigator.sendBeacon` when the page is
hidden; clicks, add-to-cart and purchases right away. Every event has a UUIDv7 `event_id` that
retries reuse.

From the repo root: `make web-install`, `make serve-web` (port 13000), `make test-web`,
`make e2e` (Playwright with the local Chrome, servers running).

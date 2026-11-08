# web

RecLoop's storefront (Next.js 16, App Router, shadcn/ui): the home feed, item pages with
"similar items", a user switcher (dataset users, or any id), a debug panel that shows, per
request and per item, the recall sources, RRF score, model version, experiment groups and
filters, the advertiser console (`/advertiser`) and the shopping assistant (`/chat`).

The browser only talks to this app; `/api/v1/*` is rewritten to the Go api (`RECLOOP_API_URL`,
default `http://127.0.0.1:18080`), except `/api/v1/chat`, which goes to the shopping assistant's
service (`RECLOOP_AGENT_URL`, default `http://127.0.0.1:18083`) for the `/chat` page.

Events follow design §7.1: impressions when a tile or chat card is at least 50% visible for 1 s,
batched every 2 s and sent with `navigator.sendBeacon` when the page is hidden; clicks,
add-to-cart and purchases right away. Every event has a UUIDv7 `event_id` that retries reuse;
sponsored items (feed slots, the chat's sponsored card) send their `ad_token` with both.

From the repo root: `make web-install`, `make serve-web` (port 13000), `make test-web`,
`make e2e` (Playwright with the local Chrome, servers running).

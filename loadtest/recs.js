// Open-loop load test of the recommendation path (docs/design.md §5.2).
//
// constant-arrival-rate starts RATE page views per second whatever the latency, so a slow
// server cannot slow the load down and hide its own tail (coordinated omission). Each
// iteration is one page view of a simulated session, with the simulator's world and click
// model (sim/world.py, artifacts/sim/world.json): request the page, report impressions of the
// positions the user looks at, maybe a click, add-to-cart or purchase, then move on (next
// page, the clicked item's page, or a refresh). Sessions live in the VU that runs them.
//
//   k6 run -e BASE_URL=http://127.0.0.1:18180 -e RATE=200 -e DURATION=60s \
//     -e WORLD=artifacts/sim/world.json -e SUMMARY=out.json loadtest/recs.js
//
// The summary keeps exact percentiles of every request (k6 trends keep all samples).

import http from "k6/http";
import { SharedArray } from "k6/data";
import exec from "k6/execution";
import { Counter, Rate, Trend } from "k6/metrics";

const BASE = __ENV.BASE_URL || "http://127.0.0.1:18180";
const RATE = Number(__ENV.RATE || 100);
const DURATION = __ENV.DURATION || "60s";
const WORLD = __ENV.WORLD || "../artifacts/sim/world.json";
const SEED = Number(__ENV.SEED || 20261106);
const EVENTS = (__ENV.EVENTS || "on") !== "off";

const users = new SharedArray("users", () => JSON.parse(open(WORLD)).users);
const params = new SharedArray("params", () => {
  const w = JSON.parse(open(WORLD));
  return [{ click: w.click_model, sessions: w.sessions, digest: w.digest }];
})[0];
const C = params.click;
const S = params.sessions;

export const options = {
  scenarios: {
    pageviews: {
      executor: "constant-arrival-rate",
      rate: RATE,
      timeUnit: "1s",
      duration: DURATION,
      // A VU waits for its page view (request + events); 100 ms each would need RATE / 10.
      // Past maxVUs k6 drops iterations (reported): the offered rate was not met.
      preAllocatedVUs: Math.max(20, Math.ceil(RATE * 0.15)),
      maxVUs: Math.max(100, RATE),
      gracefulStop: "5s",
    },
  },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "p(99.9)", "max", "count"],
  thresholds: {
    // Declared so the summary carries these sub-metrics (no pass / fail criteria here).
    "recs_ms{kind:home}": [],
    "recs_ms{kind:item}": [],
    "recs_ms{kind:next_page}": [],
    "cache_hit{kind:home}": [],
    "cache_hit{kind:item}": [],
  },
  insecureSkipTLSVerify: false,
  discardResponseBodies: false,
};

const recsMs = new Trend("recs_ms", true);
const eventsMs = new Trend("events_ms", true);
const recsErrors = new Rate("recs_errors");
const eventErrors = new Rate("events_errors");
const degraded = new Rate("recs_degraded");
const cacheHit = new Rate("cache_hit");
const clicks = new Counter("sim_clicks");
const pageViews = new Counter("sim_page_views");

// Per-VU deterministic random stream (mulberry32).
let state = 0;
function rand() {
  state = (state + 0x6d2b79f5) | 0;
  let t = state;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

const HEX = "0123456789abcdef";
function uuid7() {
  const ms = Date.now().toString(16).padStart(12, "0");
  let r = "";
  for (let i = 0; i < 20; i++) r += HEX[Math.floor(Math.random() * 16)];
  const variant = HEX[8 + Math.floor(Math.random() * 4)];
  return `${ms.slice(0, 8)}-${ms.slice(8, 12)}-7${r.slice(0, 3)}-${variant}${r.slice(3, 6)}-${r.slice(6, 18)}`;
}

let sess = null;

function newSession() {
  const row = users[Math.floor(rand() * users.length)];
  const rel = new Map();
  for (const [item, r] of row[2]) rel.set(item, r);
  for (const t of row[1]) rel.set(t, C.rel_target);
  return { user: row[0], targets: new Set(row[1]), rel, pages: 0, surface: "home", item: null, cursor: null };
}

function pClick(s, item, position) {
  const r = s.rel.has(item) ? s.rel.get(item) : C.rel_floor;
  return Math.pow(position, -C.position_exponent) * r;
}

export default function () {
  if (state === 0) state = (SEED * 1000 + exec.vu.idInTest) | 0;
  if (sess === null) sess = newSession();
  const s = sess;
  let url;
  let kind;
  if (s.cursor) {
    url = `${BASE}/v1/recommendations?user_id=${s.user}&limit=20&cursor=${s.cursor}`;
    kind = "next_page";
  } else if (s.surface === "item") {
    url = `${BASE}/v1/recommendations?user_id=${s.user}&surface=item&item_id=${s.item}&limit=20`;
    kind = "item";
  } else {
    url = `${BASE}/v1/recommendations?user_id=${s.user}&limit=20`;
    kind = "home";
  }
  const res = http.get(url, { tags: { name: "recs", kind }, timeout: "10s" });
  pageViews.add(1, { kind });
  recsMs.add(res.timings.duration, { kind });
  const ok = res.status === 200;
  recsErrors.add(!ok, { kind });
  s.pages++;
  if (!ok) {
    endOrReset(s, false, false, null);
    return;
  }
  const body = res.json();
  degraded.add(body.degraded.length > 0, { kind });
  if (kind !== "next_page") {
    cacheHit.add(res.headers["X-Candidate-Cache"] === "hit", { kind });
  }

  const visible = S.visible_min + Math.floor(rand() * (S.visible_max - S.visible_min + 1));
  const batch = [];
  let clicked = null;
  for (const rec of body.items.slice(0, visible)) {
    batch.push({ event_id: uuid7(), type: "impression", user_id: s.user, item_id: rec.item_id, impression_id: rec.impression_id });
    if (clicked === null && rand() < pClick(s, rec.item_id, rec.position)) clicked = rec;
  }
  let bought = false;
  if (clicked !== null) {
    clicks.add(1);
    batch.push({ event_id: uuid7(), type: "click", user_id: s.user, item_id: clicked.item_id, impression_id: clicked.impression_id });
    const pCart = s.targets.has(clicked.item_id) ? S.p_cart_target : S.p_cart;
    if (rand() < pCart) {
      batch.push({ event_id: uuid7(), type: "add_to_cart", user_id: s.user, item_id: clicked.item_id });
      if (rand() < S.p_buy) {
        batch.push({ event_id: uuid7(), type: "purchase", user_id: s.user, item_id: clicked.item_id });
        bought = true;
      }
    }
  }
  if (EVENTS && batch.length > 0) {
    const r = http.post(`${BASE}/v1/events`, JSON.stringify({ events: batch }), {
      headers: { "Content-Type": "application/json" },
      tags: { name: "events" },
      timeout: "10s",
    });
    eventsMs.add(r.timings.duration);
    eventErrors.add(r.status !== 202);
  }
  endOrReset(s, true, bought, clicked, body.next_cursor);
}

function endOrReset(s, ok, bought, clicked, nextCursor) {
  if (s.pages >= S.max_pages || rand() < S.p_end || (bought && rand() < S.p_end_after_buy)) {
    sess = null;
    return;
  }
  s.cursor = null;
  if (ok && clicked !== null && rand() < S.p_item_page) {
    s.surface = "item";
    s.item = clicked.item_id;
  } else if (ok && clicked === null && nextCursor && rand() < S.p_next_page) {
    s.cursor = nextCursor;
  } else {
    s.surface = "home";
    s.item = null;
  }
}

export function handleSummary(data) {
  const out = {};
  out[__ENV.SUMMARY || "summary.json"] = JSON.stringify(data);
  return out;
}

// Client events (design §7.1). Every event gets a UUIDv7 event_id when it is created and keeps
// it across retries, so the api stores and applies it at most once (§7.3).
//   * impressions: queued, flushed every 2 s, and with navigator.sendBeacon when the page is
//     hidden;
//   * clicks, add-to-cart and purchases: sent right away, retried with the same event_id.
import { v7 as uuidv7 } from "uuid";

export type EventType = "impression" | "click" | "add_to_cart" | "purchase";

export type ClientEvent = {
  event_id: string;
  type: EventType;
  user_id: number;
  item_id: number;
  impression_id?: string;
  /** The slot's token on impressions and clicks of sponsored items (design §9.2, §9.3). */
  ad_token?: string;
  client_ts: string;
  props?: Record<string, unknown>;
};

const ENDPOINT = "/api/v1/events";
const FLUSH_MS = 2000;
const MAX_BATCH = 100;

export function newEvent(
  type: EventType,
  userId: number,
  itemId: number,
  extra: { impressionId?: string; adToken?: string; props?: Record<string, unknown> } = {},
): ClientEvent {
  return {
    event_id: uuidv7(),
    type,
    user_id: userId,
    item_id: itemId,
    impression_id: extra.impressionId,
    ad_token: extra.adToken,
    client_ts: new Date().toISOString(),
    props: extra.props,
  };
}

const queue: ClientEvent[] = [];
const sentImpressions = new Set<string>();
let started = false;

function start() {
  if (started || typeof window === "undefined") return;
  started = true;
  window.setInterval(() => void flush(false), FLUSH_MS);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") void flush(true);
  });
  window.addEventListener("pagehide", () => void flush(true));
}

async function flush(beacon: boolean) {
  while (queue.length > 0) {
    const batch = queue.splice(0, MAX_BATCH);
    const body = JSON.stringify({ events: batch });
    const blob = new Blob([body], { type: "application/json" });
    if (beacon && navigator.sendBeacon && navigator.sendBeacon(ENDPOINT, blob)) {
      continue;
    }
    try {
      const r = await fetch(ENDPOINT, {
        method: "POST",
        body,
        headers: { "Content-Type": "application/json" },
        keepalive: true,
      });
      if (r.status >= 500) throw new Error(String(r.status));
    } catch {
      // Put the batch back: it is retried on the next flush with the same event_ids.
      queue.unshift(...batch);
      return;
    }
  }
}

/**
 * Queue an impression; each impression_id is reported once per page load. A sponsored slot's
 * impression carries its ad_token: it counts towards the campaign's frequency cap.
 */
export function trackImpression(userId: number, itemId: number, impressionId: string, adToken?: string) {
  if (sentImpressions.has(impressionId)) return;
  sentImpressions.add(impressionId);
  start();
  queue.push(newEvent("impression", userId, itemId, { impressionId, adToken }));
}

/** Send one event now; retried with the same event_id. Resolves to the api's answer. */
export async function sendNow(ev: ClientEvent, timeoutMs = 3000): Promise<unknown> {
  start();
  const body = JSON.stringify({ events: [ev] });
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown = new Error("not sent");
  for (let attempt = 0; attempt < 3 && Date.now() < deadline; attempt++) {
    try {
      const r = await fetch(ENDPOINT, {
        method: "POST",
        body,
        headers: { "Content-Type": "application/json" },
        keepalive: true,
      });
      if (r.status === 202) return await r.json();
      if (r.status < 500) throw new Error(`${r.status} ${await r.text()}`);
      lastErr = new Error(String(r.status));
    } catch (e) {
      lastErr = e;
    }
    await new Promise((res) => setTimeout(res, 150 * 2 ** attempt));
  }
  throw lastErr;
}

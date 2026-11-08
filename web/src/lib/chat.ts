// The shopping assistant's wire types (agent/chat.py) and its event stream: POST /api/v1/chat
// answers with server-sent events (meta, status..., reply, done; or error). EventSource only
// does GET, so the stream is read with fetch.

export type ChatEvidence = { field: string; snippet: string };

export type ChatCompat = {
  status: "confirmed" | "unconfirmed" | "incompatible";
  source: string;
  /** The reply's line about the device, as the code found it. */
  line: string | null;
  evidence: ChatEvidence[];
  note: string;
};

export type ChatCard = {
  position: number;
  slot: "organic" | "sponsored";
  /** "Sponsored" on the sponsored card, set by the assistant's rendering code. */
  label?: string;
  /** Organic: "{reply_id}:{position}"; sponsored: the ads module's, with its ad_token. */
  impression_id: string;
  ad_token?: string;
  campaign_id?: number;
  request_id?: string;
  item_id: number;
  title: string;
  brand: string | null;
  price_cents: number | null;
  image_url: string | null;
  stock: number;
  reason: string;
  reason_source?: "model" | "code";
  search_rank?: number;
  personal?: { rank: number; source: string } | null;
  compat: ChatCompat | null;
};

export type ChatReply = {
  reply_id: string;
  user_id: number;
  status: "ok" | "no_results" | "parse_failed" | "blocked";
  text: string;
  intro: string[];
  items: ChatCard[];
  sponsored: ChatCard | null;
  footer: string[];
  summary: string | null;
  constraints: Record<string, unknown> | null;
  reference_item_id: number | null;
  parse: { source: string; attempts: number };
  personal: { request_id: string | null; seeds: number[]; hits: number; moved: boolean } | null;
  ads: { called: boolean; candidates: number; outcome?: string | null; request_id?: string | null; ms?: number } | null;
  degraded: string[];
};

export type ChatDone = {
  timings_ms: Record<string, number>;
  server_ms: number;
  llm: { calls: number; cost_usd: number; input_tokens: number; output_tokens: number };
};

export type ChatEvent =
  | { event: "meta"; data: { reply_id: string; parser: string } }
  | { event: "status"; data: { stage: string } }
  | { event: "reply"; data: ChatReply }
  | { event: "done"; data: ChatDone }
  | { event: "error"; data: { message: string } };

function parseBlock(block: string): ChatEvent | null {
  let event = "";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    // ": ping" keep-alives and unknown fields are ignored.
  }
  if (!event || data.length === 0) return null;
  return { event, data: JSON.parse(data.join("\n")) } as ChatEvent;
}

/** Sends one message and calls `on` for every event of the reply as it arrives. */
export async function streamChat(
  userId: number,
  message: string,
  on: (ev: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ user_id: userId, message }),
    signal,
  });
  if (!r.ok || !r.body) {
    throw new Error(`${r.status} ${await r.text()}`);
  }
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value.replace(/\r\n/g, "\n");
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const ev = parseBlock(buf.slice(0, i));
      buf = buf.slice(i + 2);
      if (ev) on(ev);
    }
  }
  const last = parseBlock(buf.trim());
  if (last) on(last);
}

/** What the status events mean, for the line under the input. */
export const STAGE_LABEL: Record<string, string> = {
  parse: "Reading your request…",
  search: "Searching the catalog…",
  check: "Checking prices, stock and compatibility…",
  personalize: "Looking at what you bought and viewed…",
  reasons: "Writing the reasons…",
  ads: "Checking for a sponsored match…",
};

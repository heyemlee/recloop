"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { ChatItemCard } from "@/components/chat-card";
import { ArrowUpIcon } from "@/components/icons";
import { STAGE_LABEL, streamChat, type ChatCard, type ChatDone, type ChatReply } from "@/lib/chat";
import { newEvent, sendNow } from "@/lib/events";
import { useUser } from "@/lib/user";

// Shopping assistant (design §10, §22.1): each message is one turn of the agent. Items come as
// cards; prices, titles and compatibility come from the listings, a sponsored item carries its
// label. Every turn remembers the user it was asked as, so its events stay that user's.
type Turn = {
  id: number;
  userId: number;
  message: string;
  parser?: string;
  stage?: string;
  reply?: ChatReply;
  done?: ChatDone;
  error?: string;
};

const EXAMPLES = [
  "Ink cartridges for my Canon PIXMA MG3620 under $25",
  "Genuine toner cartridges for my Brother HL-2240",
  "Pilot gel pens, please.",
  "I loved the gel pens I bought last time. More from the same brand, please.",
  "post-its for my desk",
];

function Debug({ turn }: { turn: Turn }) {
  const r = turn.reply;
  const d = turn.done;
  if (!r) return null;
  const stages = d
    ? Object.entries(d.timings_ms)
        .filter(([k]) => k !== "total")
        .map(([k, v]) => `${k} ${v.toFixed(1)}`)
        .join(" · ")
    : "";
  return (
    <div
      data-testid="chat-debug"
      className="space-y-0.5 rounded-[20px] bg-card px-4 py-3 font-mono text-[11px] leading-relaxed tracking-normal text-muted-foreground"
    >
      <div>
        reply {r.reply_id} · parser {turn.parser} ({r.parse.source}, {r.parse.attempts} attempt
        {r.parse.attempts === 1 ? "" : "s"}) · status {r.status}
      </div>
      {r.constraints && <div className="break-all">constraints {JSON.stringify(r.constraints)}</div>}
      {r.personal && (
        <div>
          personalised: seeds {r.personal.seeds.join(",")} · {r.personal.hits} also recommended ·{" "}
          {r.personal.moved ? "order changed" : "order kept"}
        </div>
      )}
      {r.ads && (
        <div>
          ads: {r.ads.called ? `${r.ads.candidates} candidates · ${r.ads.outcome ?? "failed"}` : "not asked (no candidate)"}
        </div>
      )}
      <div>degraded {r.degraded.length ? r.degraded.join(", ") : "none"}</div>
      {d && (
        <div>
          {d.timings_ms.total?.toFixed(1)} ms ({stages}) · LLM calls {d.llm.calls} · ${d.llm.cost_usd.toFixed(4)}
        </div>
      )}
    </div>
  );
}

export default function ChatPage() {
  const { userId, debug } = useUser();
  const router = useRouter();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const nextId = useRef(1);
  const bottom = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);
  const asked = useRef(false);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);
  useEffect(() => () => abort.current?.abort(), []);

  function update(id: number, patch: Partial<Turn>) {
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch } : t)));
  }

  async function send(text: string) {
    const message = text.trim();
    if (!message || userId === null || busy) return;
    const id = nextId.current++;
    setTurns((ts) => [...ts, { id, userId, message }]);
    setDraft("");
    setBusy(true);
    const ctrl = new AbortController();
    abort.current = ctrl;
    try {
      await streamChat(
        userId,
        message,
        (ev) => {
          switch (ev.event) {
            case "meta":
              update(id, { parser: ev.data.parser });
              break;
            case "status":
              update(id, { stage: ev.data.stage });
              break;
            case "reply":
              update(id, { reply: ev.data, stage: undefined });
              break;
            case "done":
              update(id, { done: ev.data });
              break;
            case "error":
              update(id, { error: ev.data.message, stage: undefined });
              break;
          }
        },
        ctrl.signal,
      );
    } catch (e) {
      if (!ctrl.signal.aborted) update(id, { error: String(e), stage: undefined });
    } finally {
      setBusy(false);
    }
  }

  // The store pages' search pill lands here with ?q=...: asked once, as the first message, and
  // taken out of the address so a reload does not ask again. The timer keeps the state updates
  // of send() out of the effect itself.
  useEffect(() => {
    if (userId === null || asked.current) return;
    const q = new URLSearchParams(window.location.search).get("q")?.trim();
    if (!q) return;
    const t = setTimeout(() => {
      asked.current = true;
      window.history.replaceState(null, "", window.location.pathname);
      void send(q);
    }, 0);
    return () => clearTimeout(t);
    // send is recreated on every render; this runs once per page, when the user is known.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  async function open(turn: Turn, card: ChatCard) {
    // A sponsored card's click carries its ad_token: billing charges it (design §9.3).
    const ev = newEvent("click", turn.userId, card.item_id, {
      impressionId: card.impression_id,
      adToken: card.ad_token,
      props: { surface: "chat", position: card.position, slot: card.slot, reply_id: turn.reply?.reply_id },
    });
    await sendNow(ev, 1500).catch(() => undefined);
    router.push(`/items/${card.item_id}`);
  }

  // Built without an assistant (the Kubernetes image, next.config.ts): say so instead of
  // sending messages that cannot be answered.
  if (process.env.NEXT_PUBLIC_CHAT_BACKEND === "off") {
    return (
      <main className="mx-auto mt-9 flex w-full max-w-[720px] flex-1 flex-col gap-2">
        <h1 className="text-xl leading-tight font-semibold tracking-[-0.05em]">Shopping assistant</h1>
        <p className="text-sm text-muted-foreground" data-testid="chat-unavailable">
          The shopping assistant is not part of this deployment. It runs with the local compose setup;
          this store runs on the Kubernetes deployment, which has the recommendation services only.
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto mt-9 flex w-full max-w-[720px] flex-1 flex-col gap-10">
      <div className="flex flex-col gap-1.5">
        <h1 className="text-xl leading-tight font-semibold tracking-[-0.05em]">Shopping assistant</h1>
        <p className="text-sm text-muted-foreground">
          Prices and compatibility come from the listings, never from the model.
        </p>
      </div>

      <div className="flex flex-1 flex-col gap-10" data-testid="chat-turns">
        {turns.map((t) => (
          <section key={t.id} className="flex flex-col gap-4" data-testid="chat-turn" data-reply-id={t.reply?.reply_id}>
            <div className="max-w-[80%] self-end rounded-[20px] bg-primary px-[18px] py-[11px] leading-[1.38] tracking-[-0.031em] text-primary-foreground">
              {t.message}
            </div>
            {t.stage && <p className="text-sm text-muted-foreground">{STAGE_LABEL[t.stage] ?? "Working…"}</p>}
            {!t.reply && !t.error && !t.stage && <p className="text-sm text-muted-foreground">Sending…</p>}
            {t.error && <p className="text-sm text-destructive">The assistant could not answer: {t.error}</p>}
            {t.reply && (
              <>
                {t.reply.intro.map((line, i) => (
                  <p key={i} className="leading-[1.38] tracking-[-0.031em]">
                    {line}
                  </p>
                ))}
                {(t.reply.items.length > 0 || t.reply.sponsored) && (
                  <div className="grid gap-3 sm:grid-cols-2">
                    {t.reply.items.map((c) => (
                      <ChatItemCard
                        key={c.impression_id}
                        card={c}
                        userId={t.userId}
                        debug={debug}
                        onOpen={(card) => open(t, card)}
                      />
                    ))}
                    {t.reply.sponsored && (
                      <ChatItemCard
                        card={t.reply.sponsored}
                        userId={t.userId}
                        debug={debug}
                        onOpen={(card) => open(t, card)}
                      />
                    )}
                  </div>
                )}
                {t.reply.footer.map((line, i) => (
                  <p key={i} className="text-xs leading-[1.4] text-muted-foreground">
                    {line}
                  </p>
                ))}
                {debug && <Debug turn={t} />}
              </>
            )}
          </section>
        ))}
        <div ref={bottom} />
      </div>

      <div className="sticky bottom-0 -mx-2 flex flex-col gap-3 bg-linear-to-t from-background from-70% to-transparent px-2 pt-6 pb-5">
        {turns.length === 0 && (
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => void send(ex)}
                disabled={busy || userId === null}
                className="rounded-full border bg-card px-4 py-2 text-sm shadow-pill transition-colors hover:bg-muted disabled:opacity-50"
              >
                {ex}
              </button>
            ))}
          </div>
        )}
        <form
          className="flex h-16 items-center gap-3 rounded-full border border-black/10 bg-card py-2 pr-2 pl-7 shadow-float"
          onSubmit={(e) => {
            e.preventDefault();
            void send(draft);
          }}
        >
          <input
            data-testid="chat-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="What are you shopping for today?"
            maxLength={2000}
            aria-label="Message"
            className="min-w-0 flex-1 bg-transparent text-base tracking-[-0.031em] outline-none placeholder:text-muted-foreground"
          />
          <button
            type="submit"
            aria-label="Send"
            disabled={busy || userId === null || !draft.trim()}
            className="flex size-12 shrink-0 items-center justify-center rounded-full bg-brand text-white shadow-brand transition-colors hover:bg-brand/90 disabled:bg-[#cccccc] disabled:shadow-none"
          >
            <ArrowUpIcon className="size-5" />
          </button>
        </form>
      </div>
    </main>
  );
}

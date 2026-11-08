"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { ChatItemCard } from "@/components/chat-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
      className="space-y-0.5 rounded-lg bg-muted/60 px-3 py-2 font-mono text-[11px] leading-relaxed text-muted-foreground"
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
  const { userId, demoUsers, debug } = useUser();
  const router = useRouter();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const nextId = useRef(1);
  const bottom = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);
  const demo = demoUsers.find((u) => u.user_id === userId);

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

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 px-4 py-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Shopping assistant</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Ask for office supplies with a budget, a brand or your printer model.{" "}
          {userId === null ? "" : demo ? `Asking as ${demo.label}.` : `Asking as user ${userId}.`} Prices and
          compatibility come from the listings, never from the model.
        </p>
      </div>

      <div className="flex flex-1 flex-col gap-6" data-testid="chat-turns">
        {turns.map((t) => (
          <section key={t.id} className="space-y-3" data-testid="chat-turn" data-reply-id={t.reply?.reply_id}>
            <div className="flex justify-end">
              <div className="max-w-[80%] rounded-2xl rounded-br-md bg-primary px-3 py-2 text-sm text-primary-foreground">
                {t.message}
              </div>
            </div>
            <div className="space-y-2">
              {t.stage && <p className="text-sm text-muted-foreground">{STAGE_LABEL[t.stage] ?? "Working…"}</p>}
              {!t.reply && !t.error && !t.stage && <p className="text-sm text-muted-foreground">Sending…</p>}
              {t.error && <p className="text-sm text-destructive">The assistant could not answer: {t.error}</p>}
              {t.reply && (
                <>
                  {t.reply.intro.map((line, i) => (
                    <p key={i} className="text-sm">
                      {line}
                    </p>
                  ))}
                  {(t.reply.items.length > 0 || t.reply.sponsored) && (
                    <div className="grid gap-2 sm:grid-cols-2">
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
                    <p key={i} className="text-xs text-muted-foreground">
                      {line}
                    </p>
                  ))}
                  {debug && <Debug turn={t} />}
                </>
              )}
            </div>
          </section>
        ))}
        <div ref={bottom} />
      </div>

      <div className="sticky bottom-0 space-y-2 border-t bg-background/95 py-3 backdrop-blur">
        {turns.length === 0 && (
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => void send(ex)}
                disabled={busy || userId === null}
                className="rounded-full border px-3 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-50"
              >
                {ex}
              </button>
            ))}
          </div>
        )}
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void send(draft);
          }}
        >
          <Input
            data-testid="chat-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="e.g. toner for my Brother HL-2240 under $40"
            maxLength={2000}
            aria-label="Message"
          />
          <Button type="submit" disabled={busy || userId === null || !draft.trim()}>
            {busy ? "…" : "Send"}
          </Button>
        </form>
      </div>
    </main>
  );
}

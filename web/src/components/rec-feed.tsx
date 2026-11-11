"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getRecommendations } from "@/lib/api";
import { newEvent, sendNow } from "@/lib/events";
import type { RecItem, RecResponse } from "@/lib/types";
import { useUser } from "@/lib/user";

import { DebugPanel } from "./debug-panel";
import { ItemTile } from "./item-tile";

const GRID = "grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-x-3 gap-y-7 sm:grid-cols-[repeat(auto-fill,minmax(190px,1fr))]";

/** A recommendation list (home feed or "similar items"), with paging and the debug panel. */
export function RecFeed({
  surface,
  itemId,
  limit = 20,
  title,
  header,
}: {
  surface: "home" | "item";
  itemId?: number;
  limit?: number;
  title: string;
  /** Drawn above the list from its first page (the home page's hero). */
  header?: (feed: { items: RecItem[]; onOpen: (rec: RecItem) => void }) => React.ReactNode;
}) {
  const { userId, debug } = useUser();
  const router = useRouter();
  // Results are keyed by the request parameters; a change of user or item starts over.
  const key = `${userId}|${surface}|${itemId}|${limit}`;
  const [state, setState] = useState<{ key: string; pages: RecResponse[]; error: string | null }>({
    key: "",
    pages: [],
    error: null,
  });
  const [moreLoading, setMoreLoading] = useState(false);

  useEffect(() => {
    if (userId === null) return;
    let cancelled = false;
    getRecommendations({ userId, surface, itemId, limit })
      .then((r) => !cancelled && setState({ key, pages: [r], error: null }))
      .catch((e) => !cancelled && setState({ key, pages: [], error: String(e) }));
    return () => {
      cancelled = true;
    };
  }, [key, userId, surface, itemId, limit]);

  const current = state.key === key ? state : null;
  const pages = current?.pages ?? [];
  const error = current?.error ?? null;
  const loading = current === null || moreLoading;
  const first = pages[0];
  const last = pages[pages.length - 1];
  const items = pages.flatMap((p) => p.items);
  const titles = useMemo(
    () => new Map((first?.debug?.history ?? []).map((h) => [h.item_id, h.title])),
    [first],
  );
  const recall = first?.experiments?.recall;
  const adPrices = useMemo(() => new Map((first?.debug?.ads?.slots ?? []).map((s) => [s.pos, s.price])), [first]);
  const meta = {
    modelVersion: first?.model_version ?? "",
    group: recall ? `recall: ${recall.exp_id ? `${recall.exp_id} / ` : ""}${recall.variant}` : "",
    degraded: first?.degraded ?? [],
    adPrices,
  };

  async function loadMore() {
    if (!last?.next_cursor || userId === null) return;
    setMoreLoading(true);
    try {
      const r = await getRecommendations({ userId, surface, cursor: last.next_cursor, limit });
      setState((s) => (s.key === key ? { ...s, pages: [...s.pages, r] } : s));
    } catch (e) {
      setState((s) => (s.key === key ? { ...s, error: String(e) } : s));
    } finally {
      setMoreLoading(false);
    }
  }

  async function open(rec: RecItem) {
    if (userId !== null) {
      // A sponsored slot's click carries its ad_token: billing charges it (design §9.3).
      const ev = newEvent("click", userId, rec.item_id, {
        impressionId: rec.impression_id,
        adToken: rec.ad_token,
        props: { surface, position: rec.position, source: rec.source, slot: rec.slot },
      });
      // Wait (briefly) for the click to be stored, so the next page already reflects it.
      await sendNow(ev, 1500).catch(() => undefined);
    }
    router.push(`/items/${rec.item_id}`);
  }

  return (
    <>
      {header?.({ items: first?.items ?? [], onOpen: open })}
      <section className="flex flex-col gap-6" data-testid={`feed-${surface}`} data-request-id={first?.request_id}>
        <div className="flex items-baseline justify-between gap-4">
          <h2 className="text-xl leading-tight font-semibold tracking-[-0.05em]">{title}</h2>
          {debug && first && (
            <span className="text-xs text-muted-foreground">
              {items.length} items · {first.surface} surface
            </span>
          )}
        </div>
        {debug && first && <DebugPanel resp={first} />}
        {error && <p className="text-sm text-destructive">Could not load recommendations: {error}</p>}
        {!first && loading && (
          <div className={GRID}>
            {Array.from({ length: 10 }, (_, i) => (
              <Skeleton key={i} className="aspect-square rounded-[20px]" />
            ))}
          </div>
        )}
        {first && (
          <div className={GRID}>
            {items.map((rec) => (
              <ItemTile
                key={rec.impression_id}
                rec={rec}
                userId={userId}
                debug={debug}
                titles={titles}
                meta={meta}
                onOpen={open}
              />
            ))}
          </div>
        )}
        {last?.next_cursor && (
          <div className="flex justify-center pt-4">
            <Button variant="outline" className="h-12 px-7 text-base shadow-pill" onClick={loadMore} disabled={loading}>
              {loading ? "Loading…" : "Load more"}
            </Button>
          </div>
        )}
      </section>
    </>
  );
}

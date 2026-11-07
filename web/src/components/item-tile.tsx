"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";

import { Badge } from "@/components/ui/badge";
import { formatPrice } from "@/lib/api";
import { trackImpression } from "@/lib/events";
import type { RecItem } from "@/lib/types";
import { cn } from "@/lib/utils";

// One colour per recall source, shared by the tiles and the debug panel.
export const SOURCE_STYLE: Record<string, string> = {
  sasrec: "bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-200",
  itemcf: "bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-200",
  pop: "bg-zinc-200 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200",
  text: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200",
  backfill: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200",
};

/** Reports an impression once the tile is at least 50% visible for 1 second (design §7.1). */
function useImpression(
  ref: React.RefObject<HTMLElement | null>,
  userId: number | null,
  itemId: number,
  impressionId: string,
) {
  useEffect(() => {
    const el = ref.current;
    if (!el || userId === null) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.intersectionRatio >= 0.5) {
            timer ??= setTimeout(() => {
              trackImpression(userId, itemId, impressionId);
              obs.disconnect();
            }, 1000);
          } else if (timer !== undefined) {
            clearTimeout(timer);
            timer = undefined;
          }
        }
      },
      { threshold: [0, 0.5, 1] },
    );
    obs.observe(el);
    return () => {
      obs.disconnect();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [ref, userId, itemId, impressionId]);
}

/** Request-level facts repeated on every tile's debug box (design §22.1). */
export type TileMeta = { modelVersion: string; group: string; degraded: string[] };

export function ItemTile({
  rec,
  userId,
  debug,
  titles,
  meta,
  onOpen,
}: {
  rec: RecItem;
  userId: number | null;
  debug: boolean;
  titles: Map<number, string>;
  meta: TileMeta;
  onOpen: (rec: RecItem) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useImpression(ref, userId, rec.item_id, rec.impression_id);
  const it = rec.item;

  return (
    <div
      ref={ref}
      data-testid="rec-tile"
      data-item-id={rec.item_id}
      data-sources={rec.sources.map((h) => h.source).join(",")}
      data-via={rec.sources
        .filter((h) => h.source === "itemcf" && h.via !== undefined)
        .map((h) => h.via)
        .join(",")}
      className="flex flex-col"
    >
      <Link
        href={`/items/${rec.item_id}`}
        onClick={(e) => {
          e.preventDefault();
          onOpen(rec);
        }}
        className="group flex flex-1 flex-col overflow-hidden rounded-xl border bg-card transition-shadow hover:shadow-md"
      >
        <div className="relative flex aspect-square items-center justify-center bg-white p-3">
          {it?.image_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={it.image_url} alt="" loading="lazy" className="max-h-full max-w-full object-contain" />
          ) : (
            <span className="text-xs text-muted-foreground">no image</span>
          )}
          <span className="absolute left-2 top-2 rounded-md bg-background/90 px-1.5 text-xs tabular-nums text-muted-foreground">
            #{rec.position}
          </span>
        </div>
        <div className="flex flex-1 flex-col gap-1.5 p-3">
          <div className="line-clamp-2 text-sm font-medium leading-snug group-hover:underline">
            {it?.title ?? `Item ${rec.item_id}`}
          </div>
          <div className="text-xs text-muted-foreground">
            {it?.brand ?? "—"} · {formatPrice(it?.price_cents)}
          </div>
          <div className="mt-auto flex flex-wrap gap-1 pt-1">
            <Badge className={cn("rounded-md", SOURCE_STYLE[rec.source])}>{rec.source}</Badge>
            {rec.sources
              .filter((h) => h.source !== rec.source)
              .map((h) => (
                <Badge key={h.source} variant="outline" className="rounded-md text-muted-foreground">
                  +{h.source}
                </Badge>
              ))}
          </div>
        </div>
      </Link>
      {debug && (
        <div data-testid="tile-debug" className="mt-1 rounded-lg bg-muted/60 px-2 py-1.5 font-mono text-[11px] leading-relaxed text-muted-foreground">
          <div>
            item {rec.item_id} · RRF {rec.score.toFixed(4)}
          </div>
          {rec.sources.map((h) => (
            <div key={h.source} className="truncate">
              {h.source} #{h.rank}
              {h.source !== "backfill" && ` · ${h.score.toFixed(3)}`}
              {h.via !== undefined && h.via >= 0 && h.source === "itemcf" && (
                <span title={titles.get(h.via)}> · via {titles.get(h.via)?.slice(0, 24) ?? h.via}</span>
              )}
            </div>
          ))}
          <div className="truncate" title={meta.modelVersion}>
            model {meta.modelVersion}
          </div>
          <div className="truncate">
            {meta.group} · degraded {meta.degraded.length ? meta.degraded.join(",") : "none"}
          </div>
        </div>
      )}
    </div>
  );
}

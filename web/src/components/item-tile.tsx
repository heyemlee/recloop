"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";

import { formatPrice } from "@/lib/api";
import { trackImpression } from "@/lib/events";
import type { RecItem } from "@/lib/types";

/** Reports an impression once the tile is at least 50% visible for 1 second (design §7.1). */
export function useImpression(
  ref: React.RefObject<HTMLElement | null>,
  userId: number | null,
  itemId: number,
  impressionId: string,
  adToken?: string,
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
              trackImpression(userId, itemId, impressionId, adToken);
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
  }, [ref, userId, itemId, impressionId, adToken]);
}

/** Request-level facts repeated on every tile's debug box (design §22.1). */
export type TileMeta = { modelVersion: string; group: string; degraded: string[]; adPrices?: Map<number, number> };

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
  useImpression(ref, userId, rec.item_id, rec.impression_id, rec.ad_token);
  const it = rec.item;
  const sponsored = rec.slot === "sponsored";
  const price = meta.adPrices?.get(rec.position);

  return (
    <div
      ref={ref}
      data-testid="rec-tile"
      data-item-id={rec.item_id}
      data-slot={rec.slot}
      data-sources={rec.sources.map((h) => h.source).join(",")}
      data-via={rec.sources
        .filter((h) => h.source === "itemcf" && h.via !== undefined)
        .map((h) => h.via)
        .join(",")}
      className="flex min-w-0 flex-col"
    >
      <Link
        href={`/items/${rec.item_id}`}
        onClick={(e) => {
          e.preventDefault();
          onOpen(rec);
        }}
        className="group flex flex-col gap-2.5"
      >
        <div className="relative flex aspect-square items-center justify-center overflow-hidden rounded-[20px] bg-card transition-shadow group-hover:shadow-card">
          {it?.image_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={it.image_url}
              alt=""
              loading="lazy"
              className="absolute inset-5 size-[calc(100%-2.5rem)] object-contain transition-transform duration-300 group-hover:scale-[1.04]"
            />
          ) : (
            <span className="text-xs text-muted-foreground">no image</span>
          )}
          {sponsored && (
            <span
              data-testid="sponsored-label"
              className="absolute top-2.5 left-2.5 rounded-xl border bg-card px-2.5 py-1 text-[11px] font-medium tracking-normal"
            >
              {rec.label ?? "Sponsored"}
            </span>
          )}
          {debug && (
            <span className="absolute bottom-2.5 left-2.5 rounded-xl bg-muted px-2 py-1 text-[11px] font-medium tracking-normal text-muted-foreground tabular-nums">
              #{rec.position}
            </span>
          )}
        </div>
        <div className="flex flex-col gap-1 px-1">
          <div className="line-clamp-2 text-sm leading-[1.29] font-medium">{it?.title ?? `Item ${rec.item_id}`}</div>
          <div className="text-sm leading-[1.29] text-muted-foreground">
            {it?.brand ? `${it.brand} · ` : ""}
            <span className="text-foreground">{formatPrice(it?.price_cents)}</span>
          </div>
        </div>
      </Link>
      {debug && (
        <div data-testid="tile-debug" className="mt-2 px-1 font-mono text-[11px] leading-relaxed tracking-normal text-muted-foreground">
          <div>
            {sponsored
              ? `item ${rec.item_id} · campaign ${rec.campaign_id}${price !== undefined ? ` · ${price}¢/click` : ""}`
              : `item ${rec.item_id} · RRF ${rec.score.toFixed(4)}`}
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

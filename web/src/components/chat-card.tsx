"use client";

import Link from "next/link";
import { useRef } from "react";

import { formatPrice } from "@/lib/api";
import type { ChatCard } from "@/lib/chat";
import { cn } from "@/lib/utils";

import { useImpression } from "./item-tile";

/**
 * One item of an assistant reply. Its impression is reported like a feed tile's (≥ 50% visible
 * for 1 s); the sponsored card's impression and click carry its ad_token.
 */
export function ChatItemCard({
  card,
  userId,
  debug,
  onOpen,
}: {
  card: ChatCard;
  userId: number;
  debug: boolean;
  onOpen: (card: ChatCard) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useImpression(ref, userId, card.item_id, card.impression_id, card.ad_token);
  const sponsored = card.slot === "sponsored";
  const confirmed = card.compat?.status === "confirmed";

  return (
    <div
      ref={ref}
      data-testid="chat-card"
      data-item-id={card.item_id}
      data-slot={card.slot}
      className="flex flex-col"
    >
      <Link
        href={`/items/${card.item_id}`}
        onClick={(e) => {
          e.preventDefault();
          onOpen(card);
        }}
        className={cn(
          "group flex flex-1 gap-3 rounded-xl border bg-card p-3 transition-shadow hover:shadow-md",
          sponsored && "border-orange-300 dark:border-orange-800",
        )}
      >
        <div className="flex size-20 shrink-0 items-center justify-center rounded-lg bg-white p-1">
          {card.image_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={card.image_url} alt="" loading="lazy" className="max-h-full max-w-full object-contain" />
          ) : (
            <span className="text-[10px] text-muted-foreground">no image</span>
          )}
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex items-start justify-between gap-2">
            <span className="line-clamp-2 text-sm font-medium leading-snug group-hover:underline">
              {sponsored ? "" : `${card.position}. `}
              {card.title}
            </span>
            {sponsored && (
              <span
                data-testid="sponsored-label"
                className="shrink-0 rounded-md bg-orange-100 px-1.5 text-xs font-medium text-orange-900 dark:bg-orange-900/60 dark:text-orange-100"
              >
                {card.label ?? "Sponsored"}
              </span>
            )}
          </div>
          <div className="text-xs text-muted-foreground">
            {card.brand ?? "—"} ·{" "}
            <span data-testid="chat-price" className="font-medium text-foreground">
              {formatPrice(card.price_cents)}
            </span>
            {card.stock <= 3 && card.stock > 0 && ` · only ${card.stock} left`}
          </div>
          {card.compat?.line && (
            <div
              data-testid="compat-line"
              className={cn(
                "text-xs",
                confirmed ? "text-emerald-700 dark:text-emerald-300" : "text-amber-700 dark:text-amber-300",
              )}
            >
              {card.compat.line}
            </div>
          )}
          <div className="text-xs leading-relaxed">{card.reason}</div>
        </div>
      </Link>
      {debug && (
        <div className="mt-1 rounded-lg bg-muted/60 px-2 py-1.5 font-mono text-[11px] leading-relaxed text-muted-foreground">
          {sponsored ? (
            <div>
              item {card.item_id} · campaign {card.campaign_id} · ads request {card.request_id?.slice(0, 13)}…
            </div>
          ) : (
            <div>
              item {card.item_id} · search #{card.search_rank} · reason by {card.reason_source}
              {card.personal ? ` · for you #${card.personal.rank} (${card.personal.source})` : ""}
            </div>
          )}
          <div className="truncate" title={card.impression_id}>
            impression {card.impression_id}
          </div>
        </div>
      )}
    </div>
  );
}

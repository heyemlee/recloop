"use client";

import Link from "next/link";
import { useRef } from "react";

import { formatPrice } from "@/lib/api";
import type { ChatCard } from "@/lib/chat";
import { cn } from "@/lib/utils";

import { CheckCircleIcon, HelpCircleIcon } from "./icons";
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
      className="flex min-w-0 flex-col"
    >
      <Link
        href={`/items/${card.item_id}`}
        onClick={(e) => {
          e.preventDefault();
          onOpen(card);
        }}
        className="group flex flex-1 gap-3.5 rounded-[28px] bg-card p-2 shadow-card transition-shadow hover:shadow-float"
      >
        <div className="flex size-26 shrink-0 items-center justify-center rounded-[20px] bg-muted p-2.5">
          {card.image_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={card.image_url} alt="" loading="lazy" className="max-h-full max-w-full object-contain mix-blend-multiply" />
          ) : (
            <span className="text-[10px] text-muted-foreground">no image</span>
          )}
        </div>
        <div className="flex min-w-0 flex-1 flex-col justify-center gap-1.5 py-1.5 pr-2.5">
          <div className="flex items-start justify-between gap-2">
            <span className="line-clamp-2 text-sm leading-[1.29] font-medium">{card.title}</span>
            {sponsored && (
              <span
                data-testid="sponsored-label"
                className="shrink-0 rounded-xl border px-2.5 py-1 text-[11px] font-medium tracking-normal"
              >
                {card.label ?? "Sponsored"}
              </span>
            )}
          </div>
          <div className="text-sm text-muted-foreground">
            {card.brand ? `${card.brand} · ` : ""}
            <span data-testid="chat-price" className="text-foreground">
              {formatPrice(card.price_cents)}
            </span>
            {card.stock <= 3 && card.stock > 0 && ` · only ${card.stock} left`}
          </div>
          {card.compat?.line && (
            <div
              data-testid="compat-line"
              data-status={card.compat.status}
              className={cn("flex items-start gap-1.5 text-xs", confirmed ? "font-medium" : "text-muted-foreground")}
            >
              {confirmed ? <CheckCircleIcon className="mt-px size-3.5 shrink-0" /> : <HelpCircleIcon className="mt-px size-3.5 shrink-0" />}
              <span className="min-w-0 [overflow-wrap:anywhere]">{card.compat.line}</span>
            </div>
          )}
          <div className="text-xs leading-[1.4] text-muted-foreground">{card.reason}</div>
        </div>
      </Link>
      {debug && (
        <div className="mt-2 px-3 font-mono text-[11px] leading-relaxed tracking-normal text-muted-foreground">
          {sponsored ? (
            <div>
              item {card.item_id} · campaign {card.campaign_id} · ads request {card.request_id?.slice(0, 13)}…
            </div>
          ) : (
            <div>
              #{card.position} · item {card.item_id} · search #{card.search_rank} · reason by {card.reason_source}
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

"use client";

import Link from "next/link";

import { formatPrice } from "@/lib/api";
import type { RecItem } from "@/lib/types";
import { cn } from "@/lib/utils";

// One scene: a card on each side and three cut-outs between them, placed from the centre.
const CUTOUTS = [
  { left: "calc(50% - 350px)", top: 40, size: 250, bob: "rl-bob-1" },
  { left: "calc(50% - 120px)", top: 4, size: 240, bob: "rl-bob-2" },
  { left: "calc(50% + 100px)", top: 56, size: 250, bob: "rl-bob-3" },
];
const PER_SCENE = CUTOUTS.length + 2;

function open(e: React.MouseEvent, rec: RecItem, onOpen: (rec: RecItem) => void) {
  e.preventDefault();
  onOpen(rec);
}

function SideCard({ rec, side, onOpen }: { rec: RecItem; side: "left" | "right"; onOpen: (rec: RecItem) => void }) {
  return (
    <Link
      href={`/items/${rec.item_id}`}
      onClick={(e) => open(e, rec, onOpen)}
      style={{ left: side === "left" ? "calc(50% - 570px)" : "calc(50% + 400px)", rotate: side === "left" ? "-4deg" : "3deg" }}
      className={cn(
        "absolute top-8 flex w-[168px] flex-col gap-2 rounded-[28px] bg-card p-2 shadow-card max-xl:hidden",
        side === "left" ? "rl-card-left" : "rl-card-right",
      )}
    >
      <span className="relative block aspect-square rounded-[20px] bg-muted">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={rec.item?.image_url ?? ""} alt="" className="absolute inset-3 size-[calc(100%-1.5rem)] object-contain mix-blend-multiply" />
      </span>
      <span className="flex flex-col gap-0.5 px-1.5 pb-1.5">
        <span className="truncate text-[13px] font-medium">{rec.item?.title ?? `Item ${rec.item_id}`}</span>
        <span className="text-xs text-muted-foreground">{formatPrice(rec.item?.price_cents)}</span>
      </span>
    </Link>
  );
}

function Scene({ recs, late, onOpen }: { recs: RecItem[]; late: boolean; onOpen: (rec: RecItem) => void }) {
  const [left, ...rest] = recs;
  const right = rest.length > CUTOUTS.length ? rest[CUTOUTS.length] : undefined;
  return (
    <div className={cn("rl-scene", late && "rl-scene-late")}>
      {right && <SideCard rec={left} side="left" onOpen={onOpen} />}
      {(right ? rest : recs).slice(0, CUTOUTS.length).map((rec, i) => {
        const c = CUTOUTS[i];
        return (
          <Link
            key={rec.impression_id}
            href={`/items/${rec.item_id}`}
            onClick={(e) => open(e, rec, onOpen)}
            aria-label={rec.item?.title ?? `Item ${rec.item_id}`}
            className="absolute"
            style={{ left: c.left, top: c.top, width: c.size, height: c.size }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={rec.item?.image_url ?? ""} alt="" className={cn("rl-cutout size-full object-contain p-6", c.bob)} />
          </Link>
        );
      })}
      {right && <SideCard rec={right} side="right" onOpen={onOpen} />}
    </div>
  );
}

/**
 * The home page's hero: the first recommendations of the feed below, floating above the
 * wordmark. Organic items with a picture only (a sponsored slot is shown, and billed, in the
 * feed). Ten of them make two scenes that take turns; opening one is the same click as on its
 * tile. The hero reports no impressions of its own: the tiles do.
 */
export function Hero({ items, onOpen }: { items: RecItem[]; onOpen: (rec: RecItem) => void }) {
  const recs = items.filter((r) => r.slot !== "sponsored" && r.item?.image_url);
  const scenes = recs.length >= 2 * PER_SCENE ? 2 : recs.length >= CUTOUTS.length ? 1 : 0;
  return (
    <section
      data-testid="hero"
      aria-label="Picked for you"
      className={cn("relative h-[300px] overflow-hidden max-md:[zoom:0.55]", scenes === 2 && "rl-cycle")}
    >
      {scenes >= 1 && <Scene recs={recs.slice(0, PER_SCENE)} late={false} onOpen={onOpen} />}
      {scenes === 2 && <Scene recs={recs.slice(PER_SCENE, 2 * PER_SCENE)} late onOpen={onOpen} />}
    </section>
  );
}

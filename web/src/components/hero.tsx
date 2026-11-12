"use client";

import Link from "next/link";
import { useMemo } from "react";

import { formatPrice } from "@/lib/api";
import { cutoutOf, useCutoutsJudged } from "@/lib/cutout";
import type { RecItem } from "@/lib/types";
import { cn } from "@/lib/utils";

// One scene: three cut-outs placed from the centre, and a card on each side.
const CUTOUTS = [
  { left: "calc(50% - 350px)", top: 40, size: 250, bob: "rl-bob-1" },
  { left: "calc(50% - 120px)", top: 4, size: 240, bob: "rl-bob-2" },
  { left: "calc(50% + 100px)", top: 56, size: 250, bob: "rl-bob-3" },
];
const PER_SCENE = CUTOUTS.length;

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

function Scene({
  cutouts,
  left,
  right,
  late,
  onOpen,
}: {
  cutouts: RecItem[];
  left?: RecItem;
  right?: RecItem;
  late: boolean;
  onOpen: (rec: RecItem) => void;
}) {
  return (
    <div className={cn("rl-scene", late && "rl-scene-late")}>
      {left && <SideCard rec={left} side="left" onOpen={onOpen} />}
      {cutouts.map((rec, i) => {
        const c = CUTOUTS[i];
        return (
          <Link
            key={rec.impression_id}
            href={`/items/${rec.item_id}`}
            onClick={(e) => open(e, rec, onOpen)}
            aria-label={rec.item?.title ?? `Item ${rec.item_id}`}
            className={cn("rl-cutout absolute", c.bob)}
            style={{ left: c.left, top: c.top, width: c.size, height: c.size }}
          >
            {/* The slot keeps its place and its soft edge; a small product is enlarged inside it. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={rec.item?.image_url ?? ""}
              alt=""
              style={{ scale: cutoutOf(rec.item?.image_url).zoom }}
              className="size-full object-contain p-6"
            />
          </Link>
        );
      })}
      {right && <SideCard rec={right} side="right" onOpen={onOpen} />}
    </div>
  );
}

const byPosition = (a: RecItem, b: RecItem) => a.position - b.position;

/**
 * Who goes where. The cut-outs float on the page itself, so they are the photos that work
 * there: first the clean ones (on white, the product filling its frame) in feed order, then
 * the other photos on white, the closest to clean first. The side cards have a well of their
 * own, like a tile, and take the next items of the feed, the better photos first. Two scenes
 * when six photos are on white; with fewer than three, one scene of the feed's first items
 * rather than none.
 */
function cast(recs: RecItem[]) {
  const photo = (r: RecItem) => cutoutOf(r.item?.image_url);
  const tier = (r: RecItem) => photo(r).tier;
  const onWhite = recs
    .filter((r) => tier(r) < 3)
    .sort((a, b) => tier(a) - tier(b) || (tier(a) === 2 ? photo(b).score - photo(a).score : 0) || byPosition(a, b));
  const pool = onWhite.length >= PER_SCENE ? onWhite : recs;
  const scenes = onWhite.length >= 2 * PER_SCENE ? 2 : pool.length >= PER_SCENE ? 1 : 0;
  const cutouts = pool.slice(0, scenes * PER_SCENE);
  const cards = recs
    .filter((r) => !cutouts.includes(r))
    .sort((a, b) => tier(a) - tier(b) || byPosition(a, b))
    .slice(0, 2 * scenes);
  return Array.from({ length: scenes }, (_, s) => ({
    cutouts: cutouts.slice(s * PER_SCENE, (s + 1) * PER_SCENE).sort(byPosition),
    left: cards.length >= 2 * (s + 1) ? cards[2 * s] : undefined,
    right: cards.length >= 2 * (s + 1) ? cards[2 * s + 1] : undefined,
  }));
}

/**
 * The home page's hero: recommendations from the first page of the feed below, floating above
 * the wordmark. Organic items with a picture only (a sponsored slot is shown, and billed, in
 * the feed), chosen by how their photo looks as a cut-out (lib/cutout). Two scenes take turns;
 * opening an item is the same click as on its tile. The hero reports no impressions of its
 * own: the tiles do. It stays empty for the moment the photos are being judged.
 */
export function Hero({ items, onOpen }: { items: RecItem[]; onOpen: (rec: RecItem) => void }) {
  const recs = useMemo(() => items.filter((r) => r.slot !== "sponsored" && r.item?.image_url), [items]);
  const urls = useMemo(() => recs.map((r) => r.item?.image_url ?? ""), [recs]);
  const judged = useCutoutsJudged(urls);
  const scenes = useMemo(() => (judged ? cast(recs) : []), [recs, judged]);
  return (
    <section
      data-testid="hero"
      aria-label="Picked for you"
      className={cn("relative h-[300px] overflow-hidden max-md:[zoom:0.55]", scenes.length === 2 && "rl-cycle")}
    >
      {scenes.map((scene, s) => (
        <Scene key={s} {...scene} late={s === 1} onOpen={onOpen} />
      ))}
    </section>
  );
}

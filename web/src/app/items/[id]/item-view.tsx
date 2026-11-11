"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { RecFeed } from "@/components/rec-feed";
import { formatPrice, getItem } from "@/lib/api";
import { newEvent, sendNow } from "@/lib/events";
import type { Item } from "@/lib/types";
import { useUser } from "@/lib/user";

export function ItemView() {
  const { id } = useParams<{ id: string }>();
  const itemId = Number(id);
  const { userId, debug } = useUser();
  const [loaded, setLoaded] = useState<{ id: number; item: Item | null; error: string | null }>({
    id: -1,
    item: null,
    error: null,
  });
  const [status, setStatus] = useState<{ id: number; text: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    getItem(itemId)
      .then((it) => !cancelled && setLoaded({ id: itemId, item: it, error: null }))
      .catch((e) => !cancelled && setLoaded({ id: itemId, item: null, error: String(e) }));
    return () => {
      cancelled = true;
    };
  }, [itemId]);

  const item = loaded.id === itemId ? loaded.item : null;
  const error = loaded.id === itemId ? loaded.error : null;
  const message = status?.id === itemId ? status.text : null;
  const say = (text: string) => setStatus({ id: itemId, text });

  async function act(type: "add_to_cart" | "purchase") {
    if (userId === null || item === null) return;
    say("Sending…");
    try {
      await sendNow(newEvent(type, userId, item.item_id, { props: { surface: "item" } }));
      if (type === "purchase") {
        say("Bought. It will not be recommended again; stock updated.");
        setLoaded((l) => (l.item ? { ...l, item: { ...l.item, stock: Math.max(0, l.item.stock - 1) } } : l));
      } else {
        say("Added to cart. Your history now includes this item.");
      }
    } catch (e) {
      say(`Not sent: ${String(e)}`);
    }
  }

  return (
    <main className="mt-9 flex flex-col gap-16">
      {error && <p className="text-sm text-destructive">{error}</p>}
      {!item && !error && <Skeleton className="h-96 w-full rounded-[28px]" />}
      {item && (
        <article className="flex flex-wrap items-start gap-x-12 gap-y-8">
          <div className="max-w-[520px] min-w-0 flex-[1_1_340px] rounded-[28px] bg-card p-2 shadow-card">
            <div className="relative flex aspect-square items-center justify-center rounded-[20px]">
              {item.image_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={item.image_url} alt="" className="absolute inset-10 size-[calc(100%-5rem)] object-contain" />
              ) : (
                <span className="text-sm text-muted-foreground">no image</span>
              )}
            </div>
          </div>
          <div className="flex min-w-0 flex-[1.3_1_380px] flex-col gap-7 pt-4">
            <div className="flex flex-col gap-2.5">
              <p className="text-sm text-muted-foreground">
                {[item.brand, (item.categories ?? []).slice(1).at(-1)].filter(Boolean).join(" · ")}
              </p>
              <h1 data-testid="item-title" className="text-[28px] leading-[1.14] font-semibold tracking-[-0.05em] text-balance">
                {item.title}
              </h1>
              {debug && (
                <p className="font-mono text-[11px] tracking-normal text-muted-foreground">
                  item {item.item_id} · {(item.categories ?? []).slice(1).join(" › ")}
                </p>
              )}
            </div>
            <div className="flex items-baseline gap-3.5">
              <span className="text-[28px] leading-none font-semibold tracking-[-0.05em]">{formatPrice(item.price_cents)}</span>
              <span className={item.stock > 0 ? "text-sm text-muted-foreground" : "text-sm text-destructive"}>
                {item.stock > 0 ? `${item.stock} in stock` : "Sold out"}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="brand" size="lg" onClick={() => act("purchase")} disabled={item.stock <= 0}>
                Buy now
              </Button>
              <Button variant="outline" size="lg" className="shadow-pill" onClick={() => act("add_to_cart")} disabled={item.stock <= 0}>
                Add to cart
              </Button>
            </div>
            {message && (
              <p role="status" className="text-sm text-muted-foreground">
                {message}
              </p>
            )}
            {(item.features ?? []).length > 0 && (
              <ul className="flex flex-col gap-2.5 text-sm leading-[1.38]">
                {(item.features ?? []).slice(0, 6).map((f, i) => (
                  <li key={i} className="flex gap-2.5">
                    <span aria-hidden="true" className="mt-[7px] size-[5px] shrink-0 rounded-full bg-foreground" />
                    {f}
                  </li>
                ))}
              </ul>
            )}
            {item.description && (
              <p className="line-clamp-6 max-w-[560px] text-sm leading-[1.57] whitespace-pre-line text-muted-foreground">
                {item.description}
              </p>
            )}
          </div>
        </article>
      )}
      <RecFeed surface="item" itemId={itemId} limit={10} title="Similar items" />
    </main>
  );
}

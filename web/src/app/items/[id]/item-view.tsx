"use client";

import Link from "next/link";
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
  const { userId } = useUser();
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
    <main className="mx-auto w-full max-w-7xl space-y-10 px-4 py-6">
      <Link href="/" className="text-sm text-muted-foreground hover:underline">
        ← Back to your feed
      </Link>
      {error && <p className="text-sm text-destructive">{error}</p>}
      {!item && !error && <Skeleton className="h-80 w-full rounded-xl" />}
      {item && (
        <article className="grid gap-8 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
          <div className="flex aspect-square items-center justify-center rounded-xl border bg-white p-6">
            {item.image_url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={item.image_url} alt="" className="max-h-full max-w-full object-contain" />
            ) : (
              <span className="text-sm text-muted-foreground">no image</span>
            )}
          </div>
          <div className="space-y-4">
            <div>
              <p className="text-sm text-muted-foreground">{item.brand}</p>
              <h1 data-testid="item-title" className="text-xl font-semibold leading-snug">
                {item.title}
              </h1>
              <p className="mt-1 text-xs text-muted-foreground">
                item {item.item_id} · {(item.categories ?? []).slice(1).join(" › ")}
              </p>
            </div>
            <div className="flex items-baseline gap-4">
              <span className="text-2xl font-semibold">{formatPrice(item.price_cents)}</span>
              <span className={item.stock > 0 ? "text-sm text-muted-foreground" : "text-sm text-destructive"}>
                {item.stock > 0 ? `${item.stock} in stock` : "Sold out"}
              </span>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => act("add_to_cart")} disabled={item.stock <= 0}>
                Add to cart
              </Button>
              <Button onClick={() => act("purchase")} disabled={item.stock <= 0}>
                Buy now
              </Button>
            </div>
            {message && <p className="text-sm text-muted-foreground">{message}</p>}
            {(item.features ?? []).length > 0 && (
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {(item.features ?? []).slice(0, 6).map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            )}
            {item.description && (
              <p className="line-clamp-6 whitespace-pre-line text-sm text-muted-foreground">{item.description}</p>
            )}
          </div>
        </article>
      )}
      <RecFeed surface="item" itemId={itemId} limit={10} title="Similar items" />
    </main>
  );
}

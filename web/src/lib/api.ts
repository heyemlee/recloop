import type { DemoUser, Item, RecResponse } from "./types";

const BASE = "/api/v1";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path, { cache: "no-store" });
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`${r.status} ${body}`);
  }
  return (await r.json()) as T;
}

export type RecParams = {
  userId: number;
  surface: "home" | "item";
  itemId?: number;
  limit?: number;
  cursor?: string;
};

export function getRecommendations(p: RecParams): Promise<RecResponse> {
  const q = new URLSearchParams({ user_id: String(p.userId), debug: "1" });
  q.set("limit", String(p.limit ?? 20));
  if (p.cursor) {
    q.set("cursor", p.cursor);
  } else {
    q.set("surface", p.surface);
    if (p.itemId !== undefined) q.set("item_id", String(p.itemId));
  }
  return getJSON<RecResponse>(`/recommendations?${q}`);
}

export const getItem = (id: number) => getJSON<Item>(`/items/${id}`);
export const getDemoUsers = () => getJSON<DemoUser[]>(`/demo/users`);

export function formatPrice(cents: number | null | undefined): string {
  if (cents === null || cents === undefined) return "Price unknown";
  return `$${(cents / 100).toFixed(2)}`;
}

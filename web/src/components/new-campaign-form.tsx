"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type Campaign, RESERVE_CENTS, createCampaign, dollarsToCents } from "@/lib/ads";

/** Create a campaign for one advertiser: budget and bid in dollars, items by id. */
export function NewCampaignForm({
  advertiserId,
  onCreated,
  onMessage,
}: {
  advertiserId: number;
  onCreated: (c: Campaign) => void;
  onMessage: (msg: string | null) => void;
}) {
  const [name, setName] = useState("");
  const [budget, setBudget] = useState("20.00");
  const [bid, setBid] = useState("0.40");
  const [cap, setCap] = useState("3");
  const [items, setItems] = useState("");
  const [paused, setPaused] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const ids = items
      .split(/[\s,]+/)
      .filter(Boolean)
      .map(Number);
    const b = dollarsToCents(budget);
    const p = dollarsToCents(bid);
    if (Number.isNaN(b) || Number.isNaN(p) || p < RESERVE_CENTS) {
      onMessage(`Budget must be a positive amount and the bid at least $${(RESERVE_CENTS / 100).toFixed(2)}.`);
      return;
    }
    if (ids.length === 0 || ids.some((id) => !Number.isInteger(id) || id < 0)) {
      onMessage("List the promoted items as item ids, e.g. 12, 40, 311.");
      return;
    }
    setBusy(true);
    onMessage(null);
    try {
      const c = await createCampaign({
        advertiser_id: advertiserId,
        name,
        status: paused ? "paused" : "active",
        daily_budget_cents: b,
        bid_cpc_cents: p,
        freq_cap_per_day: Number(cap) || 3,
        item_ids: ids,
      });
      onCreated(c);
      setName("");
      setItems("");
    } catch (err) {
      onMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const field = "flex flex-col gap-1.5 text-xs text-muted-foreground";
  const input = "border-0 bg-muted";
  return (
    <form onSubmit={submit} className="flex flex-wrap items-end gap-3" data-testid="new-campaign">
      <label className={`${field} flex-[2_1_200px]`}>
        New campaign
        <Input className={input} required value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" />
      </label>
      <label className={`${field} flex-[2_1_180px]`}>
        Promoted item ids
        <Input className={input} required value={items} onChange={(e) => setItems(e.target.value)} placeholder="12, 40, 311" />
      </label>
      <label className={`${field} flex-[1_1_110px]`}>
        Daily budget ($)
        <Input className={input} inputMode="decimal" required value={budget} onChange={(e) => setBudget(e.target.value)} />
      </label>
      <label className={`${field} flex-[1_1_100px]`}>
        Max CPC ($)
        <Input className={input} inputMode="decimal" required value={bid} onChange={(e) => setBid(e.target.value)} />
      </label>
      <label className={`${field} flex-[1_1_100px]`}>
        Views / user / day
        <Input className={input} inputMode="numeric" value={cap} onChange={(e) => setCap(e.target.value)} />
      </label>
      <label className="flex h-10 items-center gap-2 text-sm">
        <input type="checkbox" className="size-[18px] accent-black" checked={paused} onChange={(e) => setPaused(e.target.checked)} />
        Start paused
      </label>
      <Button type="submit" variant="brand" className="px-6" disabled={busy}>
        Create
      </Button>
    </form>
  );
}

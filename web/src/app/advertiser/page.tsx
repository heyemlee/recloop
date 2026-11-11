"use client";

import { useEffect, useState } from "react";

import { CampaignRow } from "@/components/campaign-row";
import { PlusIcon } from "@/components/icons";
import { NewCampaignForm } from "@/components/new-campaign-form";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type Advertiser, type Campaign, createAdvertiser, listAdvertisers, listCampaigns } from "@/lib/ads";

const REFRESH_MS = 5000;

/** The advertiser console (design §9.5, §22.1): campaigns, budgets, bids and today's spend. */
export default function AdvertiserPage() {
  const [advertisers, setAdvertisers] = useState<Advertiser[] | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  // Campaigns per advertiser, so a slow answer for another advertiser cannot replace these.
  const [campaigns, setCampaigns] = useState<Record<number, Campaign[]>>({});
  const [message, setMessage] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [newBrand, setNewBrand] = useState("");
  const [adding, setAdding] = useState(false);

  useEffect(() => {
    listAdvertisers()
      .then((list) => {
        setAdvertisers(list);
        setSelected((cur) => cur ?? list[0]?.advertiser_id ?? null);
      })
      .catch((e) => setMessage(`Could not load advertisers: ${e}`));
  }, []);

  // Today's spend moves as clicks are billed: poll while the page is visible.
  useEffect(() => {
    if (selected === null) return;
    let cancelled = false;
    const load = () =>
      listCampaigns(selected)
        .then((list) => !cancelled && setCampaigns((cur) => ({ ...cur, [selected]: list })))
        .catch((e) => !cancelled && setMessage(`Could not load campaigns: ${e}`));
    load();
    const t = setInterval(() => {
      if (document.visibilityState === "visible") load();
    }, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [selected]);

  function replace(c: Campaign) {
    setCampaigns((cur) => ({
      ...cur,
      [c.advertiser_id]: (cur[c.advertiser_id] ?? []).map((x) => (x.campaign_id === c.campaign_id ? c : x)),
    }));
  }

  async function addAdvertiser(e: React.FormEvent) {
    e.preventDefault();
    try {
      const a = await createAdvertiser(newName, newBrand);
      setAdvertisers((cur) => [...(cur ?? []), a]);
      setSelected(a.advertiser_id);
      setNewName("");
      setNewBrand("");
      setAdding(false);
      setMessage(null);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    }
  }

  const shown = selected === null ? null : (campaigns[selected] ?? null);
  const current = advertisers?.find((a) => a.advertiser_id === selected);

  return (
    <main className="mt-9 flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h1 className="text-xl leading-tight font-semibold tracking-[-0.05em]">Campaigns</h1>
          <p className="text-sm text-muted-foreground">
            Pay per click, up to your max CPC. The daily budget (UTC) caps what is billed.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <select
            data-testid="advertiser-select"
            aria-label="Advertiser"
            className="h-11 max-w-64 rounded-full border bg-card px-4 text-sm"
            value={selected ?? ""}
            onChange={(e) => e.target.value && setSelected(Number(e.target.value))}
          >
            {advertisers?.length === 0 && <option value="">no advertiser yet</option>}
            {advertisers?.map((a) => (
              <option key={a.advertiser_id} value={a.advertiser_id}>
                {a.name}
                {a.brand ? ` (${a.brand})` : ""}
              </option>
            ))}
          </select>
          {adding ? (
            <form onSubmit={addAdvertiser} className="flex items-center gap-1.5">
              <Input className="h-11 w-36" required autoFocus placeholder="advertiser name" value={newName} onChange={(e) => setNewName(e.target.value)} aria-label="New advertiser name" />
              <Input className="h-11 w-28" placeholder="brand" value={newBrand} onChange={(e) => setNewBrand(e.target.value)} aria-label="Brand" />
              <Button type="submit" className="h-11">
                Add
              </Button>
              <Button type="button" variant="ghost" className="h-11" onClick={() => setAdding(false)}>
                Cancel
              </Button>
            </form>
          ) : (
            <Button variant="outline" className="h-11 pl-3.5" onClick={() => setAdding(true)}>
              <PlusIcon />
              Advertiser
            </Button>
          )}
        </div>
      </div>

      {message && (
        <p role="status" data-testid="message" className="rounded-[20px] bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {message}
        </p>
      )}

      {selected !== null && (
        <section aria-label={`New campaign${current ? ` for ${current.name}` : ""}`} className="rounded-[28px] bg-card px-7 py-6 shadow-card">
          <NewCampaignForm
            advertiserId={selected}
            onMessage={setMessage}
            onCreated={(c) =>
              setCampaigns((cur) => ({ ...cur, [c.advertiser_id]: [...(cur[c.advertiser_id] ?? []), c] }))
            }
          />
        </section>
      )}

      <section aria-label="Campaigns" className="rounded-[28px] bg-card px-7 py-2 shadow-card">
        {advertisers?.length === 0 && <p className="py-5 text-sm text-muted-foreground">Add an advertiser to start.</p>}
        {shown?.length === 0 && <p className="py-5 text-sm text-muted-foreground">No campaigns yet.</p>}
        {shown && shown.length > 0 && (
          <ul className="divide-y" data-testid="campaigns">
            {shown.map((c) => (
              <CampaignRow key={c.campaign_id} campaign={c} onChange={replace} onMessage={setMessage} />
            ))}
          </ul>
        )}
        {shown === null && advertisers?.length !== 0 && <p className="py-5 text-sm text-muted-foreground">Loading…</p>}
      </section>
      <p className="px-1 text-xs text-muted-foreground">
        Today&rsquo;s spend (UTC) refreshes every {REFRESH_MS / 1000} seconds. The bid is the most a click can cost; the
        auction usually charges less (second price).
      </p>
    </main>
  );
}

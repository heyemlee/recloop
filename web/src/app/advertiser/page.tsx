"use client";

import { useEffect, useState } from "react";

import { CampaignRow } from "@/components/campaign-row";
import { NewCampaignForm } from "@/components/new-campaign-form";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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
      setMessage(null);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    }
  }

  const shown = selected === null ? null : (campaigns[selected] ?? null);
  const current = advertisers?.find((a) => a.advertiser_id === selected);

  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Advertiser</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Campaigns pay per click, at most their max CPC; the daily budget (UTC day) caps what a campaign is billed.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-muted-foreground">Advertiser</span>
            <select
              data-testid="advertiser-select"
              className="h-8 max-w-64 rounded-lg border bg-background px-2 text-sm"
              value={selected ?? ""}
              onChange={(e) => e.target.value && setSelected(Number(e.target.value))}
            >
              {advertisers?.length === 0 && <option value="">none yet</option>}
              {advertisers?.map((a) => (
                <option key={a.advertiser_id} value={a.advertiser_id}>
                  {a.name}
                  {a.brand ? ` (${a.brand})` : ""}
                </option>
              ))}
            </select>
          </label>
          <form onSubmit={addAdvertiser} className="flex items-center gap-1">
            <Input className="h-8 w-32" required placeholder="new advertiser" value={newName} onChange={(e) => setNewName(e.target.value)} aria-label="New advertiser name" />
            <Input className="h-8 w-24" placeholder="brand" value={newBrand} onChange={(e) => setNewBrand(e.target.value)} aria-label="Brand" />
            <Button type="submit" size="sm" variant="outline">
              Add
            </Button>
          </form>
        </div>
      </div>

      {message && (
        <p role="status" data-testid="message" className="rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {message}
        </p>
      )}

      {selected !== null && (
        <Card>
          <CardHeader>
            <CardTitle>New campaign{current ? ` for ${current.name}` : ""}</CardTitle>
            <CardDescription>The bid is the most a click can cost; the auction usually charges less (second price).</CardDescription>
          </CardHeader>
          <CardContent>
            <NewCampaignForm
              advertiserId={selected}
              onMessage={setMessage}
              onCreated={(c) =>
                setCampaigns((cur) => ({ ...cur, [c.advertiser_id]: [...(cur[c.advertiser_id] ?? []), c] }))
              }
            />
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Campaigns</CardTitle>
          <CardDescription>Spend is today&rsquo;s (UTC), refreshed every {REFRESH_MS / 1000} seconds.</CardDescription>
        </CardHeader>
        <CardContent>
          {advertisers?.length === 0 && <p className="text-sm text-muted-foreground">Add an advertiser to start.</p>}
          {shown?.length === 0 && <p className="text-sm text-muted-foreground">No campaigns yet.</p>}
          {shown && shown.length > 0 && (
            <ul className="divide-y" data-testid="campaigns">
              {shown.map((c) => (
                <CampaignRow key={c.campaign_id} campaign={c} onChange={replace} onMessage={setMessage} />
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </main>
  );
}

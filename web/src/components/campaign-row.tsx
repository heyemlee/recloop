"use client";

import Link from "next/link";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { formatPrice } from "@/lib/api";
import {
  type Campaign,
  type CampaignPatch,
  type Report,
  centsToDollars,
  conflictCampaign,
  dollarsToCents,
  endCampaign,
  getReport,
  patchCampaign,
} from "@/lib/ads";
import { cn } from "@/lib/utils";

const STATUS_STYLE: Record<string, string> = {
  active: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200",
  paused: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200",
  ended: "bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
};

/** Today's spend against today's budget (the budget row is capped at the spend if lowered). */
function SpendBar({ c }: { c: Campaign }) {
  const { spent_cents: spent, budget_cents: budget } = c.today;
  const pct = budget > 0 ? Math.min(100, (100 * spent) / budget) : 0;
  return (
    <div className="min-w-40 space-y-1" data-testid="spend">
      <div className="flex justify-between text-xs tabular-nums">
        <span>{formatPrice(spent)}</span>
        <span className="text-muted-foreground">of {formatPrice(budget)}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full", pct >= 90 ? "bg-destructive" : "bg-primary")}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function pct(v: number | null) {
  return v === null ? "—" : `${(100 * v).toFixed(1)}%`;
}

function ReportView({ report }: { report: Report }) {
  const t = report.total;
  const stats: [string, string][] = [
    ["Impressions", String(t.impressions)],
    ["Clicks", String(t.clicks)],
    ["CTR", pct(t.ctr)],
    ["Avg CPC", t.avg_cpc_cents === null ? "—" : formatPrice(Math.round(t.avg_cpc_cents))],
    ["Spend", formatPrice(t.billed_cents)],
    ["Not billed", formatPrice(t.unbilled_cents)],
  ];
  return (
    <div className="space-y-3 rounded-lg bg-muted/50 p-3" data-testid="report">
      <div className="text-xs text-muted-foreground">
        {report.from} to {report.to} (UTC). Clicks are billed once per impression; &ldquo;not billed&rdquo; is
        the auction price of clicks that came in after the budget ran out.
      </div>
      <dl className="grid grid-cols-3 gap-3 sm:grid-cols-6">
        {stats.map(([k, v]) => (
          <div key={k}>
            <dt className="text-xs text-muted-foreground">{k}</dt>
            <dd className="text-sm font-medium tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
      {(t.repeat_clicks > 0 || t.expired_clicks > 0 || t.mismatch_clicks > 0) && (
        <div className="text-xs text-muted-foreground">
          Not charged: {t.repeat_clicks} repeat clicks, {t.expired_clicks} after the 30-minute token expiry,{" "}
          {t.mismatch_clicks} with a token for another user or item.
        </div>
      )}
    </div>
  );
}

/** One campaign: status, bid, budget and spend, with edit, pause / resume, end and report. */
export function CampaignRow({
  campaign: c,
  onChange,
  onMessage,
}: {
  campaign: Campaign;
  onChange: (c: Campaign) => void;
  onMessage: (msg: string | null) => void;
}) {
  // The campaign as it was when editing started: the save sends its version in If-Match, so a
  // change made elsewhere meanwhile (the list refreshes underneath) is a 412, not overwritten.
  const [editing, setEditing] = useState<Campaign | null>(null);
  const [budget, setBudget] = useState("");
  const [bid, setBid] = useState("");
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<Report | null>(null);
  const ended = c.status === "ended";

  async function run(action: () => Promise<Campaign>, done?: () => void) {
    setBusy(true);
    onMessage(null);
    try {
      onChange(await action());
      done?.();
    } catch (e) {
      const current = conflictCampaign(e);
      if (current) {
        // Optimistic lock: someone changed the campaign since we read it.
        onChange(current);
        setEditing(null);
        onMessage(`“${c.name}” was changed elsewhere (now version ${current.version}); review it and try again.`);
      } else {
        onMessage(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(false);
    }
  }

  function save() {
    const base = editing;
    if (base === null) return;
    const patch: CampaignPatch = {};
    const b = dollarsToCents(budget);
    const p = dollarsToCents(bid);
    if (Number.isNaN(b) || Number.isNaN(p)) {
      onMessage("Budget and bid must be positive dollar amounts.");
      return;
    }
    if (b !== base.daily_budget_cents) patch.daily_budget_cents = b;
    if (p !== base.bid_cpc_cents) patch.bid_cpc_cents = p;
    if (Object.keys(patch).length === 0) {
      setEditing(null);
      return;
    }
    run(
      () => patchCampaign(base, patch),
      () => setEditing(null),
    );
  }

  async function toggleReport() {
    if (report) {
      setReport(null);
      return;
    }
    try {
      setReport(await getReport(c.campaign_id));
    } catch (e) {
      onMessage(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <li className="space-y-3 py-4" data-testid="campaign" data-campaign-id={c.campaign_id}>
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="min-w-48 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-medium">{c.name}</span>
            <Badge className={cn("rounded-md", STATUS_STYLE[c.status])}>{c.status}</Badge>
          </div>
          <div className="mt-0.5 text-xs text-muted-foreground">
            #{c.campaign_id} · v{c.version} · items{" "}
            {c.item_ids.map((id, i) => (
              <span key={id}>
                {i > 0 && ", "}
                <Link href={`/items/${id}`} className="underline-offset-2 hover:underline">
                  {id}
                </Link>
              </span>
            ))}{" "}
            · cap {c.freq_cap_per_day}/day
          </div>
        </div>
        {editing ? (
          <div className="flex flex-wrap items-end gap-2">
            <label className="text-xs text-muted-foreground">
              Daily budget ($)
              <Input className="mt-1 w-28" inputMode="decimal" value={budget} onChange={(e) => setBudget(e.target.value)} />
            </label>
            <label className="text-xs text-muted-foreground">
              Max CPC ($)
              <Input className="mt-1 w-24" inputMode="decimal" value={bid} onChange={(e) => setBid(e.target.value)} />
            </label>
            <Button size="sm" disabled={busy} onClick={save}>
              Save
            </Button>
            <Button size="sm" variant="ghost" disabled={busy} onClick={() => setEditing(null)}>
              Cancel
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-6 text-sm">
            <div>
              <div className="text-xs text-muted-foreground">Max CPC</div>
              <div className="tabular-nums">{formatPrice(c.bid_cpc_cents)}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Daily budget</div>
              <div className="tabular-nums">{formatPrice(c.daily_budget_cents)}</div>
            </div>
            <SpendBar c={c} />
          </div>
        )}
        <div className="flex gap-1">
          {!ended && !editing && (
            <Button
              size="sm"
              variant="outline"
              disabled={busy}
              onClick={() => {
                setBudget(centsToDollars(c.daily_budget_cents));
                setBid(centsToDollars(c.bid_cpc_cents));
                setEditing(c);
              }}
            >
              Edit
            </Button>
          )}
          {!ended && (
            <Button
              size="sm"
              variant="outline"
              disabled={busy}
              onClick={() => run(() => patchCampaign(c, { status: c.status === "active" ? "paused" : "active" }))}
            >
              {c.status === "active" ? "Pause" : "Resume"}
            </Button>
          )}
          {!ended && (
            <Button
              size="sm"
              variant="destructive"
              disabled={busy}
              onClick={() => {
                if (window.confirm(`End “${c.name}”? An ended campaign cannot be resumed.`)) run(() => endCampaign(c));
              }}
            >
              End
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={toggleReport}>
            {report ? "Hide report" : "Report"}
          </Button>
        </div>
      </div>
      {report && <ReportView report={report} />}
    </li>
  );
}

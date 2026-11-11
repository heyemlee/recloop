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

// No colour: running is solid, paused an outline, ended greyed out.
const STATUS_STYLE: Record<string, string> = {
  active: "bg-foreground text-background",
  paused: "border-[#cccccc] bg-transparent text-foreground",
  ended: "bg-muted text-muted-foreground",
};

/** Today's spend against today's budget (the budget row is capped at the spend if lowered). */
function SpendBar({ c }: { c: Campaign }) {
  const { spent_cents: spent, budget_cents: budget } = c.today;
  const pct = budget > 0 ? Math.min(100, (100 * spent) / budget) : 0;
  const high = pct >= 90;
  return (
    <div className="flex min-w-[150px] flex-[0_1_200px] flex-col gap-1.5" data-testid="spend">
      <div className="flex justify-between text-xs tabular-nums">
        <span className={cn(high && "font-medium text-destructive")}>
          {formatPrice(spent)}
          {high && " · almost spent"}
        </span>
        <span className="text-muted-foreground">of {formatPrice(budget)} today</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full", high ? "bg-destructive" : "bg-primary")} style={{ width: `${pct}%` }} />
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
    <div className="flex flex-col gap-3.5 rounded-[20px] bg-muted px-6 py-5" data-testid="report">
      <dl className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-4">
        {stats.map(([k, v]) => (
          <div key={k} className="flex flex-col gap-1">
            <dt className="text-xs text-muted-foreground">{k}</dt>
            <dd className="text-xl font-semibold tracking-[-0.05em] tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
      <div className="text-xs leading-[1.4] text-muted-foreground">
        {report.from} to {report.to} (UTC). Clicks are billed once per impression; &ldquo;not billed&rdquo; is
        the auction price of clicks that came in after the budget ran out.
        {(t.repeat_clicks > 0 || t.expired_clicks > 0 || t.mismatch_clicks > 0) && (
          <>
            {" "}
            Not charged: {t.repeat_clicks} repeat clicks, {t.expired_clicks} after the 30-minute token expiry,{" "}
            {t.mismatch_clicks} with a token for another user or item.
          </>
        )}
      </div>
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
  const [reportLoading, setReportLoading] = useState(false);
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
    // The report reads a week of events: it can take several seconds.
    setReportLoading(true);
    try {
      setReport(await getReport(c.campaign_id));
    } catch (e) {
      onMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setReportLoading(false);
    }
  }

  return (
    <li className="flex flex-col gap-4 py-5" data-testid="campaign" data-campaign-id={c.campaign_id}>
      <div className="flex flex-wrap items-center gap-x-8 gap-y-4">
        <div className="flex min-w-0 flex-[1_1_220px] flex-col gap-1">
          <div className="flex items-center gap-2.5">
            <span className="text-base font-medium tracking-[-0.031em]">{c.name}</span>
            <Badge className={cn("tracking-normal", STATUS_STYLE[c.status])}>{c.status}</Badge>
          </div>
          <div className="text-xs text-muted-foreground" title={`version ${c.version} · at most ${c.freq_cap_per_day} impressions per user per day`}>
            #{c.campaign_id} · items{" "}
            {c.item_ids.map((id, i) => (
              <span key={id}>
                {i > 0 && ", "}
                <Link href={`/items/${id}`} className="underline underline-offset-2 hover:text-foreground">
                  {id}
                </Link>
              </span>
            ))}{" "}
            · max CPC {formatPrice(c.bid_cpc_cents)}
          </div>
        </div>
        {editing ? (
          <div className="flex flex-wrap items-end gap-2">
            <label className="flex flex-col gap-1.5 text-xs text-muted-foreground">
              Daily budget ($)
              <Input className="w-28 border-0 bg-muted" inputMode="decimal" value={budget} onChange={(e) => setBudget(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1.5 text-xs text-muted-foreground">
              Max CPC ($)
              <Input className="w-24 border-0 bg-muted" inputMode="decimal" value={bid} onChange={(e) => setBid(e.target.value)} />
            </label>
            <Button disabled={busy} onClick={save}>
              Save
            </Button>
            <Button variant="ghost" disabled={busy} onClick={() => setEditing(null)}>
              Cancel
            </Button>
          </div>
        ) : (
          <SpendBar c={c} />
        )}
        <div className="flex flex-wrap gap-1.5">
          {!ended && !editing && (
            <Button
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
              variant="outline"
              disabled={busy}
              onClick={() => run(() => patchCampaign(c, { status: c.status === "active" ? "paused" : "active" }))}
            >
              {c.status === "active" ? "Pause" : "Resume"}
            </Button>
          )}
          {!ended && (
            <Button
              variant="destructive"
              disabled={busy}
              onClick={() => {
                if (window.confirm(`End “${c.name}”? An ended campaign cannot be resumed.`)) run(() => endCampaign(c));
              }}
            >
              End
            </Button>
          )}
          <Button variant={report ? "secondary" : "ghost"} disabled={reportLoading} onClick={toggleReport}>
            {reportLoading ? "Loading…" : report ? "Hide report" : "Report"}
          </Button>
        </div>
      </div>
      {report && <ReportView report={report} />}
    </li>
  );
}

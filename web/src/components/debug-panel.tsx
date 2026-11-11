"use client";

import { Badge } from "@/components/ui/badge";
import type { RecResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

/** Request-level view: what the backend did for this page (design §22.1). */
export function DebugPanel({ resp }: { resp: RecResponse }) {
  const d = resp.debug;
  return (
    <section data-testid="debug-panel" className="rounded-[28px] bg-card p-6 text-xs tracking-normal shadow-card">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1">
        <Field label="request" value={resp.request_id} mono />
        <Field label="model_version" value={resp.model_version} mono testId="model-version" />
        <Field label="config_version" value={String(resp.config_version)} mono />
        {d?.ranker && <Field label="ranker" value={d.ranker} mono testId="ranker" />}
        {d && <Field label="took" value={`${d.took_ms.toFixed(1)} ms`} />}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground">experiments</span>
        {Object.entries(resp.experiments).map(([layer, a]) => (
          <Badge key={layer} variant="outline" className="font-mono" data-testid={`exp-${layer}`}>
            {layer}: {a.exp_id ? `${a.exp_id} / ${a.variant}` : a.variant} (bucket {a.bucket})
          </Badge>
        ))}
        <span className="ml-2 text-muted-foreground">degraded</span>
        {resp.degraded.length === 0 ? (
          <span>none</span>
        ) : (
          resp.degraded.map((x) => (
            <Badge key={x} variant="destructive">
              {x}
            </Badge>
          ))
        )}
      </div>
      {d && (
        <div className="mt-3 grid gap-4 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          <div>
            <table className="w-full border-collapse">
              <thead className="text-left text-muted-foreground">
                <tr>
                  <th className="py-1 pr-3 font-normal">source</th>
                  <th className="py-1 pr-3 font-normal">RRF weight</th>
                  <th className="py-1 pr-3 font-normal">candidates</th>
                  <th className="py-1 pr-3 font-normal">ms</th>
                  <th className="py-1 font-normal">version / error</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {(d.sources ?? []).map((s) => (
                  <tr key={s.source} className="border-t border-border/60">
                    <td className="py-1 pr-3">
                      {s.source}
                    </td>
                    <td className="py-1 pr-3">{s.weight}</td>
                    <td className="py-1 pr-3">{s.candidates}</td>
                    <td className="py-1 pr-3">{s.took_ms.toFixed(1)}</td>
                    <td className={cn("py-1", s.error && "text-destructive")}>{s.error ?? s.version ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-2 text-muted-foreground">
              {d.candidates} fused candidates · removed: bought {d.filters.bought}, seen in 24h{" "}
              <span data-testid="filter-seen">{d.filters.seen}</span>, sold out {d.filters.sold_out}
              {d.filters.seed > 0 && `, current item ${d.filters.seed}`} · brand-cap promotions{" "}
              {d.diversity_promotions} · backfill {d.backfill}
            </div>
            {d.ads && (
              <div data-testid="debug-ads" className="mt-2 text-muted-foreground">
                ads: <span className="text-foreground">{d.ads.outcome}</span> · {d.ads.stats.campaigns} active campaigns,{" "}
                {d.ads.stats.in_flight} in flight, {d.ads.stats.candidates} candidates · held back: budget{" "}
                {d.ads.stats.budget}, frequency cap {d.ads.stats.freq_cap}, pacing {d.ads.stats.pacing}, no item{" "}
                {d.ads.stats.no_item} · margin {d.ads.policy.safety_margin_clicks} clicks
                {d.ads.policy.pacing && " · paced"}
                {(d.ads.slots ?? []).map((s) => (
                  <span key={s.pos} className="ml-2 font-mono">
                    #{s.pos}: campaign {s.campaign_id} item {s.item_id} {s.price}¢
                  </span>
                ))}
              </div>
            )}
          </div>
          <div>
            <div className="mb-1 text-muted-foreground">
              user {d.user_id} · history version <span data-testid="hv">{d.hv}</span> · recent history used by
              recall (newest first)
            </div>
            <ol data-testid="history" className="space-y-0.5">
              {(d.history ?? []).length === 0 && <li className="text-muted-foreground">no history (cold start)</li>}
              {(d.history ?? []).map((h, i) => (
                <li key={`${h.item_id}-${i}`} className="truncate" data-item-id={h.item_id}>
                  <span className="font-mono text-muted-foreground">{h.item_id}</span> {h.title}
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}
    </section>
  );
}

function Field({ label, value, mono, testId }: { label: string; value: string; mono?: boolean; testId?: string }) {
  return (
    <span>
      <span className="text-muted-foreground">{label} </span>
      <span data-testid={testId} className={cn(mono && "font-mono")}>
        {value}
      </span>
    </span>
  );
}

"use client";

import { useCallback, useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { admin, type Experiment, type Fault, type ModelVersion, type Overview } from "@/lib/admin";

const REFRESH_MS = 3000;

function when(t?: string): string {
  return t ? new Date(t).toLocaleTimeString() : "";
}

const STATUS_VARIANT: Record<ModelVersion["status"], "default" | "secondary" | "outline" | "destructive"> = {
  active: "default",
  canary: "secondary",
  validated: "outline",
  built: "outline",
  retired: "destructive",
};

/** The internal admin page (design §22.1): experiments and their split, model versions with
 * canary / full / rollback, and the model server's fault switch. Not exposed: it works only when
 * the web app runs with RECLOOP_ADMIN=1, through a server-side route to the api's /internal. */
export default function AdminPage() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [fault, setFault] = useState<Fault | null>(null);
  const [faultForm, setFaultForm] = useState<Fault>({ delay_ms: 200, probability: 0.3 });
  const [canaryPct, setCanaryPct] = useState(5);
  const [register, setRegister] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [off, setOff] = useState(false);

  const load = useCallback(async () => {
    try {
      const [o, e, f] = await Promise.all([admin.overview(), admin.experiments(), admin.fault().catch(() => null)]);
      setOverview(o);
      setExperiments(e.experiments);
      setFault(f);
      setOff(false);
    } catch (err) {
      const text = err instanceof Error ? err.message : String(err);
      if (text.includes("admin api is off")) setOff(true);
      else setMessage(text);
    }
  }, []);

  useEffect(() => {
    // Polling: the first load runs from the timer too, so no state is set inside the effect.
    const t = setInterval(() => {
      if (document.visibilityState === "visible") void load();
    }, REFRESH_MS);
    const first = setTimeout(() => void load(), 0);
    return () => {
      clearInterval(t);
      clearTimeout(first);
    };
  }, [load]);

  async function step(label: string, f: () => Promise<unknown>) {
    setBusy(label);
    setMessage(null);
    try {
      await f();
      setMessage(`${label}: done`);
    } catch (err) {
      setMessage(`${label}: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(null);
      void load();
    }
  }

  if (off) {
    return (
      <main className="mx-auto w-full max-w-3xl px-4 py-10">
        <h1 className="text-2xl font-semibold tracking-tight">Admin</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          The admin api is off. It is internal: start the web app with <code>RECLOOP_ADMIN=1</code> (
          <code>make serve-web ADMIN=1</code>) on a machine you control.
        </p>
      </main>
    );
  }

  const versions = overview?.versions ?? [];
  const canary = versions.find((v) => v.status === "canary");

  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Admin</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Internal: model releases (design §6), experiments (§8) and fault injection (§5.2). Active bundle{" "}
          <span className="font-mono">{overview?.active ?? "…"}</span>; a replaced version stays loaded for{" "}
          {overview?.keep_previous ?? "30m"}, so a rollback is a pointer switch.
        </p>
        {message && <p className="mt-2 text-sm" data-testid="admin-message">{message}</p>}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Model versions</CardTitle>
          <CardDescription>
            built → validated → canary → active → retired. The canary pins its share of users (rank-layer buckets) to
            the version; full rollout makes it active and retires the previous one.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-muted-foreground">Canary share</span>
            <Input
              className="h-8 w-20"
              type="number"
              min={1}
              max={50}
              value={canaryPct}
              onChange={(e) => setCanaryPct(Number(e.target.value))}
              aria-label="Canary percent"
            />
            <span className="text-muted-foreground">%</span>
            <Button
              size="sm"
              variant="outline"
              disabled={!canary || busy !== null}
              onClick={() => step("Abort canary", () => admin.abortCanary("aborted from the admin page"))}
            >
              Abort canary
            </Button>
            <Button
              size="sm"
              variant="destructive"
              disabled={busy !== null}
              onClick={() => step("Rollback", () => admin.rollback("rollback from the admin page"))}
            >
              Roll back
            </Button>
            <form
              className="ml-auto flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (register) void step(`Register ${register}`, () => admin.register(register));
                setRegister("");
              }}
            >
              <Input
                className="h-8 w-80 font-mono"
                placeholder="bundle version under artifacts/bundles"
                value={register}
                onChange={(e) => setRegister(e.target.value)}
                aria-label="Register a bundle version"
              />
              <Button size="sm" type="submit" disabled={!register || busy !== null}>
                Register
              </Button>
            </form>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm" data-testid="model-versions">
              <thead className="text-left text-muted-foreground">
                <tr>
                  <th className="py-1 pr-3">Version</th>
                  <th className="py-1 pr-3">Status</th>
                  <th className="py-1 pr-3">Validated</th>
                  <th className="py-1 pr-3">Canary</th>
                  <th className="py-1 pr-3">Activated</th>
                  <th className="py-1 pr-3">Retired</th>
                  <th className="py-1 pr-3">Loaded</th>
                  <th className="py-1" />
                </tr>
              </thead>
              <tbody>
                {versions.map((v) => (
                  <tr key={v.version} className="border-t">
                    <td className="py-1.5 pr-3 font-mono text-xs">{v.version}</td>
                    <td className="py-1.5 pr-3">
                      <Badge variant={STATUS_VARIANT[v.status]}>{v.status}</Badge>
                    </td>
                    <td className="py-1.5 pr-3">{when(v.validated_at)}</td>
                    <td className="py-1.5 pr-3">{when(v.canary_at)}</td>
                    <td className="py-1.5 pr-3">{when(v.activated_at)}</td>
                    <td className="py-1.5 pr-3">{when(v.retired_at)}</td>
                    <td className="py-1.5 pr-3 text-xs text-muted-foreground">
                      {[
                        overview?.loaded.includes(v.version) ? "api" : null,
                        overview?.model_server?.loaded.some((l) => l.version === v.version) ? "model server" : null,
                      ]
                        .filter(Boolean)
                        .join(", ")}
                    </td>
                    <td className="space-x-1 py-1.5 text-right whitespace-nowrap">
                      {(v.status === "built" || v.status === "retired") && (
                        <Button size="xs" variant="outline" disabled={busy !== null} onClick={() => step(`Validate ${v.version}`, () => admin.validate(v.version))}>
                          Validate
                        </Button>
                      )}
                      {v.status === "validated" && (
                        <Button size="xs" variant="secondary" disabled={busy !== null} onClick={() => step(`Canary ${v.version}`, () => admin.canary(v.version, canaryPct))}>
                          Canary {canaryPct}%
                        </Button>
                      )}
                      {(v.status === "validated" || v.status === "canary") && (
                        <Button size="xs" disabled={busy !== null} onClick={() => step(`Full rollout ${v.version}`, () => admin.activate(v.version, "full rollout from the admin page"))}>
                          Full rollout
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details>
            <summary className="cursor-pointer text-sm text-muted-foreground">Release record (latest 50 steps)</summary>
            <ul className="mt-2 space-y-1 text-xs" data-testid="model-events">
              {(overview?.events ?? []).map((e) => (
                <li key={e.id} className="font-mono">
                  {new Date(e.at).toLocaleString()} {e.version} {e.from ?? "∅"} → {e.to} ({e.actor}) {e.note}
                </li>
              ))}
            </ul>
          </details>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Experiments</CardTitle>
          <CardDescription>
            bucket = xxhash64(layer salt + &quot;:&quot; + user id) mod 1000; an experiment owns buckets [from, to) of its layer
            and splits them by the variants&apos; shares. Reloaded by every api instance within 10 s.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <table className="w-full text-sm" data-testid="experiments">
            <thead className="text-left text-muted-foreground">
              <tr>
                <th className="py-1 pr-3">Experiment</th>
                <th className="py-1 pr-3">Layer</th>
                <th className="py-1 pr-3">Status</th>
                <th className="py-1 pr-3">Users</th>
                <th className="py-1 pr-3">Variants (share of the experiment)</th>
                <th className="py-1" />
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.exp_id} className="border-t align-top">
                  <td className="py-1.5 pr-3 font-mono text-xs">{e.exp_id}</td>
                  <td className="py-1.5 pr-3">{e.layer}</td>
                  <td className="py-1.5 pr-3">
                    <Badge variant={e.status === "running" ? "default" : "outline"}>{e.status}</Badge>
                  </td>
                  <td className="py-1.5 pr-3">
                    {((e.bucket_to - e.bucket_from) / 10).toFixed(1)}% <span className="text-xs text-muted-foreground">[{e.bucket_from}, {e.bucket_to})</span>
                  </td>
                  <td className="py-1.5 pr-3 text-xs">
                    {e.variants.map((v) => (
                      <div key={v.name}>
                        <span className="font-medium">{v.name}</span> {v.share}%{" "}
                        <span className="font-mono text-muted-foreground">{v.params ? JSON.stringify(v.params) : ""}</span>
                      </div>
                    ))}
                  </td>
                  <td className="py-1.5 text-right">
                    {e.status === "running" && (
                      <Button size="xs" variant="outline" disabled={busy !== null} onClick={() => step(`Stop ${e.exp_id}`, () => admin.stopExperiment(e.exp_id))}>
                        Stop
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Fault injection</CardTitle>
          <CardDescription>
            The model server delays a share of its calls (up to the caller&apos;s deadline). The api degrades: SASRec
            recall times out and the feed falls back to ItemCF + popularity, without ads.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-2 text-sm">
          <span data-testid="fault-state">
            Now:{" "}
            {fault
              ? fault.delay_ms > 0 && fault.probability > 0
                ? `+${fault.delay_ms} ms on ${(fault.probability * 100).toFixed(0)}% of calls`
                : "off"
              : "unknown"}
          </span>
          <span className="ml-4 text-muted-foreground">delay</span>
          <Input
            className="h-8 w-24"
            type="number"
            min={0}
            value={faultForm.delay_ms}
            onChange={(e) => setFaultForm({ ...faultForm, delay_ms: Number(e.target.value) })}
            aria-label="Delay ms"
          />
          <span className="text-muted-foreground">ms, share</span>
          <Input
            className="h-8 w-24"
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={faultForm.probability}
            onChange={(e) => setFaultForm({ ...faultForm, probability: Number(e.target.value) })}
            aria-label="Probability"
          />
          <Button size="sm" disabled={busy !== null} onClick={() => step("Fault on", () => admin.setFault(faultForm))}>
            Apply
          </Button>
          <Button size="sm" variant="outline" disabled={busy !== null} onClick={() => step("Fault off", () => admin.setFault({ delay_ms: 0, probability: 0 }))}>
            Off
          </Button>
        </CardContent>
      </Card>
    </main>
  );
}

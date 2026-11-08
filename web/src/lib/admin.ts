// The admin page's client for /admin/api/* (forwarded to the api's /internal/admin/*).

export type ModelVersion = {
  version: string;
  status: "built" | "validated" | "canary" | "active" | "retired";
  artifact_uri: string;
  created_at: string;
  validated_at?: string;
  canary_at?: string;
  activated_at?: string;
  retired_at?: string;
};

export type ModelEvent = {
  id: number;
  version: string;
  from?: string;
  to: string;
  at: string;
  actor: string;
  note?: string;
};

export type Experiment = {
  exp_id: string;
  layer: string;
  status: "running" | "stopped" | "draft";
  bucket_from: number;
  bucket_to: number;
  variants: { name: string; share: number; params?: Record<string, unknown> }[];
  config_version: number;
  start_at?: string;
  end_at?: string;
};

export type Overview = {
  active: string;
  loaded: string[];
  versions: ModelVersion[];
  events: ModelEvent[];
  model_server?: { default: string; loaded: { version: string }[]; loading: Record<string, string> };
  canary?: Experiment;
  keep_previous: string;
};

export type Fault = { delay_ms: number; probability: number };

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(`/admin/api/${path}`, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error ?? `${r.status} ${r.statusText}`);
  return data as T;
}

const actor = { actor: "admin-ui" };

export const admin = {
  overview: () => call<Overview>("GET", "models"),
  register: (version: string) => call("POST", "models", { ...actor, version }),
  validate: (version: string) => call("POST", `models/${version}/validate`, actor),
  canary: (version: string, percent: number) => call("POST", `models/${version}/canary`, { ...actor, percent }),
  activate: (version: string, note: string) => call("POST", `models/${version}/activate`, { ...actor, note }),
  abortCanary: (note: string) => call("POST", "canary/abort", { ...actor, note }),
  rollback: (note: string) => call<{ active: string }>("POST", "rollback", { ...actor, note }),
  experiments: () => call<{ experiments: Experiment[]; loaded_config_version: number; layers: string[] }>("GET", "experiments"),
  putExperiment: (e: Experiment) => call("PUT", `experiments/${e.exp_id}`, e),
  stopExperiment: (id: string) => call("POST", `experiments/${id}/stop`),
  fault: () => call<Fault>("GET", "fault"),
  setFault: (f: Fault) => call<Fault>("PUT", "fault", f),
};

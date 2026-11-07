// Advertiser api of the Go service (serving/internal/api/campaigns.go, design §9.5).

const BASE = "/api/v1";

export type Advertiser = { advertiser_id: number; name: string; brand?: string; created_at: string };

export type CampaignStatus = "active" | "paused" | "ended";

export type BudgetDay = { day: string; budget_cents: number; spent_cents: number };

export type Campaign = {
  campaign_id: number;
  advertiser_id: number;
  name: string;
  status: CampaignStatus;
  daily_budget_cents: number;
  bid_cpc_cents: number;
  freq_cap_per_day: number;
  start_at: string | null;
  end_at: string | null;
  item_ids: number[];
  version: number;
  created_at: string;
  updated_at: string;
  today: BudgetDay;
};

export type NewCampaign = {
  advertiser_id: number;
  name: string;
  status: "active" | "paused";
  daily_budget_cents: number;
  bid_cpc_cents: number;
  freq_cap_per_day: number;
  item_ids: number[];
};

export type CampaignPatch = Partial<
  Pick<Campaign, "name" | "status" | "daily_budget_cents" | "bid_cpc_cents" | "freq_cap_per_day" | "item_ids">
>;

export type ReportDay = {
  day: string;
  budget_cents: number;
  spent_cents: number;
  impressions: number;
  clicks: number;
  billed_cents: number;
  unbilled_cents: number;
  repeat_clicks: number;
  expired_clicks: number;
  mismatch_clicks: number;
  ctr: number | null;
  avg_cpc_cents: number | null;
};

export type Report = { campaign_id: number; from: string; to: string; days: ReportDay[]; total: ReportDay };

/** The reserve price: the smallest bid the api accepts (ads.ReserveCents). */
export const RESERVE_CENTS = 5;

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body: unknown,
  ) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown, headers?: Record<string, string>): Promise<T> {
  const r = await fetch(BASE + path, {
    method,
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await r.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!r.ok) {
    const msg = (data as { error?: string } | null)?.error ?? `${r.status} ${text}`;
    throw new ApiError(r.status, msg, data);
  }
  return data as T;
}

export const listAdvertisers = () => request<Advertiser[]>("GET", "/advertisers");

export const createAdvertiser = (name: string, brand: string) =>
  request<Advertiser>("POST", "/advertisers", { name, brand });

export const listCampaigns = (advertiserId: number) =>
  request<Campaign[]>("GET", `/campaigns?advertiser_id=${advertiserId}`);

export const createCampaign = (c: NewCampaign) => request<Campaign>("POST", "/campaigns", c);

/** PATCH with the version that was read; a 412 ApiError carries the current campaign. */
export const patchCampaign = (c: Campaign, patch: CampaignPatch) =>
  request<Campaign>("PATCH", `/campaigns/${c.campaign_id}`, patch, { "If-Match": `"${c.version}"` });

export const endCampaign = (c: Campaign) =>
  request<Campaign>("DELETE", `/campaigns/${c.campaign_id}`, undefined, { "If-Match": `"${c.version}"` });

export const getReport = (id: number) => request<Report>("GET", `/campaigns/${id}/report`);

/** The current campaign sent back with a 412, if this is one. */
export function conflictCampaign(e: unknown): Campaign | null {
  if (e instanceof ApiError && e.status === 412) {
    return (e.body as { campaign?: Campaign } | null)?.campaign ?? null;
  }
  return null;
}

/** "12.5" (dollars) -> 1250 cents; NaN if it is not a positive amount. */
export function dollarsToCents(s: string): number {
  const v = Number(s.trim());
  return Number.isFinite(v) && v > 0 ? Math.round(v * 100) : NaN;
}

export function centsToDollars(c: number): string {
  return (c / 100).toFixed(2);
}

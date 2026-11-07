// Wire types of the Go api (serving/internal/api).

export type Hit = { source: string; rank: number; score: number; via?: number };

export type ItemCard = {
  title: string;
  brand?: string;
  price_cents: number | null;
  image_url?: string;
  stock: number;
};

export type RecItem = {
  position: number;
  item_id: number;
  slot: string;
  source: string;
  score: number;
  impression_id: string;
  sources: Hit[];
  item?: ItemCard;
};

export type Assignment = { bucket: number; exp_id?: string; variant: string };

export type SourceDebug = {
  source: string;
  weight: number;
  candidates: number;
  took_ms: number;
  version?: string;
  error?: string;
};

export type FilterStats = {
  bought: number;
  seen: number;
  sold_out: number;
  seed: number;
  unknown: number;
};

export type Debug = {
  user_id: number;
  hv: number;
  history: { item_id: number; title: string }[] | null;
  sources: SourceDebug[] | null;
  rrf_weights: Record<string, number> | null;
  candidates: number;
  filters: FilterStats;
  diversity_promotions: number;
  backfill: number;
  offset: number;
  took_ms: number;
};

export type RecResponse = {
  request_id: string;
  model_version: string;
  surface: string;
  degraded: string[];
  experiments: Record<string, Assignment>;
  config_version: number;
  items: RecItem[];
  next_cursor?: string;
  debug?: Debug;
};

export type Item = {
  item_id: number;
  title: string;
  brand?: string;
  categories?: string[];
  price_cents: number | null;
  image_url?: string;
  description?: string;
  features?: string[];
  stock: number;
};

export type DemoUser = {
  user_id: number;
  label: string;
  category: string | null;
  history_len: number;
  recent: string[];
};

-- Ads (docs/design.md §9): advertisers and campaigns for the api, the billing transaction's
-- statements (internal/billing) and the campaign report.

-- ---------------------------------------------------------------- advertisers

-- name: CreateAdvertiser :one
INSERT INTO advertisers (name, brand)
VALUES ($1, $2)
RETURNING *;

-- name: ListAdvertisers :many
SELECT * FROM advertisers
ORDER BY advertiser_id;

-- ---------------------------------------------------------------- campaigns

-- name: CreateCampaign :one
INSERT INTO campaigns (
  advertiser_id, name, status, daily_budget_cents, bid_cpc_cents, freq_cap_per_day,
  start_at, end_at, targeting
) VALUES (
  $1, $2, $3, $4, $5, $6, $7, $8, $9
)
RETURNING *;

-- name: GetCampaign :one
SELECT * FROM campaigns
WHERE campaign_id = $1;

-- Campaigns with their items and today's budget row (absent until the first charge or budget
-- change of the day). Both filters are optional.
-- name: ListCampaigns :many
SELECT c.campaign_id, c.advertiser_id, c.name, c.status, c.daily_budget_cents, c.bid_cpc_cents,
       c.freq_cap_per_day, c.start_at, c.end_at, c.targeting, c.version, c.created_at,
       c.updated_at,
       b.budget_cents AS today_budget_cents,
       COALESCE(b.spent_cents, 0)::int AS today_spent_cents,
       ARRAY(
         SELECT ci.item_id FROM campaign_items AS ci
         WHERE ci.campaign_id = c.campaign_id
         ORDER BY ci.item_id
       )::int[] AS item_ids
FROM campaigns AS c
LEFT JOIN campaign_budget_daily AS b ON b.campaign_id = c.campaign_id AND b.day = sqlc.arg(today)
WHERE (sqlc.narg(advertiser_id)::int IS NULL OR c.advertiser_id = sqlc.narg(advertiser_id))
  AND (sqlc.narg(campaign_id)::bigint IS NULL OR c.campaign_id = sqlc.narg(campaign_id))
ORDER BY c.campaign_id;

-- Optimistic lock: no row when the campaign changed since the caller read `version`.
-- name: UpdateCampaign :one
UPDATE campaigns
SET name = sqlc.arg(name),
    status = sqlc.arg(status),
    daily_budget_cents = sqlc.arg(daily_budget_cents),
    bid_cpc_cents = sqlc.arg(bid_cpc_cents),
    freq_cap_per_day = sqlc.arg(freq_cap_per_day),
    start_at = sqlc.narg(start_at),
    end_at = sqlc.narg(end_at),
    targeting = sqlc.narg(targeting),
    version = version + 1,
    updated_at = now()
WHERE campaign_id = sqlc.arg(campaign_id) AND version = sqlc.arg(version)
RETURNING *;

-- name: DeleteCampaignItems :exec
DELETE FROM campaign_items
WHERE campaign_id = $1;

-- name: InsertCampaignItems :exec
INSERT INTO campaign_items (campaign_id, item_id)
SELECT sqlc.arg(campaign_id), unnest(sqlc.arg(item_ids)::int[])
ON CONFLICT DO NOTHING;

-- A budget change during the day applies to today's row at once, but never below what the
-- campaign already spent today (§9.5). Upsert, not update: if billing creates today's row
-- concurrently from the old campaign budget, one of the two waits for the other and the row
-- ends with the new budget either way.
-- name: SetBudgetDay :exec
INSERT INTO campaign_budget_daily (campaign_id, day, budget_cents)
VALUES (sqlc.arg(campaign_id), sqlc.arg(day), sqlc.arg(budget_cents))
ON CONFLICT (campaign_id, day) DO UPDATE
SET budget_cents = GREATEST(EXCLUDED.budget_cents, campaign_budget_daily.spent_cents);

-- ---------------------------------------------------------------- billing (§9.3)

-- The processed marker of a click event. No row: this event_id was processed before.
-- name: ClaimClick :one
INSERT INTO ad_clicks (
  event_id, impression_id, user_id, item_id, token_status, campaign_id, day, received_at
) VALUES (
  $1, $2, $3, $4, $5, $6, $7, $8
)
ON CONFLICT (event_id) DO NOTHING
RETURNING event_id;

-- name: EnsureBudgetDay :exec
INSERT INTO campaign_budget_daily (campaign_id, day, budget_cents)
SELECT c.campaign_id, sqlc.arg(day)::date, c.daily_budget_cents
FROM campaigns AS c
WHERE c.campaign_id = sqlc.arg(campaign_id)
ON CONFLICT (campaign_id, day) DO NOTHING;

-- Serialises the charges of one campaign on one day.
-- name: LockBudgetDay :one
SELECT budget_cents, spent_cents
FROM campaign_budget_daily
WHERE campaign_id = $1 AND day = $2
FOR UPDATE;

-- No row: this impression has a charge already (a repeat click).
-- name: InsertCharge :one
INSERT INTO ad_charges (impression_id, click_event_id, campaign_id, day, price_cents, billed_cents)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (impression_id) DO NOTHING
RETURNING impression_id;

-- name: AddSpend :exec
UPDATE campaign_budget_daily
SET spent_cents = spent_cents + sqlc.arg(billed_cents)
WHERE campaign_id = sqlc.arg(campaign_id) AND day = sqlc.arg(day);

-- ---------------------------------------------------------------- report (§9.5)

-- name: ReportBudgetDays :many
SELECT day, budget_cents, spent_cents
FROM campaign_budget_daily
WHERE campaign_id = sqlc.arg(campaign_id) AND day BETWEEN sqlc.arg(from_day) AND sqlc.arg(to_day)
ORDER BY day;

-- name: ReportChargeDays :many
SELECT day,
       count(*)::int AS clicks,
       COALESCE(sum(billed_cents), 0)::bigint AS billed_cents,
       COALESCE(sum(price_cents), 0)::bigint AS price_cents
FROM ad_charges
WHERE campaign_id = sqlc.arg(campaign_id) AND day BETWEEN sqlc.arg(from_day) AND sqlc.arg(to_day)
GROUP BY day
ORDER BY day;

-- name: ReportClickDays :many
SELECT day, token_status, count(*)::int AS clicks
FROM ad_clicks
WHERE campaign_id = sqlc.arg(campaign_id) AND day BETWEEN sqlc.arg(from_day) AND sqlc.arg(to_day)
GROUP BY day, token_status
ORDER BY day, token_status;

-- Sponsored impressions: impression events whose served record has this campaign at that
-- position (served items carry campaign_id on sponsored slots, §7.1), by the served day.
-- Impression events of late-evening requests may land on the next day.
-- name: ReportImpressionDays :many
SELECT s.event_day AS day, count(DISTINCT i.impression_id)::int AS impressions
FROM events_raw AS i
JOIN events_raw AS s ON s.request_id = i.request_id AND s.type = 'served'
WHERE i.type = 'impression'
  AND i.event_day BETWEEN sqlc.arg(from_day)::date AND sqlc.arg(to_day)::date + 1
  AND s.event_day BETWEEN sqlc.arg(from_day)::date AND sqlc.arg(to_day)::date
  AND s.payload -> 'items' @> jsonb_build_array(jsonb_build_object(
        'pos', split_part(i.impression_id, ':', 2)::int,
        'campaign_id', sqlc.arg(campaign_id)::bigint))
GROUP BY s.event_day
ORDER BY s.event_day;

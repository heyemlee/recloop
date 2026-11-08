-- Raise a spend counter to at least the ledger's sum for its campaign-day (reconciliation,
-- docs/design.md §9.4). Every charge in that sum already has its dedup marker, so a later
-- increment can only come from a newer charge: the counter never passes the ledger.
-- KEYS: 1 spend:{campaign_id}:{yyyymmdd}
-- ARGV: 1 the ledger's sum (cents)  2 counter TTL (ms)
local cur = tonumber(redis.call('GET', KEYS[1]) or '0')
local want = tonumber(ARGV[1])
if cur >= want then
  return 0
end
redis.call('SET', KEYS[1], want, 'PX', ARGV[2])
return want - cur

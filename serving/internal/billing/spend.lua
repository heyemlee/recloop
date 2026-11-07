-- Add one charge to a campaign's spend counter for a day, at most once per click event
-- (docs/design.md §7.3, §9.3, §12.2).
-- KEYS: 1 dedup:billing:{event_id}  2 spend:{campaign_id}:{yyyymmdd}
-- ARGV: 1 cents  2 dedup TTL (ms)  3 counter TTL (ms)
if not redis.call('SET', KEYS[1], '1', 'NX', 'PX', ARGV[2]) then
  return 0  -- already counted
end
redis.call('INCRBY', KEYS[2], ARGV[1])
redis.call('PEXPIRE', KEYS[2], ARGV[3])
return 1

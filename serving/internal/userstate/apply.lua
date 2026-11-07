-- Apply one event to a user's state, at most once (docs/design.md §7.3, §12.2).
-- KEYS: 1 dedup:{group}:{event_id}  2 u:{uid}:hist  3 u:{uid}:hv  4 u:{uid}:bought  5 u:{uid}:seen
-- ARGV: 1 kind (history | purchase | impression)  2 item_id  3 event time (ms)  4 now (ms)
--       5 history size  6 seen window (ms)  7 dedup TTL (ms)
if not redis.call('SET', KEYS[1], '1', 'NX', 'PX', ARGV[7]) then
  return 0  -- already applied
end
local kind, item, ts = ARGV[1], ARGV[2], tonumber(ARGV[3])
local window = tonumber(ARGV[6])
if kind == 'history' or kind == 'purchase' then
  -- History is ordered by event time; GT keeps the newer time when an older event arrives late.
  redis.call('ZADD', KEYS[2], 'GT', ts, item)
  redis.call('ZREMRANGEBYRANK', KEYS[2], 0, -(tonumber(ARGV[5]) + 1))
  redis.call('INCR', KEYS[3])
  if kind == 'purchase' then
    redis.call('SADD', KEYS[4], item)
  end
end
-- A click (or cart / purchase) means the item was seen, even if no impression was reported.
if kind == 'impression' or kind == 'history' or kind == 'purchase' then
  redis.call('ZADD', KEYS[5], 'GT', ts, item)
  redis.call('ZREMRANGEBYSCORE', KEYS[5], '-inf', '(' .. (tonumber(ARGV[4]) - window))
  redis.call('PEXPIRE', KEYS[5], window)
end
return 1

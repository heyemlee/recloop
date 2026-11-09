-- Apply one event to a user's state, at most once (docs/design.md §7.3, §12.2).
-- KEYS: 1 dedup:{group}:{event_id}  2 u:{uid}:hist  3 u:{uid}:hv  4 u:{uid}:bought  5 u:{uid}:seen
--       6 gen:worker  7 gen:queue  8 gen:pending:{uid}
--       then, when ARGV[9] = '1', fc:{uid}:{campaign_id}:{yyyymmdd} (the impression of a sponsored
--       slot), then, when ARGV[10] is not empty, ictr:5m:{bucket} and ictr:1h:{bucket} (the item's
--       CTR buckets of the event's receive time, package itemctr)
-- ARGV: 1 kind (history | purchase | impression)  2 item_id  3 event time (ms)  4 now (ms)
--       5 history size  6 seen window (ms)  7 dedup TTL (ms)  8 frequency-cap TTL (ms)
--       9 '1' when a frequency-cap key follows  10 CTR field (i:{item} | c:{item} | '')
--       11, 12 when the 5-minute and the hourly bucket expire (ms)
--       13 user id  14 generation debounce (ms)  15 generation max wait (ms)  16 pending TTL (ms)
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
  -- Nearline generation (design §11 step 5): while a generation worker is registered, the change
  -- queues the user. The user is due once the history has been quiet for the debounce, at the
  -- latest max wait after the first change the worker has not picked up yet. Without a worker
  -- nothing is written.
  if redis.call('EXISTS', KEYS[6]) == 1 then
    local now = tonumber(ARGV[4])
    redis.call('RPUSH', KEYS[8], ARGV[3] .. ':' .. ARGV[4])
    redis.call('PEXPIRE', KEYS[8], ARGV[16])
    local first = tonumber(string.match(redis.call('LINDEX', KEYS[8], 0), ':(%d+)$'))
    local due = math.min(now + tonumber(ARGV[14]), first + tonumber(ARGV[15]))
    redis.call('ZADD', KEYS[7], due, ARGV[13])
  end
end
-- A click (or cart / purchase) means the item was seen, even if no impression was reported.
if kind == 'impression' or kind == 'history' or kind == 'purchase' then
  redis.call('ZADD', KEYS[5], 'GT', ts, item)
  redis.call('ZREMRANGEBYSCORE', KEYS[5], '-inf', '(' .. (tonumber(ARGV[4]) - window))
  redis.call('PEXPIRE', KEYS[5], window)
end
local k = 9
if ARGV[9] == '1' then
  -- A sponsored impression counts towards the campaign's daily frequency cap (design §9.2).
  if kind == 'impression' then
    redis.call('INCR', KEYS[k])
    redis.call('PEXPIRE', KEYS[k], ARGV[8])
  end
  k = k + 1
end
-- An impression or click counts towards the item's recent CTR (design §9.2 step 3) in the buckets
-- of its receive time; a bucket past its expiry (a replayed old event) is left alone.
if ARGV[10] ~= '' then
  local now = tonumber(ARGV[4])
  for j = 0, 1 do
    local expire = tonumber(ARGV[11 + j])
    if expire > now then
      redis.call('HINCRBY', KEYS[k + j], ARGV[10], 1)
      redis.call('PEXPIREAT', KEYS[k + j], expire)
    end
  end
end
return 1

package userstate

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/redis/go-redis/v9"
)

// INFO memory + errorstats as Redis 8 writes them (CRLF, sections, comments).
const infoText = "# Memory\r\nused_memory:1048000000\r\nused_memory_human:999.45M\r\nmaxmemory:1073741824\r\n" +
	"maxmemory_human:1.00G\r\nmaxmemory_policy:noeviction\r\n\r\n# Errorstats\r\nerrorstat_ERR:count=3\r\n" +
	"errorstat_OOM:count=12\r\n"

func TestParseInfo(t *testing.T) {
	got := ParseInfo(infoText)
	if want := (Info{UsedMemory: 1048000000, MaxMemory: 1073741824, OOMErrors: 12}); got != want {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	if got := ParseInfo("# Memory\r\nused_memory:10\r\nmaxmemory:0\r\n"); got != (Info{UsedMemory: 10}) {
		t.Fatalf("no limit, no errors: %+v", got)
	}
}

func TestFullness(t *testing.T) {
	const max = 1000
	var f Fullness
	steps := []struct {
		in   Info
		full bool
	}{
		{Info{UsedMemory: 500, MaxMemory: max, OOMErrors: 7}, false}, // first reading: errors from before
		{Info{UsedMemory: 979, MaxMemory: max, OOMErrors: 7}, false},
		{Info{UsedMemory: 980, MaxMemory: max, OOMErrors: 7}, true}, // FullAt
		{Info{UsedMemory: 600, MaxMemory: max, OOMErrors: 9}, true}, // writes refused since the last check
		{Info{UsedMemory: 600, MaxMemory: max, OOMErrors: 9}, false},
		{Info{UsedMemory: 600, MaxMemory: max, OOMErrors: 0}, false}, // restarted: counters reset
		{Info{UsedMemory: 5000, MaxMemory: 0, OOMErrors: 0}, false},  // no limit
	}
	for i, s := range steps {
		f.observe(s.in)
		if f.Full() != s.full {
			t.Fatalf("step %d (%+v): full %v, want %v", i, s.in, f.Full(), s.full)
		}
	}
}

func TestErrorKind(t *testing.T) {
	cases := []struct {
		err  error
		kind string
	}{
		{redis.Nil, ErrKindOther},
		{fmt.Errorf("apply: %w", errors.New("OOM command not allowed when used memory > 'maxmemory'.")), ErrKindOOM},
		// The same refusal from inside a script names the script and line instead of the prefix.
		{errors.New("ERR command not allowed when used memory > 'maxmemory'. script: 1726d2d4, on @user_script:6."), ErrKindOOM},
		{context.DeadlineExceeded, ErrKindTimeout},
		{errors.New("dial tcp 127.0.0.1:16390: connect: connection refused"), ErrKindOther},
	}
	for _, c := range cases {
		if got := ErrorKind(c.err); got != c.kind {
			t.Errorf("%v: %q, want %q", c.err, got, c.kind)
		}
	}
	if ErrorKind(nil) != "" {
		t.Error("nil error has a kind")
	}
}

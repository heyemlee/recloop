package events

import (
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
)

func i64(v int64) *int64 { return &v }
func i32(v int32) *int32 { return &v }

func known(id int32) bool { return id >= 0 && id < 100 }

func TestV7TimeAndDay(t *testing.T) {
	before := time.Now().Truncate(time.Millisecond)
	id := uuid.Must(uuid.NewV7())
	after := time.Now()
	at := V7Time(id)
	if at.Before(before) || at.After(after) {
		t.Fatalf("V7Time %v not in [%v, %v]", at, before, after)
	}
	d := Day(time.Date(2026, 11, 6, 23, 59, 59, 0, time.FixedZone("PST", -8*3600)))
	if d.Format(time.DateOnly) != "2026-11-07" || d.Location() != time.UTC {
		t.Fatalf("Day used local time: %v", d)
	}
}

func TestValidate(t *testing.T) {
	now := time.Now()
	rid := uuid.Must(uuid.NewV7())
	good := ClientEvent{
		EventID: uuid.Must(uuid.NewV7()).String(), Type: TypeClick, UserID: i64(42), ItemID: i32(7),
		ImpressionID: ImpressionID(rid, 3), Props: []byte(`{"surface":"home"}`),
	}
	ev, err := Validate(good, now, known)
	if err != nil {
		t.Fatal(err)
	}
	if ev.RequestID.UUID != rid || !ev.RequestID.Valid || ev.Day != Day(ev.Time) || ev.ItemID != 7 {
		t.Fatalf("event %+v", ev)
	}

	old := uuid.Must(uuid.NewV7())
	copy(old[:6], []byte{0, 0, 0, 0, 0, 1}) // 1970
	cases := map[string]func(c *ClientEvent){
		"not a uuid":            func(c *ClientEvent) { c.EventID = "nope" },
		"uuid v4":               func(c *ClientEvent) { c.EventID = uuid.New().String() },
		"too old":               func(c *ClientEvent) { c.EventID = old.String() },
		"served from client":    func(c *ClientEvent) { c.Type = TypeServed },
		"unknown type":          func(c *ClientEvent) { c.Type = "view" },
		"missing user":          func(c *ClientEvent) { c.UserID = nil },
		"unknown item":          func(c *ClientEvent) { c.ItemID = i32(1000) },
		"bad impression id":     func(c *ClientEvent) { c.ImpressionID = "x:1" },
		"position 0":            func(c *ClientEvent) { c.ImpressionID = rid.String() + ":0" },
		"request id mismatch":   func(c *ClientEvent) { c.RequestID = uuid.Must(uuid.NewV7()).String() },
		"impression without id": func(c *ClientEvent) { c.Type = TypeImpression; c.ImpressionID = "" },
		"props not an object":   func(c *ClientEvent) { c.Props = []byte(`[1]`) },
		"props too large":       func(c *ClientEvent) { c.Props = []byte(`{"x":"` + strings.Repeat("a", 3000) + `"}`) },
	}
	for name, mutate := range cases {
		c := good
		mutate(&c)
		if _, err := Validate(c, now, known); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestServedEventAndKinds(t *testing.T) {
	rid := uuid.Must(uuid.NewV7())
	ev, err := Served(42, rid, []byte(`{"model_ver":"v"}`))
	if err != nil {
		t.Fatal(err)
	}
	p := Params(ev)
	if p.Type != TypeServed || p.ItemID.Valid || !p.RequestID.Valid || p.RequestID.Bytes != rid || !p.ServerTs.Valid {
		t.Fatalf("served params %+v", p)
	}
	if p.EventDay.Time != Day(V7Time(uuid.UUID(p.EventID.Bytes))) {
		t.Fatal("served event day is not the day of its event_id")
	}
	if KindOf(TypePurchase) != "purchase" || KindOf(TypeClick) != "history" || KindOf(TypeImpression) != "impression" || KindOf(TypeServed) != "" {
		t.Fatal("event kinds")
	}
}

func TestRecordRoundTrip(t *testing.T) {
	now := time.Now()
	rid := uuid.Must(uuid.NewV7())
	ce := ClientEvent{
		EventID: uuid.Must(uuid.NewV7()).String(), Type: TypeClick, UserID: i64(42), ItemID: i32(7),
		ImpressionID: ImpressionID(rid, 3), AdToken: "v1.token", ClientTS: "2026-11-06T10:00:00.5Z",
		Props: []byte(`{"surface":"home"}`),
	}
	ev, err := Validate(ce, now, known)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := EncodeRecord(ev)
	if err != nil {
		t.Fatal(err)
	}
	back, err := DecodeRecord(raw)
	if err != nil {
		t.Fatal(err)
	}
	if back.ID != ev.ID || back.Day != ev.Day || !back.Time.Equal(ev.Time) || back.Type != ev.Type ||
		back.UserID != 42 || back.ItemID != 7 || back.ImpressionID != ev.ImpressionID || back.RequestID != ev.RequestID ||
		back.AdToken != "v1.token" || !back.ClientTS.Equal(ev.ClientTS) || !back.Received.Equal(ev.Received.Truncate(0)) ||
		string(back.Payload) != `{"surface":"home"}` {
		t.Fatalf("round trip:\n%+v\n%+v", ev, back)
	}
	for name, bad := range map[string]string{
		"not json":      `{`,
		"other version": `{"v":2}`,
		"v4 event_id":   `{"v":1,"event_id":"` + uuid.New().String() + `","type":"click","item_id":1}`,
		"day mismatch":  `{"v":1,"event_id":"` + ev.ID.String() + `","event_day":"2000-01-01","type":"click","item_id":1}`,
		"no item":       `{"v":1,"event_id":"` + ev.ID.String() + `","event_day":"` + ev.Day.Format(time.DateOnly) + `","type":"click"}`,
	} {
		if _, err := DecodeRecord([]byte(bad)); err == nil {
			t.Errorf("%s: decoded", name)
		}
	}
}

func TestAdTokenRules(t *testing.T) {
	now := time.Now()
	rid := uuid.Must(uuid.NewV7())
	click := ClientEvent{EventID: uuid.Must(uuid.NewV7()).String(), Type: TypeClick, UserID: i64(1), ItemID: i32(2),
		ImpressionID: ImpressionID(rid, 3), AdToken: "t"}
	if _, err := Validate(click, now, known); err != nil {
		t.Fatal(err)
	}
	imp := click
	imp.Type = TypeImpression
	if _, err := Validate(imp, now, known); err == nil {
		t.Fatal("ad_token on an impression accepted")
	}
	noImp := click
	noImp.ImpressionID = ""
	if _, err := Validate(noImp, now, known); err == nil {
		t.Fatal("ad_token without impression_id accepted")
	}
	if got := string(PayloadWithAdToken([]byte(`{"a":1}`), "tok")); got != `{"a":1,"ad_token":"tok"}` {
		t.Fatalf("payload %s", got)
	}
	if got := string(PayloadWithAdToken(nil, "tok")); got != `{"ad_token":"tok"}` {
		t.Fatalf("payload %s", got)
	}
	if got := PayloadWithAdToken([]byte(`{"a":1}`), ""); string(got) != `{"a":1}` {
		t.Fatalf("payload %s", got)
	}
}

package cache

import (
	"testing"

	"github.com/google/uuid"
)

func TestCursorRoundTrip(t *testing.T) {
	c := Cursor{RequestID: uuid.Must(uuid.NewV7()), Offset: 40}
	got, err := DecodeCursor(c.Encode())
	if err != nil || got != c {
		t.Fatalf("got %+v, %v", got, err)
	}
	for _, bad := range []string{"", "!!", "bm90LWEtdXVpZDoz", Cursor{RequestID: uuid.New(), Offset: 0}.Encode()} {
		if _, err := DecodeCursor(bad); err == nil {
			t.Errorf("cursor %q should be rejected", bad)
		}
	}
}

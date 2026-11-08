package api

import (
	"slices"
	"testing"
)

func TestParseSeeds(t *testing.T) {
	known := func(id int32) bool { return id >= 0 && id < 100 }
	got, err := parseSeeds("7, 3,42", known)
	if err != nil || !slices.Equal(got, []int32{7, 3, 42}) {
		t.Fatalf("%v %v", got, err)
	}
	if got, err := parseSeeds("", known); err != nil || got != nil {
		t.Fatalf("empty: %v %v", got, err)
	}
	for _, bad := range []string{"7,x", "100", "-1", "1,2,3,4,5,6,7,8,9,10,11", "1,,2"} {
		if _, err := parseSeeds(bad, known); err == nil {
			t.Fatalf("%q accepted", bad)
		}
	}
}

func TestChatHistory(t *testing.T) {
	hist := []int32{1, 2, 3}
	if got := chatHistory(hist, nil); !slices.Equal(got, hist) {
		t.Fatalf("no seeds: %v", got)
	}
	// The first seed is the most relevant: it becomes the most recent item.
	if got := chatHistory(hist, []int32{9, 8}); !slices.Equal(got, []int32{1, 2, 3, 8, 9}) {
		t.Fatalf("with seeds: %v", got)
	}
	if !slices.Equal(hist, []int32{1, 2, 3}) {
		t.Fatal("the user's history was modified")
	}
}

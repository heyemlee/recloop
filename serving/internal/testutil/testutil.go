// Package testutil gives integration tests a real PostgreSQL test database, a Redis database
// and the Redpanda broker, and skips the test when they are not reachable (run `make up`).
//
// Each test binary (package) gets its own database, recloop_test_<package>: `go test ./...`
// runs packages in parallel processes, and two of them migrating or emptying one database
// would race.
package testutil

import (
	"context"
	"database/sql"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	_ "github.com/jackc/pgx/v5/stdlib" // database/sql driver for goose
	"github.com/pressly/goose/v3"
	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/store"
)

const (
	defaultAdminURL = "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"
	testDB          = "recloop_test"
	defaultRedisURL = "redis://localhost:16390/0"
	testRedisDB     = 15
)

// TestDB is the database of this test binary: recloop_test_<package>.
func TestDB() string {
	name := strings.TrimSuffix(filepath.Base(os.Args[0]), ".test")
	name = strings.Map(func(r rune) rune {
		switch {
		case 'a' <= r && r <= 'z', '0' <= r && r <= '9':
			return r
		case 'A' <= r && r <= 'Z':
			return r + 'a' - 'A'
		}
		return '_'
	}, name)
	return testDB + "_" + name
}

func migrationsDir() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "migrations")
}

// PG returns a store on this package's test database, created if needed and migrated to the
// latest version, with the ad, event, order and item tables emptied.
func PG(t *testing.T) *store.Store {
	t.Helper()
	admin := os.Getenv("DATABASE_URL")
	if admin == "" {
		admin = defaultAdminURL
	}
	u, err := url.Parse(admin)
	if err != nil {
		t.Fatal(err)
	}
	name := TestDB()
	u.Path = "/" + name
	testURL := u.String()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, err := pgx.Connect(ctx, admin)
	if err != nil {
		t.Skipf("PostgreSQL not reachable (%v): run `make up`", err)
	}
	var exists bool
	if err := conn.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = $1)", name).Scan(&exists); err != nil {
		t.Fatal(err)
	}
	if !exists {
		if _, err := conn.Exec(ctx, "CREATE DATABASE "+name); err != nil {
			t.Fatal(err)
		}
	}
	conn.Close(ctx)

	db, err := sql.Open("pgx", testURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	goose.SetLogger(goose.NopLogger())
	if err := goose.SetDialect("postgres"); err != nil {
		t.Fatal(err)
	}
	if err := goose.Up(db, migrationsDir()); err != nil {
		t.Fatalf("migrating %s: %v", name, err)
	}
	st, err := store.Open(ctx, testURL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(st.Close)
	if _, err := st.Pool.Exec(ctx, `TRUNCATE ad_clicks, ad_charges, campaign_budget_daily, campaign_items,
		campaigns, advertisers, events_raw, orders, model_version_events, model_versions; DELETE FROM items;
		DELETE FROM users`); err != nil {
		t.Fatal(err)
	}
	return st
}

// SeedItems inserts n items (ids 0..n-1) with brand B{i mod 4} and the given stock.
func SeedItems(t *testing.T, st *store.Store, n int, stock func(i int) int) {
	t.Helper()
	for i := 0; i < n; i++ {
		_, err := st.Pool.Exec(context.Background(),
			"INSERT INTO items (item_id, asin, title, brand, price_cents, stock) VALUES ($1, $2, $3, $4, $5, $6)",
			i, fmt.Sprintf("TEST%06d", i), fmt.Sprintf("Test item %d", i), fmt.Sprintf("B%d", i%4), 100+i, stock(i))
		if err != nil {
			t.Fatal(err)
		}
	}
}

// Redis returns a client on database 15 of the dev Redis, flushed.
func Redis(t *testing.T) *redis.Client {
	t.Helper()
	return RedisDB(t, testRedisDB)
}

// RedisDB returns a client on database db of the dev Redis, flushed. Packages whose tests may
// run at the same time use different databases.
func RedisDB(t *testing.T, db int) *redis.Client {
	t.Helper()
	raw := os.Getenv("REDIS_URL")
	if raw == "" {
		raw = defaultRedisURL
	}
	opts, err := redis.ParseURL(raw)
	if err != nil {
		t.Fatal(err)
	}
	opts.DB = db
	c := redis.NewClient(opts)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := c.Ping(ctx).Err(); err != nil {
		t.Skipf("Redis not reachable (%v): run `make up`", err)
	}
	if err := c.FlushDB(ctx).Err(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

// Brokers returns the dev Redpanda's seed brokers, or skips the test when it is not reachable.
func Brokers(t *testing.T) []string {
	t.Helper()
	raw := os.Getenv("KAFKA_BROKERS")
	if raw == "" {
		raw = "localhost:19092"
	}
	brokers := strings.Split(raw, ",")
	d := net.Dialer{Timeout: time.Second}
	conn, err := d.Dial("tcp", brokers[0])
	if err != nil {
		t.Skipf("Redpanda not reachable (%v): run `make up`", err)
	}
	conn.Close()
	return brokers
}

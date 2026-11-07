package billing

import (
	"context"

	"github.com/jackc/pgx/v5"
)

// SetHook makes Handle call h at every Step (tx is nil at StepCommitted). Tests use it to
// crash a click at a chosen point.
func SetHook(s *Service, h func(ctx context.Context, step Step, tx pgx.Tx) error) { s.hook = h }

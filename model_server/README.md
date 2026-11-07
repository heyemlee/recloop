# model_server

Stateless gRPC model server (docs/design.md §3, §6, §13.2). It loads one bundle built by
`make bundle` (SASRec user encoder as ONNX, item vectors in a FAISS `IndexFlatIP`), validates it
(checksums, dimensions, item count, golden queries) and serves `Retrieve`: history in, top-k
organic candidates out. It never reads Redis or PostgreSQL.

Run from the repo root: `make serve-model` (see `CLAUDE.md`).

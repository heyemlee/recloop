import os

# onnxruntime starts a telemetry uploader thread at import on macOS that can crash the
# interpreter at exit; the server sends nothing anywhere.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

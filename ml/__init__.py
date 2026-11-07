import os

# onnxruntime (bundle build, its tests) starts a telemetry uploader thread at import on macOS
# that can crash the interpreter at exit (seen as a segfault after a passing pytest run).
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

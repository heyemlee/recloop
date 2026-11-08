"""The bundles a model server holds (design §6): several versions at once, one of them the default.

A version is loaded in a background thread and validated (checksums, dimensions, item count,
golden queries, M4's ranking golden data) before it can serve; the version already serving keeps
serving meanwhile. A request names its version (`pin_version`: the canary, or whatever the api
treats as active); without one it gets the default. Switching the default is one reference
swap. Nothing is unloaded on its own: the api keeps the previous active version loaded for 30
minutes after a switch so a rollback is a pointer switch back, then unloads it
(`DELETE /bundles/{version}` on the admin port).

Bundles are found by version under one root directory (the parent of `--bundle`, i.e.
artifacts/bundles): the api and the model server agree on version names, not on paths, which
differ between the host and the bench containers.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from model_server import bundle as bundle_mod

log = logging.getLogger("model_server.registry")


@dataclass
class Entry:
    bundle: bundle_mod.Bundle
    loaded_at: float
    last_used: float


class Registry:
    def __init__(self, root: Path, default: bundle_mod.Bundle, loader=bundle_mod.load):
        self.root = Path(root)
        self._lock = threading.Lock()
        now = time.time()
        self._entries: dict[str, Entry] = {default.version: Entry(default, now, now)}
        self._default = default.version
        self._loading: dict[str, str] = {}  # version -> "loading" | "failed: ..."
        self._loader = loader

    def get(self, version: str = "") -> bundle_mod.Bundle | None:
        """The bundle of `version` (the default when empty), or None if it is not loaded."""
        with self._lock:
            e = self._entries.get(version or self._default)
            if e is None:
                return None
            e.last_used = time.time()
            return e.bundle

    @property
    def default(self) -> bundle_mod.Bundle:
        with self._lock:
            return self._entries[self._default].bundle

    def set_default(self, version: str) -> None:
        with self._lock:
            if version not in self._entries:
                raise KeyError(version)
            old, self._default = self._default, version
        if old != version:
            log.warning("default bundle %s -> %s", old, version)

    def unload(self, version: str) -> None:
        with self._lock:
            if version == self._default:
                raise ValueError(f"{version} is the default; switch the default first")
            if self._entries.pop(version, None) is None:
                raise KeyError(version)
        log.warning("bundle %s unloaded", version)

    def load(self, version: str, wait: bool = False) -> str:
        """Start loading `version` from <root>/<version> (no-op if loaded or loading); returns
        its state: ready, loading or failed: <why>."""
        with self._lock:
            if version in self._entries:
                return "ready"
            if self._loading.get(version) == "loading":
                state = "loading"
            else:
                self._loading[version] = state = "loading"
                threading.Thread(
                    target=self._load, args=(version,), name=f"load-{version}", daemon=True
                ).start()
        if wait:
            while self.state(version) == "loading":
                time.sleep(0.05)
            return self.state(version)
        return state

    def _load(self, version: str) -> None:
        path = self.root / version
        t0 = time.monotonic()
        try:
            if "/" in version or version.startswith(".") or not path.is_dir():
                raise bundle_mod.BundleError(f"no bundle directory {path}")
            b = self._loader(path)
            if b.version != version:
                raise bundle_mod.BundleError(f"{path} holds version {b.version}")
        except Exception as e:  # noqa: BLE001 (any failure makes the version unusable)
            with self._lock:
                self._loading[version] = f"failed: {e}"
            log.error("bundle %s failed to load: %s", version, e)
            return
        now = time.time()
        with self._lock:
            self._entries[version] = Entry(b, now, now)
            self._loading.pop(version, None)
        log.warning("bundle %s loaded and validated in %.2fs", version, time.monotonic() - t0)

    def state(self, version: str) -> str:
        with self._lock:
            if version in self._entries:
                return "ready"
            return self._loading.get(version, "absent")

    def status(self) -> dict:
        with self._lock:
            return {
                "default": self._default,
                "loaded": [
                    {
                        "version": v,
                        "default": v == self._default,
                        "path": str(e.bundle.path),
                        "loaded_at": e.loaded_at,
                        "last_used": e.last_used,
                        "ranker": bool(e.bundle.ranking and e.bundle.ranking.ranker is not None),
                        "pctr": bool(e.bundle.ranking and e.bundle.ranking.pctr is not None),
                    }
                    for v, e in sorted(self._entries.items())
                ],
                "loading": dict(self._loading),
            }

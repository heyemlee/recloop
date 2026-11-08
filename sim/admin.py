"""The api's internal admin endpoints (M4: releases, experiments, faults), for the drills."""

from __future__ import annotations

import time

from sim.client import Api


class AdminError(SystemExit):
    pass


class Admin:
    def __init__(self, api_url: str, actor: str):
        self.api = Api(api_url, timeout=180)  # validation loads a bundle on both sides
        self.actor = actor

    def _post(self, path: str, body: dict | None = None) -> dict:
        status, resp = self.api.post_json(f"/internal/admin/{path}", {"actor": self.actor, **(body or {})})
        if status != 200:
            raise AdminError(f"POST /internal/admin/{path}: {status} {resp}")
        return resp or {}

    def overview(self) -> dict:
        status, body, _ = self.api.get_json("/internal/admin/models")
        if status != 200:
            raise AdminError(f"GET /internal/admin/models: {status} {body}")
        return body

    def register(self, version: str) -> dict:
        return self._post("models", {"version": version})

    def validate(self, version: str) -> dict:
        return self._post(f"models/{version}/validate")

    def canary(self, version: str, percent: int) -> dict:
        return self._post(f"models/{version}/canary", {"percent": percent})

    def activate(self, version: str, note: str = "") -> dict:
        return self._post(f"models/{version}/activate", {"note": note})

    def abort_canary(self, note: str = "") -> dict:
        return self._post("canary/abort", {"note": note})

    def rollback(self, note: str = "") -> dict:
        return self._post("rollback", {"note": note})

    def experiments(self) -> list[dict]:
        status, body, _ = self.api.get_json("/internal/admin/experiments")
        if status != 200:
            raise AdminError(f"GET /internal/admin/experiments: {status} {body}")
        return body["experiments"]

    def put_experiment(self, exp: dict) -> int:
        status, body = self.api.put_json(f"/internal/admin/experiments/{exp['exp_id']}", exp)
        if status != 200:
            raise AdminError(f"PUT experiment {exp['exp_id']}: {status} {body}")
        return body["config_version"]

    def stop_experiment(self, exp_id: str) -> bool:
        status, body = self.api.post_json(f"/internal/admin/experiments/{exp_id}/stop", {})
        if status != 200:
            raise AdminError(f"stop {exp_id}: {status} {body}")
        return body["stopped"]

    def stop_layer(self, layer: str, keep: str = "") -> list[str]:
        """Stop every running experiment of a layer except `keep`."""
        out = []
        for e in self.experiments():
            if e["layer"] == layer and e["status"] == "running" and e["exp_id"] != keep:
                self.stop_experiment(e["exp_id"])
                out.append(e["exp_id"])
        return out

    def wait_config(self, version: int, user: int = 1, timeout: float = 30) -> None:
        """Wait until the api serves with experiment config_version >= version (10 s reload)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, body, _ = self.api.get_json(f"/v1/recommendations?user_id={user}&limit=1")
            if status == 200 and body["config_version"] >= version:
                return
            time.sleep(0.5)
        raise AdminError(f"the api did not load experiment config {version} in {timeout}s")

    def close(self) -> None:
        self.api.close()

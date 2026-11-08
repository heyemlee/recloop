"""A small keep-alive JSON client for the api, and UUIDv7 event ids (design §7.1)."""

from __future__ import annotations

import http.client
import json
import os
import time
import uuid
from urllib.parse import urlsplit


def uuid7(ms: int | None = None) -> str:
    """UUIDv7: 48-bit Unix milliseconds, version 7, variant 10, random bits (RFC 9562)."""
    if ms is None:
        ms = time.time_ns() // 1_000_000
    b = bytearray(os.urandom(16))
    b[0:6] = ms.to_bytes(6, "big")
    b[6] = (b[6] & 0x0F) | 0x70
    b[8] = (b[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(b)))


def uuid7_ms(event_id: str) -> int:
    return int.from_bytes(uuid.UUID(event_id).bytes[:6], "big")


class Api:
    """One persistent HTTP/1.1 connection (one per simulator thread)."""

    def __init__(self, base: str, timeout: float = 10.0):
        u = urlsplit(base)
        self.host, self.port = u.hostname or "127.0.0.1", u.port or 80
        self.timeout = timeout
        self.conn: http.client.HTTPConnection | None = None

    def _request(self, method: str, path: str, body: bytes | None = None):
        for attempt in range(2):  # one reconnect if the kept-alive connection was closed
            if self.conn is None:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            try:
                headers = {"Content-Type": "application/json"} if body is not None else {}
                self.conn.request(method, path, body=body, headers=headers)
                r = self.conn.getresponse()
                data = r.read()
                return r.status, data, dict(r.getheaders())
            except (http.client.HTTPException, OSError):
                self.conn.close()
                self.conn = None
                if attempt == 1:
                    raise
        raise AssertionError("unreachable")

    def get_json(self, path: str):
        status, data, headers = self._request("GET", path)
        try:
            return status, json.loads(data) if data else None, headers
        except json.JSONDecodeError:
            return status, None, headers

    def post_json(self, path: str, doc) -> tuple[int, dict | None]:
        status, data, _ = self._request("POST", path, json.dumps(doc).encode())
        try:
            return status, json.loads(data) if data else None
        except json.JSONDecodeError:
            return status, None

    def delete(self, path: str) -> int:
        status, _, _ = self._request("DELETE", path)
        return status

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()

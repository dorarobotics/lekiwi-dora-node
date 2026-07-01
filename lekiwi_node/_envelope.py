"""SPEC-VENDOR-NODE-V1 envelope helpers — same wire contract as the octos vendors."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


class InvalidEnvelope(ValueError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class CmdRequest:
    request_id: str
    verb: str
    params: dict[str, Any]
    target: str | None
    spec_version: str
    trace_id: str | None


def parse_cmd_request(env: dict[str, Any]) -> CmdRequest:
    if "verb" not in env:
        raise InvalidEnvelope("cmd_request missing required field: verb")
    return CmdRequest(
        request_id=str(env.get("request_id", "")),
        verb=str(env["verb"]),
        params=dict(env.get("params") or {}),
        target=env.get("target"),
        spec_version=str(env.get("spec_version", "1.0.0")),
        trace_id=env.get("trace_id"),
    )


def build_cmd_response(request: CmdRequest, *, ok: bool, code: str,
                       data: dict[str, Any] | None = None, msg: str = "") -> dict[str, Any]:
    resp: dict[str, Any] = {
        "envelope_version": "1.0",
        "spec_version": request.spec_version,
        "request_id": request.request_id,
        "ok": bool(ok),
        "code": str(code),
        "msg": msg or "",
        "ts": _now_iso(),
        "data": data if data is not None else {},
    }
    if request.trace_id is not None:
        resp["trace_id"] = request.trace_id
    return resp

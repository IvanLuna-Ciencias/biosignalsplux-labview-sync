"""Message definitions for Python-LabVIEW session coordination."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

from biosignalsplux_labview_sync.session import (
    SessionError,
    validate_session_id,
)


PROTOCOL_VERSION = 1

MESSAGE_TYPES = {
    "PREPARE",
    "READY",
    "START",
    "START_RECEIVED",
    "TRAJECTORY_STARTED",
    "TRAJECTORY_DONE",
    "STOP",
    "STOPPED",
    "ERROR",
}


class ProtocolError(ValueError):
    """Raised when a protocol message is malformed or invalid."""


def _require_finite_number(value: Any, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ProtocolError(f"{field_name} must be a finite number.")

    return float(value)


def validate_trajectory_payload(payload: dict[str, Any]) -> None:
    """Validate the four trajectory parameters sent in PREPARE."""

    required_fields = {
        "q_start_deg",
        "q_end_deg",
        "frequency_hz",
        "cycles",
    }

    missing_fields = required_fields - payload.keys()

    if missing_fields:
        raise ProtocolError(
            "PREPARE payload is missing: "
            + ", ".join(sorted(missing_fields))
        )

    q_start = _require_finite_number(
        payload["q_start_deg"],
        "payload.q_start_deg",
    )
    q_end = _require_finite_number(
        payload["q_end_deg"],
        "payload.q_end_deg",
    )
    frequency = _require_finite_number(
        payload["frequency_hz"],
        "payload.frequency_hz",
    )

    cycles = payload["cycles"]

    if (
        not isinstance(cycles, int)
        or isinstance(cycles, bool)
        or cycles <= 0
    ):
        raise ProtocolError(
            "payload.cycles must be a positive integer."
        )

    if q_start == q_end:
        raise ProtocolError(
            "payload.q_start_deg and payload.q_end_deg "
            "must be different."
        )

    if frequency <= 0:
        raise ProtocolError(
            "payload.frequency_hz must be greater than zero."
        )


def validate_message(message: dict[str, Any]) -> None:
    """Validate one decoded Python-LabVIEW protocol message."""

    if not isinstance(message, dict):
        raise ProtocolError("A protocol message must be a JSON object.")

    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol_version must be {PROTOCOL_VERSION}."
        )

    message_type = message.get("type")

    if message_type not in MESSAGE_TYPES:
        raise ProtocolError(
            f"Unsupported message type: {message_type!r}."
        )

    session_id = message.get("session_id")

    try:
        validate_session_id(session_id)
    except SessionError as exc:
        raise ProtocolError(str(exc)) from exc

    sequence = message.get("sequence")

    if (
        not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence < 0
    ):
        raise ProtocolError(
            "sequence must be a non-negative integer."
        )

    monotonic_time = _require_finite_number(
        message.get("sender_monotonic_s"),
        "sender_monotonic_s",
    )

    if monotonic_time < 0:
        raise ProtocolError(
            "sender_monotonic_s cannot be negative."
        )

    timestamp_text = message.get("timestamp_utc")

    if not isinstance(timestamp_text, str):
        raise ProtocolError(
            "timestamp_utc must be an ISO 8601 string."
        )

    try:
        timestamp = datetime.fromisoformat(
            timestamp_text.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ProtocolError(
            "timestamp_utc is not a valid ISO 8601 timestamp."
        ) from exc

    if timestamp.utcoffset() is None:
        raise ProtocolError(
            "timestamp_utc must include timezone information."
        )

    payload = message.get("payload")

    if not isinstance(payload, dict):
        raise ProtocolError("payload must be a JSON object.")

    if message_type == "PREPARE":
        validate_trajectory_payload(payload)


def create_message(
    *,
    message_type: str,
    session_id: str,
    sequence: int,
    sender_monotonic_s: float,
    payload: dict[str, Any] | None = None,
    timestamp_utc: datetime | None = None,
) -> dict[str, Any]:
    """Create and validate one protocol message."""

    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc)

    if timestamp_utc.utcoffset() is None:
        raise ProtocolError(
            "timestamp_utc must include timezone information."
        )

    message = {
        "protocol_version": PROTOCOL_VERSION,
        "type": message_type,
        "session_id": session_id,
        "sequence": sequence,
        "timestamp_utc": timestamp_utc.isoformat(),
        "sender_monotonic_s": float(sender_monotonic_s),
        "payload": payload or {},
    }

    validate_message(message)
    return message


def encode_message(message: dict[str, Any]) -> bytes:
    """Validate and encode one message as compact UTF-8 JSON."""

    validate_message(message)

    return json.dumps(
        message,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def decode_message(data: bytes) -> dict[str, Any]:
    """Decode and validate one UTF-8 JSON protocol message."""

    if not isinstance(data, bytes):
        raise ProtocolError("Encoded message data must be bytes.")

    try:
        message = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ProtocolError(
            "Message is not valid UTF-8."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            f"Message is not valid JSON: {exc}"
        ) from exc

    validate_message(message)
    return message
"""Tests for Python-LabVIEW protocol messages."""

from datetime import datetime, timezone

import pytest

from biosignalsplux_labview_sync.protocol import (
    ProtocolError,
    create_message,
    decode_message,
    encode_message,
)


def test_prepare_message_round_trip() -> None:
    message = create_message(
        message_type="PREPARE",
        session_id="passive_elbow_001",
        sequence=0,
        sender_monotonic_s=10.25,
        timestamp_utc=datetime(
            2026,
            7,
            10,
            19,
            30,
            tzinfo=timezone.utc,
        ),
        payload={
            "q_start_deg": 0.0,
            "q_end_deg": 90.0,
            "frequency_hz": 0.1,
            "cycles": 5,
        },
    )

    decoded = decode_message(encode_message(message))

    assert decoded == message
    assert decoded["type"] == "PREPARE"
    assert decoded["payload"]["q_end_deg"] == 90.0
    assert decoded["payload"]["cycles"] == 5


def test_start_message_accepts_empty_payload() -> None:
    message = create_message(
        message_type="START",
        session_id="passive_elbow_001",
        sequence=1,
        sender_monotonic_s=12.0,
    )

    assert message["payload"] == {}


def test_prepare_rejects_non_positive_frequency() -> None:
    with pytest.raises(
        ProtocolError,
        match="frequency_hz must be greater than zero",
    ):
        create_message(
            message_type="PREPARE",
            session_id="passive_elbow_001",
            sequence=0,
            sender_monotonic_s=0.0,
            payload={
                "q_start_deg": 0.0,
                "q_end_deg": 90.0,
                "frequency_hz": 0.0,
                "cycles": 5,
            },
        )


def test_prepare_rejects_equal_positions() -> None:
    with pytest.raises(
        ProtocolError,
        match="must be different",
    ):
        create_message(
            message_type="PREPARE",
            session_id="passive_elbow_001",
            sequence=0,
            sender_monotonic_s=0.0,
            payload={
                "q_start_deg": 45.0,
                "q_end_deg": 45.0,
                "frequency_hz": 0.1,
                "cycles": 5,
            },
        )


def test_prepare_rejects_invalid_cycle_count() -> None:
    with pytest.raises(
        ProtocolError,
        match="cycles must be a positive integer",
    ):
        create_message(
            message_type="PREPARE",
            session_id="passive_elbow_001",
            sequence=0,
            sender_monotonic_s=0.0,
            payload={
                "q_start_deg": 0.0,
                "q_end_deg": 90.0,
                "frequency_hz": 0.1,
                "cycles": 0,
            },
        )


def test_unknown_message_type_is_rejected() -> None:
    with pytest.raises(
        ProtocolError,
        match="Unsupported message type",
    ):
        create_message(
            message_type="MOVE_NOW",
            session_id="passive_elbow_001",
            sequence=0,
            sender_monotonic_s=0.0,
        )


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(
        ProtocolError,
        match="not valid JSON",
    ):
        decode_message(b"{not-json}")
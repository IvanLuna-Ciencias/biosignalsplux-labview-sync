"""Tests for synchronized session metadata."""

import json
from datetime import datetime, timezone

import pytest

from biosignalsplux_labview_sync.session_metadata import (
    BiosignalspluxMetadata,
    LabVIEWMetadata,
    MetadataError,
    ParticipantMetadata,
    SessionMetadata,
    TrajectoryMetadata,
    build_session_id,
    generate_participant_id,
    write_metadata_json,
)


def make_metadata() -> SessionMetadata:
    participant = ParticipantMetadata(
        participant_id="P-ABC123",
        name_or_alias="Participant 01",
        age_years=25,
        sex_assigned_at_birth="male",
        instrumented_arm="right",
        notes="Passive elbow test.",
    )

    biosignalsplux = BiosignalspluxMetadata(
        device_address="B00:07:80:4D:2F:0B",
    )

    trajectory = TrajectoryMetadata(
        joint="elbow",
        start_deg=0.0,
        end_deg=90.0,
        frequency_hz=0.1,
        cycles=5,
        setpoint_rate_hz=50.0,
        move_to_start_s=3.0,
        return_duration_s=3.0,
        neutral_hold_s=0.5,
    )

    labview = LabVIEWMetadata(
        host="172.22.11.2",
        port=5005,
        enabled=True,
    )

    return SessionMetadata(
        session_id="P-ABC123_elbow_20260713_120000",
        created_at_utc="2026-07-13T12:00:00+00:00",
        participant=participant,
        biosignalsplux=biosignalsplux,
        trajectory=trajectory,
        labview=labview,
    )


def test_participant_id_has_expected_format() -> None:
    participant_id = generate_participant_id()

    assert participant_id.startswith("P-")
    assert len(participant_id) == 8


def test_session_id_contains_participant_joint_and_time() -> None:
    session_id = build_session_id(
        participant_id="P-ABC123",
        joint="elbow",
        timestamp=datetime(
            2026,
            7,
            13,
            18,
            30,
            tzinfo=timezone.utc,
        ),
    )

    assert session_id == "P-ABC123_elbow_20260713_183000"


def test_trajectory_duration_is_calculated() -> None:
    metadata = make_metadata()

    assert metadata.trajectory.duration_s == pytest.approx(50.0)
    assert metadata.to_dict()["trajectory"]["duration_s"] == pytest.approx(
        50.0
    )


def test_invalid_age_is_rejected() -> None:
    with pytest.raises(
        MetadataError,
        match="age_years",
    ):
        ParticipantMetadata(
            participant_id="P-ABC123",
            name_or_alias="",
            age_years=0,
            sex_assigned_at_birth="male",
            instrumented_arm="right",
        )


def test_invalid_joint_is_rejected() -> None:
    with pytest.raises(
        MetadataError,
        match="joint must be one of",
    ):
        TrajectoryMetadata(
            joint="wrist",
            start_deg=0.0,
            end_deg=20.0,
            frequency_hz=0.1,
            cycles=1,
            setpoint_rate_hz=50.0,
            move_to_start_s=3.0,
            return_duration_s=3.0,
            neutral_hold_s=0.5,
        )


def test_invalid_labview_port_is_rejected() -> None:
    with pytest.raises(
        MetadataError,
        match="port",
    ):
        LabVIEWMetadata(
            host="127.0.0.1",
            port=70000,
        )


def test_metadata_json_is_written(tmp_path) -> None:
    metadata = make_metadata()
    output_path = tmp_path / "metadata.json"

    result = write_metadata_json(
        output_path,
        metadata,
    )

    saved = json.loads(
        result.read_text(encoding="utf-8")
    )

    assert saved["session_id"] == metadata.session_id
    assert saved["participant"]["participant_id"] == "P-ABC123"
    assert saved["biosignalsplux"]["sampling_rate_hz"] == 1000
    assert saved["biosignalsplux"]["channels"] == [
        "emg_0",
        "emg_1",
    ]
    assert saved["trajectory"]["duration_s"] == pytest.approx(50.0)


def test_created_timestamp_requires_timezone() -> None:
    metadata = make_metadata()

    with pytest.raises(
        MetadataError,
        match="timezone",
    ):
        SessionMetadata(
            session_id=metadata.session_id,
            created_at_utc="2026-07-13T12:00:00",
            participant=metadata.participant,
            biosignalsplux=metadata.biosignalsplux,
            trajectory=metadata.trajectory,
            labview=metadata.labview,
        )
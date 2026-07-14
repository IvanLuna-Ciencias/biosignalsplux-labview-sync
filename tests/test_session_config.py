"""Tests for synchronized-session configuration assembly."""

import json
from datetime import datetime, timezone

import pytest

from biosignalsplux_labview_sync.session_config import (
    SessionFormData,
    build_session_metadata,
    metadata_output_path,
    save_session_metadata,
)
from biosignalsplux_labview_sync.trajectory import TrajectoryError


def make_form() -> SessionFormData:
    return SessionFormData(
        participant_id="P-ABC123",
        name_or_alias="Participant 01",
        age_years=25,
        sex_assigned_at_birth="male",
        instrumented_arm="right",
        notes="Passive elbow acquisition.",
        device_address="B00:07:80:4D:2F:0B",
        joint="elbow",
        start_deg=0.0,
        end_deg=90.0,
        frequency_hz=0.1,
        cycles=5,
        setpoint_rate_hz=50.0,
        move_to_start_s=3.0,
        return_duration_s=3.0,
        neutral_hold_s=0.5,
        labview_host="172.22.11.2",
        labview_port=5005,
        labview_enabled=True,
    )


def test_build_session_metadata() -> None:
    metadata = build_session_metadata(
        make_form(),
        timestamp=datetime(
            2026,
            7,
            13,
            18,
            30,
            tzinfo=timezone.utc,
        ),
    )

    assert (
        metadata.session_id
        == "P-ABC123_elbow_20260713_183000"
    )
    assert metadata.biosignalsplux.sampling_rate_hz == 1000
    assert metadata.trajectory.duration_s == pytest.approx(50.0)
    assert metadata.labview.port == 5005


def test_joint_specific_limits_are_enforced() -> None:
    form = make_form()

    invalid_form = SessionFormData(
        **{
            **form.__dict__,
            "end_deg": 120.0,
        }
    )

    with pytest.raises(
        TrajectoryError,
        match="outside the selected joint limits",
    ):
        build_session_metadata(invalid_form)


def test_metadata_output_path_uses_session_directory() -> None:
    metadata = build_session_metadata(
        make_form(),
        timestamp=datetime(
            2026,
            7,
            13,
            18,
            30,
            tzinfo=timezone.utc,
        ),
    )

    path = metadata_output_path(
        "outputs",
        metadata,
    )

    assert path.as_posix().endswith(
        "outputs/P-ABC123_elbow_20260713_183000/"
        "metadata_P-ABC123_elbow_20260713_183000.json"
    )


def test_save_session_metadata(tmp_path) -> None:
    metadata = build_session_metadata(
        make_form(),
        timestamp=datetime(
            2026,
            7,
            13,
            18,
            30,
            tzinfo=timezone.utc,
        ),
    )

    saved_path = save_session_metadata(
        base_output_dir=tmp_path,
        metadata=metadata,
    )

    saved = json.loads(
        saved_path.read_text(encoding="utf-8")
    )

    assert saved["session_id"] == metadata.session_id
    assert saved["biosignalsplux"]["sampling_rate_hz"] == 1000
    assert saved["trajectory"]["joint"] == "elbow"


def test_labview_can_be_disabled_for_dry_runs() -> None:
    form = make_form()

    dry_run_form = SessionFormData(
        **{
            **form.__dict__,
            "labview_enabled": False,
        }
    )

    metadata = build_session_metadata(dry_run_form)

    assert metadata.labview.enabled is False
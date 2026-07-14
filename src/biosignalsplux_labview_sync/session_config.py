"""Build and save validated synchronized-session configurations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from biosignalsplux_labview_sync.session_metadata import (
    BiosignalspluxMetadata,
    LabVIEWMetadata,
    ParticipantMetadata,
    SessionMetadata,
    TrajectoryMetadata,
    build_session_id,
    write_metadata_json,
)
from biosignalsplux_labview_sync.trajectory import CosineTrajectory


@dataclass(frozen=True)
class SessionFormData:
    """Values collected from the synchronized-session form."""

    participant_id: str
    name_or_alias: str
    age_years: int
    sex_assigned_at_birth: str
    instrumented_arm: str
    notes: str

    device_address: str

    joint: str
    start_deg: float
    end_deg: float
    frequency_hz: float
    cycles: int
    setpoint_rate_hz: float
    move_to_start_s: float
    return_duration_s: float
    neutral_hold_s: float

    labview_host: str
    labview_port: int
    labview_enabled: bool


def build_session_metadata(
    form: SessionFormData,
    *,
    timestamp: datetime | None = None,
) -> SessionMetadata:
    """Build one immutable and validated metadata snapshot."""

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    if timestamp.utcoffset() is None:
        raise ValueError(
            "timestamp must include timezone information."
        )

    timestamp_utc = timestamp.astimezone(timezone.utc)

    participant = ParticipantMetadata(
        participant_id=form.participant_id,
        name_or_alias=form.name_or_alias,
        age_years=form.age_years,
        sex_assigned_at_birth=form.sex_assigned_at_birth,
        instrumented_arm=form.instrumented_arm,
        notes=form.notes,
    )

    biosignalsplux = BiosignalspluxMetadata(
        device_address=form.device_address,
        sampling_rate_hz=1000,
        resolution_bits=16,
        channel_mask=3,
        channels=("emg_0", "emg_1"),
    )

    trajectory = TrajectoryMetadata(
        joint=form.joint,
        start_deg=form.start_deg,
        end_deg=form.end_deg,
        frequency_hz=form.frequency_hz,
        cycles=form.cycles,
        setpoint_rate_hz=form.setpoint_rate_hz,
        move_to_start_s=form.move_to_start_s,
        return_duration_s=form.return_duration_s,
        neutral_hold_s=form.neutral_hold_s,
    )

    # Enforce the same joint-specific limits used by the trajectory generator.
    CosineTrajectory(
        joint=form.joint,
        start_deg=form.start_deg,
        end_deg=form.end_deg,
        frequency_hz=form.frequency_hz,
        cycles=form.cycles,
    )

    labview = LabVIEWMetadata(
        host=form.labview_host,
        port=form.labview_port,
        enabled=form.labview_enabled,
    )

    session_id = build_session_id(
        participant_id=participant.participant_id,
        joint=trajectory.joint,
        timestamp=timestamp_utc,
    )

    return SessionMetadata(
        session_id=session_id,
        created_at_utc=timestamp_utc.isoformat(),
        participant=participant,
        biosignalsplux=biosignalsplux,
        trajectory=trajectory,
        labview=labview,
    )


def metadata_output_path(
    base_output_dir: str | Path,
    metadata: SessionMetadata,
) -> Path:
    """Return the metadata path for one session."""

    session_directory = (
        Path(base_output_dir)
        / metadata.session_id
    )

    return (
        session_directory
        / f"metadata_{metadata.session_id}.json"
    )


def save_session_metadata(
    *,
    base_output_dir: str | Path,
    metadata: SessionMetadata,
) -> Path:
    """Save one session metadata snapshot."""

    return write_metadata_json(
        metadata_output_path(
            base_output_dir,
            metadata,
        ),
        metadata,
    )
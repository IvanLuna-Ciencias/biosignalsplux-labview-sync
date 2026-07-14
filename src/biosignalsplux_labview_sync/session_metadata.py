"""Validated metadata models for synchronized acquisition sessions."""

from __future__ import annotations

import json
import math
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METADATA_SCHEMA_VERSION = 1

VALID_JOINTS = ("shoulder", "elbow", "rotation")
VALID_ARMS = ("right", "left")
VALID_SEX_VALUES = (
    "female",
    "male",
    "intersex",
    "prefer_not_to_say",
)


class MetadataError(ValueError):
    """Raised when session metadata is incomplete or invalid."""


def _finite_number(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise MetadataError(
            f"{field_name} must be a finite number."
        )

    return float(value)


def _non_empty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetadataError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def generate_participant_id(prefix: str = "P") -> str:
    """Generate a non-identifying participant code."""

    clean_prefix = re.sub(
        r"[^A-Za-z0-9]+",
        "",
        prefix.strip(),
    ).upper()

    if not clean_prefix:
        raise MetadataError(
            "Participant ID prefix must contain letters or numbers."
        )

    random_code = secrets.token_hex(3).upper()
    return f"{clean_prefix}-{random_code}"


def build_session_id(
    *,
    participant_id: str,
    joint: str,
    timestamp: datetime | None = None,
) -> str:
    """Build a filesystem-safe session identifier."""

    participant = _non_empty_text(
        participant_id,
        "participant_id",
    )

    if joint not in VALID_JOINTS:
        raise MetadataError(
            f"joint must be one of: {', '.join(VALID_JOINTS)}."
        )

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    if timestamp.utcoffset() is None:
        raise MetadataError(
            "Session timestamp must include timezone information."
        )

    safe_participant = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        participant,
    ).strip("_")

    if not safe_participant:
        raise MetadataError(
            "participant_id does not contain usable characters."
        )

    timestamp_text = timestamp.astimezone(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    return f"{safe_participant}_{joint}_{timestamp_text}"


@dataclass(frozen=True)
class ParticipantMetadata:
    """Participant information stored locally with the session."""

    participant_id: str
    name_or_alias: str
    age_years: int
    sex_assigned_at_birth: str
    instrumented_arm: str
    notes: str = ""

    def __post_init__(self) -> None:
        participant_id = _non_empty_text(
            self.participant_id,
            "participant_id",
        )

        if (
            not isinstance(self.age_years, int)
            or isinstance(self.age_years, bool)
            or not 1 <= self.age_years <= 120
        ):
            raise MetadataError(
                "age_years must be an integer between 1 and 120."
            )

        if self.sex_assigned_at_birth not in VALID_SEX_VALUES:
            raise MetadataError(
                "sex_assigned_at_birth must be one of: "
                + ", ".join(VALID_SEX_VALUES)
                + "."
            )

        if self.instrumented_arm not in VALID_ARMS:
            raise MetadataError(
                "instrumented_arm must be right or left."
            )

        if not isinstance(self.name_or_alias, str):
            raise MetadataError(
                "name_or_alias must be a string."
            )

        if not isinstance(self.notes, str):
            raise MetadataError(
                "notes must be a string."
            )

        object.__setattr__(
            self,
            "participant_id",
            participant_id,
        )
        object.__setattr__(
            self,
            "name_or_alias",
            self.name_or_alias.strip(),
        )
        object.__setattr__(
            self,
            "notes",
            self.notes.strip(),
        )


@dataclass(frozen=True)
class BiosignalspluxMetadata:
    """Biosignalsplux acquisition configuration."""

    device_address: str
    sampling_rate_hz: int = 1000
    resolution_bits: int = 16
    channel_mask: int = 3
    channels: tuple[str, ...] = ("emg_0", "emg_1")

    def __post_init__(self) -> None:
        address = _non_empty_text(
            self.device_address,
            "device_address",
        )

        if (
            not isinstance(self.sampling_rate_hz, int)
            or isinstance(self.sampling_rate_hz, bool)
            or self.sampling_rate_hz <= 0
        ):
            raise MetadataError(
                "sampling_rate_hz must be a positive integer."
            )

        if (
            not isinstance(self.resolution_bits, int)
            or isinstance(self.resolution_bits, bool)
            or self.resolution_bits <= 0
        ):
            raise MetadataError(
                "resolution_bits must be a positive integer."
            )

        if (
            not isinstance(self.channel_mask, int)
            or isinstance(self.channel_mask, bool)
            or self.channel_mask <= 0
        ):
            raise MetadataError(
                "channel_mask must be a positive integer."
            )

        if not self.channels:
            raise MetadataError(
                "At least one Biosignalsplux channel is required."
            )

        if any(
            not isinstance(channel, str) or not channel.strip()
            for channel in self.channels
        ):
            raise MetadataError(
                "All channel names must be non-empty strings."
            )

        object.__setattr__(
            self,
            "device_address",
            address,
        )
        object.__setattr__(
            self,
            "channels",
            tuple(channel.strip() for channel in self.channels),
        )


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Selected three-DOF trajectory configuration."""

    joint: str
    start_deg: float
    end_deg: float
    frequency_hz: float
    cycles: int
    setpoint_rate_hz: float
    move_to_start_s: float
    return_duration_s: float
    neutral_hold_s: float

    def __post_init__(self) -> None:
        if self.joint not in VALID_JOINTS:
            raise MetadataError(
                f"joint must be one of: {', '.join(VALID_JOINTS)}."
            )

        start = _finite_number(
            self.start_deg,
            "start_deg",
        )
        end = _finite_number(
            self.end_deg,
            "end_deg",
        )
        frequency = _finite_number(
            self.frequency_hz,
            "frequency_hz",
        )
        setpoint_rate = _finite_number(
            self.setpoint_rate_hz,
            "setpoint_rate_hz",
        )
        move_to_start = _finite_number(
            self.move_to_start_s,
            "move_to_start_s",
        )
        return_duration = _finite_number(
            self.return_duration_s,
            "return_duration_s",
        )
        neutral_hold = _finite_number(
            self.neutral_hold_s,
            "neutral_hold_s",
        )

        if start == end:
            raise MetadataError(
                "start_deg and end_deg must be different."
            )

        if frequency <= 0:
            raise MetadataError(
                "frequency_hz must be greater than zero."
            )

        if (
            not isinstance(self.cycles, int)
            or isinstance(self.cycles, bool)
            or self.cycles <= 0
        ):
            raise MetadataError(
                "cycles must be a positive integer."
            )

        if setpoint_rate <= 0:
            raise MetadataError(
                "setpoint_rate_hz must be greater than zero."
            )

        if move_to_start <= 0:
            raise MetadataError(
                "move_to_start_s must be greater than zero."
            )

        if return_duration <= 0:
            raise MetadataError(
                "return_duration_s must be greater than zero."
            )

        if neutral_hold < 0:
            raise MetadataError(
                "neutral_hold_s cannot be negative."
            )

        object.__setattr__(self, "start_deg", start)
        object.__setattr__(self, "end_deg", end)
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(
            self,
            "setpoint_rate_hz",
            setpoint_rate,
        )
        object.__setattr__(
            self,
            "move_to_start_s",
            move_to_start,
        )
        object.__setattr__(
            self,
            "return_duration_s",
            return_duration,
        )
        object.__setattr__(
            self,
            "neutral_hold_s",
            neutral_hold,
        )

    @property
    def duration_s(self) -> float:
        return self.cycles / self.frequency_hz


@dataclass(frozen=True)
class LabVIEWMetadata:
    """LabVIEW UDP communication configuration."""

    host: str
    port: int
    enabled: bool = True

    def __post_init__(self) -> None:
        host = _non_empty_text(
            self.host,
            "labview.host",
        )

        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65535
        ):
            raise MetadataError(
                "labview.port must be between 1 and 65535."
            )

        if not isinstance(self.enabled, bool):
            raise MetadataError(
                "labview.enabled must be a boolean."
            )

        object.__setattr__(self, "host", host)


@dataclass(frozen=True)
class SessionMetadata:
    """Complete immutable configuration snapshot for one session."""

    session_id: str
    created_at_utc: str
    participant: ParticipantMetadata
    biosignalsplux: BiosignalspluxMetadata
    trajectory: TrajectoryMetadata
    labview: LabVIEWMetadata
    schema_version: int = METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        session_id = _non_empty_text(
            self.session_id,
            "session_id",
        )

        try:
            timestamp = datetime.fromisoformat(
                self.created_at_utc.replace("Z", "+00:00")
            )
        except (AttributeError, ValueError) as exc:
            raise MetadataError(
                "created_at_utc must be a valid ISO 8601 timestamp."
            ) from exc

        if timestamp.utcoffset() is None:
            raise MetadataError(
                "created_at_utc must include timezone information."
            )

        if self.schema_version != METADATA_SCHEMA_VERSION:
            raise MetadataError(
                f"schema_version must be {METADATA_SCHEMA_VERSION}."
            )

        object.__setattr__(self, "session_id", session_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a JSON-compatible dictionary."""

        data = asdict(self)
        data["biosignalsplux"]["channels"] = list(
            self.biosignalsplux.channels
        )
        data["trajectory"]["duration_s"] = (
            self.trajectory.duration_s
        )
        return data


def write_metadata_json(
    path: str | Path,
    metadata: SessionMetadata,
) -> Path:
    """Write one metadata snapshot atomically enough for local use."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            metadata.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(output_path)
    return output_path
"""Session identifiers and output-path management."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SessionError(ValueError):
    """Raised when a session identifier or output path is invalid."""


@dataclass(frozen=True)
class SessionPaths:
    """Paths generated for one acquisition session."""

    session_id: str
    directory: Path
    emg_csv: Path
    trajectories_csv: Path
    events_csv: Path
    metadata_json: Path
    sync_report_json: Path
    aligned_csv: Path


def generate_session_id(prefix: str = "session") -> str:
    """Generate a filesystem-safe UTC session identifier."""

    validate_session_id(prefix)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{timestamp}"


def validate_session_id(session_id: str) -> None:
    """Validate that a session identifier is safe for use as a folder name."""

    if not isinstance(session_id, str) or not session_id:
        raise SessionError("session_id must be a non-empty string.")

    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise SessionError(
            "session_id may contain only letters, numbers, dots, "
            "underscores, and hyphens."
        )


def create_session_paths(
    output_root: str | Path,
    session_id: str,
) -> SessionPaths:
    """Create the session directory and return all planned file paths."""

    validate_session_id(session_id)

    root = Path(output_root)
    directory = root / session_id
    directory.mkdir(parents=True, exist_ok=False)

    return SessionPaths(
        session_id=session_id,
        directory=directory,
        emg_csv=directory / f"emg_{session_id}.csv",
        trajectories_csv=directory / f"trajectories_{session_id}.csv",
        events_csv=directory / f"events_{session_id}.csv",
        metadata_json=directory / f"metadata_{session_id}.json",
        sync_report_json=directory / f"sync_report_{session_id}.json",
        aligned_csv=directory / f"aligned_{session_id}.csv",
    )
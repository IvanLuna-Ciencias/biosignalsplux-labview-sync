"""Tests for session identifiers and output paths."""

from pathlib import Path

import pytest

from biosignalsplux_labview_sync.session import (
    SessionError,
    create_session_paths,
    generate_session_id,
)


def test_generate_session_id() -> None:
    session_id = generate_session_id("passive_elbow")

    assert session_id.startswith("passive_elbow_")
    assert session_id.endswith("Z")
    assert " " not in session_id


def test_create_session_paths(tmp_path: Path) -> None:
    session_id = "passive_elbow_001"
    paths = create_session_paths(tmp_path, session_id)

    assert paths.directory.is_dir()
    assert paths.session_id == session_id
    assert paths.emg_csv.name == f"emg_{session_id}.csv"
    assert paths.trajectories_csv.name == (
        f"trajectories_{session_id}.csv"
    )
    assert paths.events_csv.name == f"events_{session_id}.csv"
    assert paths.metadata_json.name == f"metadata_{session_id}.json"
    assert paths.sync_report_json.name == (
        f"sync_report_{session_id}.json"
    )
    assert paths.aligned_csv.name == f"aligned_{session_id}.csv"


def test_existing_session_directory_is_rejected(tmp_path: Path) -> None:
    session_id = "duplicate_session"

    create_session_paths(tmp_path, session_id)

    with pytest.raises(FileExistsError):
        create_session_paths(tmp_path, session_id)


@pytest.mark.parametrize(
    "invalid_session_id",
    [
        "",
        "../outside",
        "session with spaces",
        "session/inside",
        "session:001",
    ],
)
def test_invalid_session_id_is_rejected(
    tmp_path: Path,
    invalid_session_id: str,
) -> None:
    with pytest.raises(SessionError):
        create_session_paths(tmp_path, invalid_session_id)
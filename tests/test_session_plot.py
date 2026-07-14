"""Tests for synchronized-session plotting."""

from __future__ import annotations

import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg", force=True)

from biosignalsplux_labview_sync.session_plot import (
    create_session_plots,
    find_latest_session_directory,
)


def create_test_session(directory) -> None:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    emg_path = directory / "emg_test.csv"

    with emg_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "sample_index",
                "device_sequence",
                "time_device_s",
                "time_monotonic_s",
                "timestamp_host",
                "emg_0",
                "emg_1",
            ]
        )

        for index in range(1000):
            writer.writerow(
                [
                    index,
                    index,
                    index / 1000.0,
                    index / 1000.0,
                    "2026-07-14T12:00:00+00:00",
                    index % 100,
                    -(index % 100),
                ]
            )

    trajectory_path = (
        directory / "trajectories_test.csv"
    )

    with trajectory_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "sample_index",
                "session_time_s",
                "trajectory_time_s",
                "state",
                "active_joint",
                "q_shoulder_ref_rad",
                "q_elbow_ref_rad",
                "q_rotation_ref_rad",
                "udp_sequence",
            ]
        )

        for index in range(50):
            writer.writerow(
                [
                    index,
                    index / 50.0,
                    index / 50.0,
                    "RUNNING",
                    "elbow",
                    0.0,
                    index / 100.0,
                    0.0,
                    index,
                ]
            )

    events_path = directory / "events_test.csv"

    with events_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "session_time_s",
                "event",
                "details_json",
            ]
        )

        writer.writerow(
            [
                0.1,
                "TRAJECTORY_STARTED",
                "{}",
            ]
        )

        writer.writerow(
            [
                0.9,
                "TRAJECTORY_DONE",
                "{}",
            ]
        )


def test_create_session_plots(tmp_path) -> None:
    session_directory = (
        tmp_path
        / "P-TEST_elbow_20260714_120000"
    )

    create_test_session(
        session_directory
    )

    summary = create_session_plots(
        session_directory=session_directory,
        max_emg_points=100,
        show=False,
    )

    assert summary["emg"]["rows"] == 1000
    assert summary["emg"]["points_displayed"] == 100
    assert summary["trajectory"]["rows"] == 50
    assert summary["events"]["count"] == 2

    assert (
        session_directory
        / "plots"
        / "session_emg.png"
    ).exists()

    assert (
        session_directory
        / "plots"
        / "session_references.png"
    ).exists()

    assert (
        session_directory
        / "plots"
        / "session_synchronized_comparison.png"
    ).exists()

    summary_path = (
        session_directory
        / "plots"
        / "session_plot_summary.json"
    )

    assert summary_path.exists()

    saved = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved["emg"]["channels"] == [
        "emg_0",
        "emg_1",
    ]

    assert "robot_trajectory" not in saved


def test_find_latest_session_directory(
    tmp_path,
) -> None:
    older = tmp_path / "older_session"
    newer = tmp_path / "newer_session"

    create_test_session(older)
    time.sleep(0.02)
    create_test_session(newer)

    os.utime(
        newer,
        None,
    )

    assert (
        find_latest_session_directory(
            tmp_path
        )
        == newer
    )

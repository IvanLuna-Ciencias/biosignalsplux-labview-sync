"""Tests for raw sEMG CSV validation."""

import csv
from pathlib import Path

from biosignalsplux_labview_sync.validation import validate_emg_csv


HEADER = [
    "sample_index",
    "device_sequence",
    "time_device_s",
    "time_monotonic_s",
    "timestamp_host",
    "emg_0",
    "emg_1",
]


def write_rows(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        writer.writerows(rows)


def test_valid_emg_csv(tmp_path: Path) -> None:
    path = tmp_path / "valid.csv"
    write_rows(
        path,
        [
            [0, 10, 0.000, 1.000, "2026-01-01T00:00:00+00:00", 1, 2],
            [1, 11, 0.001, 1.001, "2026-01-01T00:00:00+00:00", 3, 4],
            [2, 12, 0.002, 1.002, "2026-01-01T00:00:00+00:00", 5, 6],
        ],
    )

    report = validate_emg_csv(path)

    assert report.is_valid
    assert report.rows_checked == 3
    assert report.errors == ()
    assert report.warnings == ()


def test_sequence_gap_is_reported_as_warning(tmp_path: Path) -> None:
    path = tmp_path / "gap.csv"
    write_rows(
        path,
        [
            [0, 10, 0.000, 1.000, "2026-01-01T00:00:00+00:00", 1, 2],
            [1, 13, 0.003, 1.003, "2026-01-01T00:00:00+00:00", 3, 4],
        ],
    )

    report = validate_emg_csv(path)

    assert report.is_valid
    assert report.sequence_gap_events == 1
    assert report.missing_sequence_count == 2
    assert len(report.warnings) == 1


def test_non_increasing_sequence_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "invalid_sequence.csv"
    write_rows(
        path,
        [
            [0, 10, 0.000, 1.000, "2026-01-01T00:00:00+00:00", 1, 2],
            [1, 10, 0.001, 1.001, "2026-01-01T00:00:00+00:00", 3, 4],
        ],
    )

    report = validate_emg_csv(path)

    assert not report.is_valid
    assert report.non_increasing_sequence_count == 1
    assert report.errors


def test_missing_required_column_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "missing_column.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_index",
                "device_sequence",
                "time_device_s",
                "timestamp_host",
                "emg_0",
            ]
        )
        writer.writerow(
            [0, 0, 0.0, "2026-01-01T00:00:00+00:00", 1]
        )

    report = validate_emg_csv(path)

    assert not report.is_valid
    assert "time_monotonic_s" in report.errors[0]
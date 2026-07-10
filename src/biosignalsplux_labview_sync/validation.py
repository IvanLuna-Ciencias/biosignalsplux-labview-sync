"""Validation utilities for raw acquisition files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


_REQUIRED_COLUMNS = (
    "sample_index",
    "device_sequence",
    "time_device_s",
    "time_monotonic_s",
    "timestamp_host",
)


@dataclass(frozen=True)
class EMGValidationReport:
    """Integrity report for one raw sEMG CSV file."""

    path: Path
    is_valid: bool
    rows_checked: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    sequence_gap_events: int
    missing_sequence_count: int
    non_increasing_sequence_count: int
    sample_index_error_count: int
    device_time_error_count: int
    monotonic_time_error_count: int


def validate_emg_csv(path: str | Path) -> EMGValidationReport:
    """Validate the structure, timing, and sequence integrity of an EMG CSV."""

    csv_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []

    rows_checked = 0
    sequence_gap_events = 0
    missing_sequence_count = 0
    non_increasing_sequence_count = 0
    sample_index_error_count = 0
    device_time_error_count = 0
    monotonic_time_error_count = 0

    if not csv_path.is_file():
        errors.append(f"File not found: {csv_path}")
        return EMGValidationReport(
            path=csv_path,
            is_valid=False,
            rows_checked=0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            sequence_gap_events=0,
            missing_sequence_count=0,
            non_increasing_sequence_count=0,
            sample_index_error_count=0,
            device_time_error_count=0,
            monotonic_time_error_count=0,
        )

    previous_sequence: int | None = None
    previous_device_time: float | None = None
    previous_monotonic_time: float | None = None

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        missing_columns = [
            column
            for column in _REQUIRED_COLUMNS
            if column not in fieldnames
        ]

        if missing_columns:
            errors.append(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        channel_columns = [
            column
            for column in fieldnames
            if column not in _REQUIRED_COLUMNS
        ]

        if not channel_columns:
            errors.append("No signal channel columns were found.")

        if errors:
            return EMGValidationReport(
                path=csv_path,
                is_valid=False,
                rows_checked=0,
                errors=tuple(errors),
                warnings=tuple(warnings),
                sequence_gap_events=0,
                missing_sequence_count=0,
                non_increasing_sequence_count=0,
                sample_index_error_count=0,
                device_time_error_count=0,
                monotonic_time_error_count=0,
            )

        for line_number, row in enumerate(reader, start=2):
            rows_checked += 1

            try:
                sample_index = int(row["sample_index"])
                device_sequence = int(row["device_sequence"])
                device_time = float(row["time_device_s"])
                monotonic_time = float(row["time_monotonic_s"])

                for channel in channel_columns:
                    float(row[channel])

            except (TypeError, ValueError):
                errors.append(
                    f"Line {line_number} contains invalid numeric data."
                )
                continue

            expected_sample_index = rows_checked - 1
            if sample_index != expected_sample_index:
                sample_index_error_count += 1

            if previous_sequence is not None:
                sequence_delta = device_sequence - previous_sequence

                if sequence_delta > 1:
                    sequence_gap_events += 1
                    missing_sequence_count += sequence_delta - 1
                elif sequence_delta <= 0:
                    non_increasing_sequence_count += 1

            if (
                previous_device_time is not None
                and device_time <= previous_device_time
            ):
                device_time_error_count += 1

            if (
                previous_monotonic_time is not None
                and monotonic_time < previous_monotonic_time
            ):
                monotonic_time_error_count += 1

            previous_sequence = device_sequence
            previous_device_time = device_time
            previous_monotonic_time = monotonic_time

    if rows_checked == 0:
        errors.append("The CSV does not contain data rows.")

    if sample_index_error_count:
        errors.append(
            f"Non-consecutive sample indexes: "
            f"{sample_index_error_count}."
        )

    if non_increasing_sequence_count:
        errors.append(
            f"Non-increasing device sequences: "
            f"{non_increasing_sequence_count}."
        )

    if device_time_error_count:
        errors.append(
            f"Non-increasing device timestamps: "
            f"{device_time_error_count}."
        )

    if monotonic_time_error_count:
        errors.append(
            f"Decreasing monotonic timestamps: "
            f"{monotonic_time_error_count}."
        )

    if sequence_gap_events:
        warnings.append(
            f"Detected {sequence_gap_events} sequence gap event(s), "
            f"representing {missing_sequence_count} missing sample(s)."
        )

    return EMGValidationReport(
        path=csv_path,
        is_valid=not errors,
        rows_checked=rows_checked,
        errors=tuple(errors),
        warnings=tuple(warnings),
        sequence_gap_events=sequence_gap_events,
        missing_sequence_count=missing_sequence_count,
        non_increasing_sequence_count=non_increasing_sequence_count,
        sample_index_error_count=sample_index_error_count,
        device_time_error_count=device_time_error_count,
        monotonic_time_error_count=monotonic_time_error_count,
    )
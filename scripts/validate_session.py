#!/usr/bin/env python
"""Validate the raw sEMG file produced during an acquisition session."""

from __future__ import annotations

import argparse
from pathlib import Path

from biosignalsplux_labview_sync.validation import validate_emg_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate sample indexes, device sequences, timestamps, "
            "and numeric channel values in a raw sEMG CSV."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Session directory or raw emg_*.csv file.",
    )
    return parser.parse_args()


def resolve_emg_csv(path: Path) -> Path:
    """Resolve one raw EMG CSV from a file or session directory."""

    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    matches = sorted(path.glob("emg_*.csv"))

    if not matches:
        raise FileNotFoundError(
            f"No emg_*.csv file was found in session directory: {path}"
        )

    if len(matches) > 1:
        names = ", ".join(item.name for item in matches)
        raise RuntimeError(
            "More than one raw EMG CSV was found. "
            f"Pass the desired file explicitly: {names}"
        )

    return matches[0]


def print_report(csv_path: Path) -> bool:
    """Validate one CSV and print a readable integrity report."""

    report = validate_emg_csv(csv_path)

    print("EMG validation report")
    print("---------------------")
    print(f"File: {report.path}")
    print(f"Valid: {report.is_valid}")
    print(f"Rows checked: {report.rows_checked}")
    print(f"Sequence gap events: {report.sequence_gap_events}")
    print(f"Missing sequence count: {report.missing_sequence_count}")
    print(
        "Non-increasing sequences: "
        f"{report.non_increasing_sequence_count}"
    )
    print(
        "Sample index errors: "
        f"{report.sample_index_error_count}"
    )
    print(
        "Device time errors: "
        f"{report.device_time_error_count}"
    )
    print(
        "Monotonic time errors: "
        f"{report.monotonic_time_error_count}"
    )

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"- {warning}")

    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"- {error}")

    if report.is_valid:
        print("\nResult: raw EMG file passed integrity validation.")
    else:
        print("\nResult: raw EMG file failed integrity validation.")

    return report.is_valid


def main() -> int:
    args = parse_args()

    try:
        csv_path = resolve_emg_csv(args.path)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Validation could not start: {exc}")
        return 2

    is_valid = print_report(csv_path)
    return 0 if is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
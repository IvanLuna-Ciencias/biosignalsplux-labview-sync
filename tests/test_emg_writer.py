"""Tests for incremental raw sEMG storage."""

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from biosignalsplux_labview_sync.emg_writer import (
    EMGCSVWriter,
    EMGWriterError,
)


def test_writer_creates_expected_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "emg_test.csv"

    with EMGCSVWriter(
        output_path,
        channel_names=["emg_0", "emg_1"],
        sampling_rate_hz=1000,
        flush_interval_samples=1,
    ) as writer:
        writer.append_sample(
            device_sequence=100,
            time_monotonic_s=5.25,
            timestamp_host=datetime.now(timezone.utc),
            channel_values=[120, 240],
        )
        writer.append_sample(
            device_sequence=101,
            time_monotonic_s=5.251,
            timestamp_host=datetime.now(timezone.utc),
            channel_values=[121, 241],
        )

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert rows[0] == [
        "sample_index",
        "device_sequence",
        "time_device_s",
        "time_monotonic_s",
        "timestamp_host",
        "emg_0",
        "emg_1",
    ]
    assert rows[1][0:4] == [
        "0",
        "100",
        "0.000000000",
        "5.250000000",
    ]
    assert rows[1][-2:] == ["120", "240"]
    assert rows[2][0:3] == ["1", "101", "0.001000000"]


def test_writer_detects_sequence_gaps(tmp_path: Path) -> None:
    output_path = tmp_path / "emg_gap.csv"

    with EMGCSVWriter(
        output_path,
        channel_names=["emg_0", "emg_1"],
        sampling_rate_hz=1000,
    ) as writer:
        for sequence in (10, 11, 14):
            writer.append_sample(
                device_sequence=sequence,
                time_monotonic_s=sequence / 1000,
                timestamp_host=datetime.now(timezone.utc),
                channel_values=[1, 2],
            )

        stats = writer.stats

    assert stats.samples_written == 3
    assert stats.sequence_gap_events == 1
    assert stats.missing_sequence_count == 2
    assert stats.non_increasing_sequence_count == 0


def test_writer_rejects_wrong_channel_count(tmp_path: Path) -> None:
    output_path = tmp_path / "emg_invalid.csv"

    with EMGCSVWriter(
        output_path,
        channel_names=["emg_0", "emg_1"],
        sampling_rate_hz=1000,
    ) as writer:
        with pytest.raises(
            EMGWriterError,
            match="number of channel values",
        ):
            writer.append_sample(
                device_sequence=0,
                time_monotonic_s=0.0,
                timestamp_host=datetime.now(timezone.utc),
                channel_values=[1],
            )


def test_writer_requires_timezone_aware_timestamp(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "emg_timestamp.csv"

    with EMGCSVWriter(
        output_path,
        channel_names=["emg_0", "emg_1"],
        sampling_rate_hz=1000,
    ) as writer:
        with pytest.raises(
            EMGWriterError,
            match="timezone information",
        ):
            writer.append_sample(
                device_sequence=0,
                time_monotonic_s=0.0,
                timestamp_host=datetime.now(),
                channel_values=[1, 2],
            )
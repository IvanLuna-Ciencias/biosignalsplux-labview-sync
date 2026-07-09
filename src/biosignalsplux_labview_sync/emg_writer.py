"""Incremental storage for raw Biosignalsplux sEMG samples."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, TextIO


class EMGWriterError(ValueError):
    """Raised when an EMG sample or writer setting is invalid."""


@dataclass(frozen=True)
class EMGWriteStats:
    """Summary of samples written and sequence anomalies."""

    samples_written: int
    sequence_gap_events: int
    missing_sequence_count: int
    non_increasing_sequence_count: int


class EMGCSVWriter:
    """Write raw sEMG samples incrementally to a CSV file."""

    def __init__(
        self,
        path: str | Path,
        channel_names: Iterable[str],
        sampling_rate_hz: int,
        flush_interval_samples: int = 1000,
    ) -> None:
        self.path = Path(path)
        self.channel_names = tuple(channel_names)
        self.sampling_rate_hz = sampling_rate_hz
        self.flush_interval_samples = flush_interval_samples

        if not self.channel_names:
            raise EMGWriterError("At least one channel name is required.")

        if not all(
            isinstance(name, str) and name.strip()
            for name in self.channel_names
        ):
            raise EMGWriterError(
                "Every channel name must be a non-empty string."
            )

        if (
            not isinstance(sampling_rate_hz, int)
            or isinstance(sampling_rate_hz, bool)
            or sampling_rate_hz <= 0
        ):
            raise EMGWriterError(
                "sampling_rate_hz must be a positive integer."
            )

        if (
            not isinstance(flush_interval_samples, int)
            or isinstance(flush_interval_samples, bool)
            or flush_interval_samples <= 0
        ):
            raise EMGWriterError(
                "flush_interval_samples must be a positive integer."
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._file: TextIO = self.path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self._writer = csv.writer(self._file)

        self._writer.writerow(
            [
                "sample_index",
                "device_sequence",
                "time_device_s",
                "time_monotonic_s",
                "timestamp_host",
                *self.channel_names,
            ]
        )

        self._samples_written = 0
        self._sequence_gap_events = 0
        self._missing_sequence_count = 0
        self._non_increasing_sequence_count = 0
        self._first_sequence: int | None = None
        self._last_sequence: int | None = None
        self._closed = False

    def append_sample(
        self,
        device_sequence: int,
        time_monotonic_s: float,
        timestamp_host: datetime,
        channel_values: Iterable[int | float],
    ) -> None:
        """Append one raw sample and update sequence diagnostics."""

        if self._closed:
            raise EMGWriterError("Cannot write to a closed EMGCSVWriter.")

        if (
            not isinstance(device_sequence, int)
            or isinstance(device_sequence, bool)
            or device_sequence < 0
        ):
            raise EMGWriterError(
                "device_sequence must be a non-negative integer."
            )

        if (
            not isinstance(time_monotonic_s, (int, float))
            or isinstance(time_monotonic_s, bool)
            or time_monotonic_s < 0
        ):
            raise EMGWriterError(
                "time_monotonic_s must be a non-negative number."
            )

        if not isinstance(timestamp_host, datetime):
            raise EMGWriterError(
                "timestamp_host must be a datetime instance."
            )

        if timestamp_host.utcoffset() is None:
            raise EMGWriterError(
                "timestamp_host must include timezone information."
            )

        values = tuple(channel_values)

        if len(values) != len(self.channel_names):
            raise EMGWriterError(
                "The number of channel values must match channel_names."
            )

        if self._first_sequence is None:
            self._first_sequence = device_sequence

        if self._last_sequence is not None:
            sequence_delta = device_sequence - self._last_sequence

            if sequence_delta > 1:
                self._sequence_gap_events += 1
                self._missing_sequence_count += sequence_delta - 1
            elif sequence_delta <= 0:
                self._non_increasing_sequence_count += 1

        time_device_s = (
            device_sequence - self._first_sequence
        ) / self.sampling_rate_hz

        self._writer.writerow(
            [
                self._samples_written,
                device_sequence,
                f"{time_device_s:.9f}",
                f"{float(time_monotonic_s):.9f}",
                timestamp_host.isoformat(),
                *values,
            ]
        )

        self._samples_written += 1
        self._last_sequence = device_sequence

        if self._samples_written % self.flush_interval_samples == 0:
            self._file.flush()

    @property
    def stats(self) -> EMGWriteStats:
        """Return the current writer statistics."""

        return EMGWriteStats(
            samples_written=self._samples_written,
            sequence_gap_events=self._sequence_gap_events,
            missing_sequence_count=self._missing_sequence_count,
            non_increasing_sequence_count=(
                self._non_increasing_sequence_count
            ),
        )

    def flush(self) -> None:
        """Flush buffered CSV data to the operating system."""

        if not self._closed:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file safely."""

        if self._closed:
            return

        self._file.flush()
        self._file.close()
        self._closed = True

    def __enter__(self) -> "EMGCSVWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
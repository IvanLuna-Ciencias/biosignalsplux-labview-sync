"""Incremental CSV storage for ATI force/torque acquisition."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np


class ForceWriterError(ValueError):
    """Raised when force data or writer settings are invalid."""


@dataclass(frozen=True)
class ForceWriteStats:
    """Summary of force samples written to disk."""

    samples_written: int


class ForceCSVWriter:
    """Write raw voltages and calibrated six-axis values incrementally."""

    HEADER = (
        "sample_index",
        "device_sample_index",
        "time_device_s",
        "time_monotonic_s",
        "timestamp_host",
        "ai0_v",
        "ai1_v",
        "ai2_v",
        "ai3_v",
        "ai4_v",
        "ai5_v",
        "fx_n",
        "fy_n",
        "fz_n",
        "mx_nm",
        "my_nm",
        "mz_nm",
        "force_norm_n",
        "torque_norm_nm",
    )

    def __init__(
        self,
        *,
        path: str | Path,
        sampling_rate_hz: int,
        flush_interval_samples: int = 1000,
    ) -> None:
        if (
            not isinstance(sampling_rate_hz, int)
            or isinstance(sampling_rate_hz, bool)
            or sampling_rate_hz <= 0
        ):
            raise ForceWriterError("sampling_rate_hz must be positive.")
        if (
            not isinstance(flush_interval_samples, int)
            or isinstance(flush_interval_samples, bool)
            or flush_interval_samples <= 0
        ):
            raise ForceWriterError("flush_interval_samples must be positive.")

        self.path = Path(path)
        self.sampling_rate_hz = sampling_rate_hz
        self.flush_interval_samples = flush_interval_samples
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._samples_written = 0
        self._closed = False

    def append_sample(
        self,
        *,
        device_sample_index: int,
        time_monotonic_s: float,
        timestamp_host: datetime,
        raw_volts: Iterable[int | float],
        force_torque: Iterable[int | float],
    ) -> None:
        if self._closed:
            raise ForceWriterError("Cannot write to a closed ForceCSVWriter.")
        if (
            not isinstance(device_sample_index, int)
            or isinstance(device_sample_index, bool)
            or device_sample_index < 0
        ):
            raise ForceWriterError("device_sample_index must be non-negative.")
        if (
            not isinstance(time_monotonic_s, (int, float))
            or isinstance(time_monotonic_s, bool)
            or not math.isfinite(float(time_monotonic_s))
            or float(time_monotonic_s) < 0
        ):
            raise ForceWriterError("time_monotonic_s must be non-negative.")
        if not isinstance(timestamp_host, datetime) or timestamp_host.utcoffset() is None:
            raise ForceWriterError("timestamp_host must be timezone-aware.")

        raw = np.asarray(tuple(raw_volts), dtype=np.float64)
        ft = np.asarray(tuple(force_torque), dtype=np.float64)
        if raw.shape != (6,) or ft.shape != (6,):
            raise ForceWriterError("raw_volts and force_torque must each contain six values.")
        if not np.isfinite(raw).all() or not np.isfinite(ft).all():
            raise ForceWriterError("Force sample contains non-finite values.")

        force_norm = float(np.linalg.norm(ft[:3]))
        torque_norm = float(np.linalg.norm(ft[3:]))
        time_device_s = device_sample_index / self.sampling_rate_hz

        self._writer.writerow(
            [
                self._samples_written,
                device_sample_index,
                f"{time_device_s:.9f}",
                f"{float(time_monotonic_s):.9f}",
                timestamp_host.isoformat(),
                *[f"{value:.12g}" for value in raw],
                *[f"{value:.12g}" for value in ft],
                f"{force_norm:.12g}",
                f"{torque_norm:.12g}",
            ]
        )
        self._samples_written += 1
        if self._samples_written % self.flush_interval_samples == 0:
            self._file.flush()

    @property
    def stats(self) -> ForceWriteStats:
        return ForceWriteStats(samples_written=self._samples_written)

    def flush(self) -> None:
        if not self._closed:
            self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._file.flush()
        self._file.close()
        self._closed = True

    def __enter__(self) -> "ForceCSVWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

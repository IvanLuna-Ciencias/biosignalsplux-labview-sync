"""Tests for ATI incremental CSV storage."""

from __future__ import annotations

import csv
from datetime import datetime, timezone

from biosignalsplux_labview_sync.force_writer import ForceCSVWriter


def test_force_writer_stores_raw_and_calibrated_values(tmp_path) -> None:
    path = tmp_path / "force.csv"
    with ForceCSVWriter(path=path, sampling_rate_hz=1000) as writer:
        writer.append_sample(
            device_sample_index=10,
            time_monotonic_s=1.25,
            timestamp_host=datetime.now(timezone.utc),
            raw_volts=[1, 2, 3, 4, 5, 6],
            force_torque=[3, 4, 0, 0, 0, 2],
        )
        assert writer.stats.samples_written == 1

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert float(rows[0]["force_norm_n"]) == 5.0
    assert float(rows[0]["torque_norm_nm"]) == 2.0
    assert float(rows[0]["time_device_s"]) == 0.01

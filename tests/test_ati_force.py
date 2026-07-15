"""Tests for ATI calibration and numerical conversion."""

from __future__ import annotations

import numpy as np

from biosignalsplux_labview_sync.ati_force import (
    calibrate_force_block,
    load_ati_calibration,
    normalize_daq_block,
)


def test_load_ati_calibration(tmp_path) -> None:
    path = tmp_path / "sensor.cal"
    rows = "\n".join(
        f'<Axis Name="A{index}" values="1 0 0 0 0 {index}" />'
        for index in range(6)
    )
    path.write_text(
        '<FTSensor Serial="TEST" BodyStyle="Nano25">'
        '<Calibration ForceUnits="N" TorqueUnits="N-m">'
        f"{rows}"
        "</Calibration></FTSensor>",
        encoding="utf-8",
    )
    calibration = load_ati_calibration(path)
    assert calibration.serial == "TEST"
    assert calibration.matrix.shape == (6, 6)


def test_normalize_multi_channel_block() -> None:
    raw = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
    block = normalize_daq_block(raw)
    assert block.shape == (3, 6)
    assert block[0, 0] == 1.0
    assert block[2, 5] == 18.0


def test_calibrate_force_block_subtracts_bias() -> None:
    raw = np.asarray([[2, 3, 4, 5, 6, 7]], dtype=float)
    bias = np.asarray([1, 1, 1, 1, 1, 1], dtype=float)
    calibrated = calibrate_force_block(
        raw,
        bias_volts=bias,
        calibration_matrix=np.eye(6),
    )
    assert np.allclose(calibrated[0], [1, 2, 3, 4, 5, 6])

"""Tests for live sEMG plot-data preparation."""

import numpy as np
import pytest

from biosignalsplux_labview_sync.live_buffer import (
    EMGBufferSnapshot,
)
from biosignalsplux_labview_sync.live_window import (
    EMGLiveWindowError,
    snapshot_to_plot_arrays,
)


def test_snapshot_is_converted_to_relative_plot_time() -> None:
    snapshot = EMGBufferSnapshot(
        time_monotonic_s=(10.000, 10.001, 10.002),
        channels=(
            (100.0, 101.0, 102.0),
            (200.0, 201.0, 202.0),
        ),
        total_samples_received=3,
    )

    time_values, channels = snapshot_to_plot_arrays(snapshot)

    np.testing.assert_allclose(
        time_values,
        [-0.002, -0.001, 0.0],
    )
    np.testing.assert_allclose(
        channels[0],
        [100.0, 101.0, 102.0],
    )
    np.testing.assert_allclose(
        channels[1],
        [200.0, 201.0, 202.0],
    )


def test_empty_snapshot_produces_empty_arrays() -> None:
    snapshot = EMGBufferSnapshot(
        time_monotonic_s=(),
        channels=((), ()),
        total_samples_received=0,
    )

    time_values, channels = snapshot_to_plot_arrays(snapshot)

    assert time_values.size == 0
    assert channels[0].size == 0
    assert channels[1].size == 0


def test_misaligned_snapshot_is_rejected() -> None:
    snapshot = EMGBufferSnapshot(
        time_monotonic_s=(0.0, 0.001),
        channels=(
            (1.0,),
            (2.0, 3.0),
        ),
        total_samples_received=2,
    )

    with pytest.raises(
        EMGLiveWindowError,
        match="same number of samples",
    ):
        snapshot_to_plot_arrays(snapshot)
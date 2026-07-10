"""Tests for the thread-safe live sEMG circular buffer."""

import pytest

from biosignalsplux_labview_sync.live_buffer import (
    EMGLiveBuffer,
    EMGLiveBufferError,
)


def test_buffer_returns_aligned_snapshot() -> None:
    buffer = EMGLiveBuffer(
        channel_count=2,
        capacity_samples=5,
    )

    buffer.append_sample(0.000, [100, 200])
    buffer.append_sample(0.001, [101, 201])
    buffer.append_sample(0.002, [102, 202])

    snapshot = buffer.snapshot()

    assert snapshot.time_monotonic_s == (0.0, 0.001, 0.002)
    assert snapshot.channels == (
        (100.0, 101.0, 102.0),
        (200.0, 201.0, 202.0),
    )
    assert snapshot.buffered_samples == 3
    assert snapshot.total_samples_received == 3


def test_buffer_discards_oldest_samples_at_capacity() -> None:
    buffer = EMGLiveBuffer(
        channel_count=2,
        capacity_samples=3,
    )

    for index in range(5):
        buffer.append_sample(
            time_monotonic_s=index / 1000,
            channel_values=[index, index + 10],
        )

    snapshot = buffer.snapshot()

    assert snapshot.time_monotonic_s == (0.002, 0.003, 0.004)
    assert snapshot.channels[0] == (2.0, 3.0, 4.0)
    assert snapshot.channels[1] == (12.0, 13.0, 14.0)
    assert snapshot.buffered_samples == 3
    assert snapshot.total_samples_received == 5


def test_clear_preserves_lifetime_sample_count() -> None:
    buffer = EMGLiveBuffer(
        channel_count=2,
        capacity_samples=10,
    )

    buffer.append_sample(0.0, [1, 2])
    buffer.append_sample(0.001, [3, 4])
    buffer.clear()

    snapshot = buffer.snapshot()

    assert snapshot.buffered_samples == 0
    assert snapshot.channels == ((), ())
    assert snapshot.total_samples_received == 2


def test_buffer_rejects_wrong_channel_count() -> None:
    buffer = EMGLiveBuffer(
        channel_count=2,
        capacity_samples=10,
    )

    with pytest.raises(
        EMGLiveBufferError,
        match="number of channel values",
    ):
        buffer.append_sample(
            time_monotonic_s=0.0,
            channel_values=[1],
        )
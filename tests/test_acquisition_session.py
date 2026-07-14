"""Tests for the reusable Biosignalsplux acquisition session."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from biosignalsplux_labview_sync.acquisition_session import (
    AcquisitionSessionError,
    BiosignalsAcquisitionSession,
)
from biosignalsplux_labview_sync.live_buffer import EMGLiveBuffer


def make_fake_device_class(
    *,
    sample_count: int = 3,
    block_until_stop: bool = False,
):
    loop_entered = threading.Event()

    class FakeDevice:
        instances = []
        loop_entered_event = loop_entered

        def __init__(
            self,
            address,
            *,
            writer,
            session_origin,
            expected_channel_count,
            live_buffer,
        ):
            self.address = address
            self.writer = writer
            self.session_origin = session_origin
            self.expected_channel_count = expected_channel_count
            self.live_buffer = live_buffer

            self.samples_received = 0
            self.callback_error = None
            self.live_buffer_error_count = 0
            self.last_live_buffer_error = None

            self.started = False
            self.stop_called = False
            self.close_called = False
            self._stop_requested = threading.Event()

            type(self).instances.append(self)

        def start(self, frequency, channel_mask, resolution):
            self.started = True
            self.start_arguments = (
                frequency,
                channel_mask,
                resolution,
            )

        def loop(self):
            type(self).loop_entered_event.set()

            if block_until_stop:
                self._stop_requested.wait(timeout=2.0)
                return

            for sequence in range(sample_count):
                time_s = sequence / 1000.0
                values = [sequence, sequence + 100]

                self.writer.append_sample(
                    device_sequence=sequence,
                    time_monotonic_s=time_s,
                    timestamp_host=datetime.now(timezone.utc),
                    channel_values=values,
                )
                self.live_buffer.append_sample(
                    time_s,
                    values,
                )
                self.samples_received += 1

        def request_stop(self):
            self._stop_requested.set()

        def stop(self):
            self.stop_called = True

        def close(self):
            self.close_called = True

    return FakeDevice


def make_controller(
    tmp_path,
    device_class,
    *,
    session_origin=123.5,
):
    live_buffer = EMGLiveBuffer(
        channel_count=2,
        capacity_samples=5000,
    )

    controller = BiosignalsAcquisitionSession(
        plux=None,
        device_address="B00:07:80:4D:2F:0B",
        sampling_rate_hz=1000,
        resolution_bits=16,
        channel_mask=3,
        channel_names=("emg_0", "emg_1"),
        flush_interval_samples=1000,
        csv_path=tmp_path / "emg.csv",
        live_buffer=live_buffer,
        session_origin=session_origin,
        device_class=device_class,
    )

    return controller, live_buffer


def test_acquisition_writes_samples_and_closes_device(
    tmp_path,
) -> None:
    device_class = make_fake_device_class(
        sample_count=3
    )
    controller, live_buffer = make_controller(
        tmp_path,
        device_class,
    )

    controller.start()

    assert controller.join(timeout=2.0)
    assert controller.is_finished
    assert not controller.is_running

    results = controller.results
    device = device_class.instances[0]

    assert results.termination_reason == "device_loop_completed"
    assert results.session_origin == pytest.approx(123.5)
    assert results.samples_received == 3
    assert results.writer_stats is not None
    assert results.writer_stats.samples_written == 3
    assert results.runtime_error is None

    assert device.started
    assert device.start_arguments == (1000, 3, 16)
    assert device.stop_called
    assert device.close_called

    assert live_buffer.snapshot().total_samples_received == 3
    assert (tmp_path / "emg.csv").exists()


def test_stop_before_start_prevents_streaming(
    tmp_path,
) -> None:
    device_class = make_fake_device_class()
    controller, _ = make_controller(
        tmp_path,
        device_class,
    )

    controller.request_stop()
    controller.start()

    assert controller.join(timeout=2.0)

    device = device_class.instances[0]

    assert (
        controller.results.termination_reason
        == "stop_before_start"
    )
    assert not device.started
    assert not device.stop_called
    assert device.close_called


def test_explicit_stop_finishes_blocking_device_loop(
    tmp_path,
) -> None:
    device_class = make_fake_device_class(
        block_until_stop=True
    )
    controller, _ = make_controller(
        tmp_path,
        device_class,
    )

    controller.start()

    assert device_class.loop_entered_event.wait(timeout=1.0)

    controller.request_stop()

    assert controller.join(timeout=2.0)

    device = device_class.instances[0]

    assert (
        controller.results.termination_reason
        == "stop_requested"
    )
    assert device.stop_called
    assert device.close_called


def test_acquisition_session_cannot_start_twice(
    tmp_path,
) -> None:
    device_class = make_fake_device_class()
    controller, _ = make_controller(
        tmp_path,
        device_class,
    )

    controller.start()

    with pytest.raises(
        AcquisitionSessionError,
        match="already been started",
    ):
        controller.start()

    assert controller.join(timeout=2.0)
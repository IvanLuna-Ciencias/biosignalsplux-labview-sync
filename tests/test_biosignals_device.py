"""Hardware-free tests for the Biosignalsplux device adapter."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from biosignalsplux_labview_sync.biosignals_device import (
    BiosignalsDeviceError,
    create_biosignals_device_class,
)
from biosignalsplux_labview_sync.emg_writer import EMGCSVWriter


class FakeSignalsDev:
    """Minimal replacement for plux.SignalsDev in unit tests."""


class FakeMemoryDev:
    """Emulate the unusual PLUX SDK initialization call."""

    initialized_addresses: list[str] = []

    def __init__(address) -> None:
        FakeMemoryDev.initialized_addresses.append(address)


def make_fake_plux() -> SimpleNamespace:
    FakeMemoryDev.initialized_addresses.clear()

    return SimpleNamespace(
        SignalsDev=FakeSignalsDev,
        MemoryDev=FakeMemoryDev,
    )


def test_device_initializes_with_address_and_writes_frame(
    tmp_path: Path,
) -> None:
    plux = make_fake_plux()
    device_class = create_biosignals_device_class(plux)
    output_path = tmp_path / "emg.csv"

    monotonic_values = iter([10.125])
    timestamp = datetime(2026, 7, 10, tzinfo=timezone.utc)

    with EMGCSVWriter(
        output_path,
        channel_names=["emg_0", "emg_1"],
        sampling_rate_hz=1000,
    ) as writer:
        device = device_class(
            address="00:07:80:4D:2F:0B",
            writer=writer,
            session_origin=10.0,
            expected_channel_count=2,
            monotonic_clock=lambda: next(monotonic_values),
            wall_clock=lambda: timestamp,
        )

        should_stop = device.onRawFrame(100, [120, 240])

        assert not should_stop
        assert device.samples_received == 1
        assert device.callback_error is None
        assert writer.stats.samples_written == 1

    assert FakeMemoryDev.initialized_addresses == [
        "00:07:80:4D:2F:0B"
    ]


def test_stop_request_ends_loop_without_writing_another_frame(
    tmp_path: Path,
) -> None:
    plux = make_fake_plux()
    device_class = create_biosignals_device_class(plux)

    with EMGCSVWriter(
        tmp_path / "emg.csv",
        channel_names=["emg_0", "emg_1"],
        sampling_rate_hz=1000,
    ) as writer:
        device = device_class(
            address="00:07:80:4D:2F:0B",
            writer=writer,
            session_origin=0.0,
            expected_channel_count=2,
            monotonic_clock=lambda: 1.0,
        )

        device.request_stop()
        should_stop = device.onRawFrame(0, [1, 2])

        assert should_stop
        assert device.samples_received == 0
        assert writer.stats.samples_written == 0


def test_callback_error_requests_safe_stop(tmp_path: Path) -> None:
    plux = make_fake_plux()
    device_class = create_biosignals_device_class(plux)

    with EMGCSVWriter(
        tmp_path / "emg.csv",
        channel_names=["emg_0", "emg_1"],
        sampling_rate_hz=1000,
    ) as writer:
        device = device_class(
            address="00:07:80:4D:2F:0B",
            writer=writer,
            session_origin=0.0,
            expected_channel_count=2,
            monotonic_clock=lambda: 1.0,
        )

        should_stop = device.onRawFrame(0, [1])

        assert should_stop
        assert device.stop_requested
        assert device.samples_received == 0
        assert isinstance(device.callback_error, BiosignalsDeviceError)
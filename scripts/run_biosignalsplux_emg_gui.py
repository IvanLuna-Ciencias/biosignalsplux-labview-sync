#!/usr/bin/env python
"""Acquire Biosignalsplux sEMG continuously with a live GUI."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from biosignalsplux_labview_sync.biosignals_device import (
    BiosignalsDeviceError,
    create_biosignals_device_class,
)
from biosignalsplux_labview_sync.config import load_config
from biosignalsplux_labview_sync.emg_writer import EMGCSVWriter
from biosignalsplux_labview_sync.live_buffer import EMGLiveBuffer
from biosignalsplux_labview_sync.live_window import EMGLiveWindow
from biosignalsplux_labview_sync.plux_api import (
    PluxAPIError,
    load_plux_api,
)
from biosignalsplux_labview_sync.session import (
    create_session_paths,
    generate_session_id,
)
from biosignalsplux_labview_sync.validation import validate_emg_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire raw Biosignalsplux sEMG continuously while displaying "
            "the most recent samples in a live window."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/acquisition.local.json"),
        help="Path to the local acquisition JSON configuration.",
    )
    parser.add_argument(
        "--prefix",
        default="biosignals_gui",
        help="Prefix used to generate the session identifier.",
    )
    return parser.parse_args()


def acquisition_worker(
    *,
    plux,
    device_address: str,
    sampling_rate_hz: int,
    resolution_bits: int,
    channel_mask: int,
    channel_names: list[str],
    flush_interval_samples: int,
    csv_path: Path,
    live_buffer: EMGLiveBuffer,
    stop_event: threading.Event,
    finished_event: threading.Event,
    device_holder: dict[str, object],
    device_lock: threading.Lock,
    results: dict[str, object],
) -> None:
    """Run the blocking PLUX device loop outside the Qt GUI thread."""

    device_class = create_biosignals_device_class(plux)
    session_origin = time.perf_counter()

    device = None
    device_started = False
    termination_reason = "unknown"

    try:
        with EMGCSVWriter(
            path=csv_path,
            channel_names=channel_names,
            sampling_rate_hz=sampling_rate_hz,
            flush_interval_samples=flush_interval_samples,
        ) as writer:
            try:
                device = device_class(
                    device_address,
                    writer=writer,
                    session_origin=session_origin,
                    expected_channel_count=len(channel_names),
                    live_buffer=live_buffer,
                )

                with device_lock:
                    device_holder["device"] = device

                if stop_event.is_set():
                    termination_reason = "stop_before_start"
                else:
                    print("Device object created.")
                    print("Starting Biosignalsplux acquisition...")

                    device.start(
                        sampling_rate_hz,
                        channel_mask,
                        resolution_bits,
                    )
                    device_started = True

                    if stop_event.is_set():
                        device.request_stop()

                    # Blocks until onRawFrame returns True or an error occurs.
                    device.loop()

                    if device.callback_error is not None:
                        raise BiosignalsDeviceError(
                            "The PLUX callback stopped after an error: "
                            f"{device.callback_error}"
                        )

                    termination_reason = (
                        "stop_requested"
                        if stop_event.is_set()
                        else "device_loop_completed"
                    )

            except Exception as exc:
                termination_reason = "unexpected_exception"
                results["runtime_error"] = exc

            finally:
                results["writer_stats"] = writer.stats

                if device is not None:
                    results["samples_received"] = device.samples_received
                    results["live_buffer_error_count"] = (
                        device.live_buffer_error_count
                    )
                    results["last_live_buffer_error"] = (
                        device.last_live_buffer_error
                    )

                if device is not None and device_started:
                    try:
                        device.stop()
                        print("Device streaming stopped.")
                    except Exception as exc:
                        results["stop_error"] = exc

                if device is not None:
                    try:
                        device.close()
                        print("Device connection closed.")
                    except Exception as exc:
                        results["close_error"] = exc

    except Exception as exc:
        termination_reason = "setup_exception"
        results["runtime_error"] = exc

    finally:
        with device_lock:
            device_holder["device"] = None

        results["termination_reason"] = termination_reason
        finished_event.set()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    session_config = config["session"]
    device_config = config["biosignalsplux"]

    api_path = device_config["api_path"]
    device_address = device_config["device_address"]
    sampling_rate_hz = device_config["sampling_rate_hz"]
    resolution_bits = device_config["resolution_bits"]
    channel_mask = device_config["channel_mask"]
    channel_names = device_config["channels"]
    flush_interval = session_config["flush_interval_samples"]

    if api_path == "REPLACE_WITH_LOCAL_PLUX_API_PATH":
        print("The local PLUX API path has not been configured.")
        return 2

    if device_address == "REPLACE_WITH_DEVICE_ADDRESS":
        print("The Biosignalsplux device address has not been configured.")
        return 2

    try:
        plux = load_plux_api(api_path)
    except PluxAPIError as exc:
        print(f"Could not load the PLUX API: {exc}")
        return 2

    session_id = generate_session_id(args.prefix)
    paths = create_session_paths(
        session_config["output_root"],
        session_id,
    )

    live_buffer = EMGLiveBuffer(
        channel_count=len(channel_names),
        capacity_samples=sampling_rate_hz * 5,
    )

    stop_event = threading.Event()
    finished_event = threading.Event()
    device_lock = threading.Lock()
    device_holder: dict[str, object] = {"device": None}

    results: dict[str, object] = {
        "termination_reason": "unknown",
        "runtime_error": None,
        "writer_stats": None,
        "samples_received": 0,
        "live_buffer_error_count": 0,
        "last_live_buffer_error": None,
        "stop_error": None,
        "close_error": None,
    }

    def request_safe_stop() -> None:
        """Request the device loop to finish at the next received frame."""

        stop_event.set()

        with device_lock:
            device = device_holder.get("device")

        if device is not None:
            try:
                device.request_stop()
            except Exception as exc:
                print(f"Warning while requesting stop: {exc}")

    print(f"Session ID: {session_id}")
    print(f"Device address: {device_address}")
    print(f"Sampling rate: {sampling_rate_hz} Hz")
    print(f"Channel mask: 0x{channel_mask:02X}")
    print(f"Resolution: {resolution_bits} bits")
    print(f"Channels: {channel_names}")
    print(f"Output CSV: {paths.emg_csv}")
    print("Opening live sEMG window...")

    app = QtWidgets.QApplication(sys.argv)

    window = EMGLiveWindow(
        live_buffer=live_buffer,
        channel_names=channel_names,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=5.0,
        refresh_interval_ms=50,
        stop_callback=request_safe_stop,
    )
    window.show()

    worker = threading.Thread(
        target=acquisition_worker,
        kwargs={
            "plux": plux,
            "device_address": device_address,
            "sampling_rate_hz": sampling_rate_hz,
            "resolution_bits": resolution_bits,
            "channel_mask": channel_mask,
            "channel_names": channel_names,
            "flush_interval_samples": flush_interval,
            "csv_path": paths.emg_csv,
            "live_buffer": live_buffer,
            "stop_event": stop_event,
            "finished_event": finished_event,
            "device_holder": device_holder,
            "device_lock": device_lock,
            "results": results,
        },
        name="biosignalsplux-acquisition",
        daemon=True,
    )
    worker.start()

    # Allows Ctrl+C to use the same safe-stop path as the GUI button.
    signal.signal(
        signal.SIGINT,
        lambda signum, frame: request_safe_stop(),
    )

    lifecycle_timer = QtCore.QTimer()

    def check_worker_state() -> None:
        if finished_event.is_set():
            lifecycle_timer.stop()
            window.timer.stop()
            window.close()
            app.quit()

    lifecycle_timer.timeout.connect(check_worker_state)
    lifecycle_timer.start(200)

    app.exec_()

    request_safe_stop()
    worker.join(timeout=10.0)

    if worker.is_alive():
        print("Acquisition thread did not stop within 10 seconds.")
        return 1

    print("\nBiosignalsplux GUI acquisition summary")
    print("---------------------------------------")
    print(f"Termination reason: {results['termination_reason']}")
    print(f"Samples received: {results['samples_received']}")

    writer_stats = results["writer_stats"]

    if writer_stats is not None:
        print(f"Samples written: {writer_stats.samples_written}")
        print(
            "Sequence gap events: "
            f"{writer_stats.sequence_gap_events}"
        )
        print(
            "Missing sequence count: "
            f"{writer_stats.missing_sequence_count}"
        )
        print(
            "Non-increasing sequences: "
            f"{writer_stats.non_increasing_sequence_count}"
        )

    snapshot = live_buffer.snapshot()
    print(
        "Samples copied to live buffer: "
        f"{snapshot.total_samples_received}"
    )
    print(
        "Samples retained in five-second buffer: "
        f"{snapshot.buffered_samples}"
    )
    print(
        "Live buffer errors: "
        f"{results['live_buffer_error_count']}"
    )
    print(f"CSV path: {paths.emg_csv}")

    for error_name in (
        "runtime_error",
        "stop_error",
        "close_error",
        "last_live_buffer_error",
    ):
        error = results[error_name]

        if error is not None:
            print(
                f"{error_name}: "
                f"{type(error).__name__}: {error}"
            )

    if results["runtime_error"] is not None:
        return 1

    if writer_stats is None or writer_stats.samples_written == 0:
        print("No samples were written; validation was not performed.")
        return 1

    counts_match = (
        results["samples_received"]
        == writer_stats.samples_written
        == snapshot.total_samples_received
    )

    print(
        "Device, CSV, and live-buffer counts match: "
        f"{counts_match}"
    )

    report = validate_emg_csv(paths.emg_csv)

    print("\nPost-acquisition validation")
    print("---------------------------")
    print(f"Valid: {report.is_valid}")
    print(f"Rows checked: {report.rows_checked}")
    print(f"Sequence gap events: {report.sequence_gap_events}")
    print(f"Missing sequence count: {report.missing_sequence_count}")
    print(f"Device time errors: {report.device_time_error_count}")
    print(f"Monotonic time errors: {report.monotonic_time_error_count}")

    for warning in report.warnings:
        print(f"Warning: {warning}")

    for error in report.errors:
        print(f"Error: {error}")

    return 0 if report.is_valid and counts_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
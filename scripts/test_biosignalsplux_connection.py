#!/usr/bin/env python
"""Acquire Biosignalsplux sEMG continuously until Ctrl+C."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from biosignalsplux_labview_sync.biosignals_device import (
    BiosignalsDeviceError,
    create_biosignals_device_class,
)
from biosignalsplux_labview_sync.config import load_config
from biosignalsplux_labview_sync.emg_writer import EMGCSVWriter
from biosignalsplux_labview_sync.live_buffer import EMGLiveBuffer
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
            "Acquire raw two-channel sEMG from Biosignalsplux "
            "continuously until Ctrl+C."
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
        default="biosignals_test",
        help="Prefix used to generate the session identifier.",
    )
    return parser.parse_args()


def print_validation_report(csv_path: Path) -> bool:
    """Validate the generated raw EMG file and print a compact report."""

    report = validate_emg_csv(csv_path)

    print("\nPost-acquisition validation")
    print("---------------------------")
    print(f"Valid: {report.is_valid}")
    print(f"Rows checked: {report.rows_checked}")
    print(f"Sequence gap events: {report.sequence_gap_events}")
    print(f"Missing sequence count: {report.missing_sequence_count}")
    print(
        "Non-increasing sequences: "
        f"{report.non_increasing_sequence_count}"
    )
    print(f"Sample index errors: {report.sample_index_error_count}")
    print(f"Device time errors: {report.device_time_error_count}")
    print(f"Monotonic time errors: {report.monotonic_time_error_count}")

    for warning in report.warnings:
        print(f"Warning: {warning}")

    for error in report.errors:
        print(f"Error: {error}")

    return report.is_valid


def report_live_status(
    stop_event: threading.Event,
    live_buffer: EMGLiveBuffer,
) -> None:
    """Print non-critical live-buffer statistics once per second."""

    while not stop_event.wait(1.0):
        snapshot = live_buffer.snapshot()
        print(
            "Live samples: "
            f"{snapshot.total_samples_received} | "
            f"Buffered: {snapshot.buffered_samples}"
        )


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

    device_class = create_biosignals_device_class(plux)
    expected_channel_count = len(channel_names)

    live_buffer = EMGLiveBuffer(
        channel_count=expected_channel_count,
        capacity_samples=sampling_rate_hz * 5,
    )
    status_stop_event = threading.Event()
    status_thread = threading.Thread(
        target=report_live_status,
        args=(status_stop_event, live_buffer),
        name="emg-status-reporter",
        daemon=True,
    )

    session_origin = time.perf_counter()

    device = None
    device_started = False
    termination_reason = "unknown"
    samples_received = 0
    writer_stats = None
    runtime_error: Exception | None = None

    print(f"Session ID: {session_id}")
    print(f"Device address: {device_address}")
    print(f"Sampling rate: {sampling_rate_hz} Hz")
    print(f"Channel mask: 0x{channel_mask:02X}")
    print(f"Resolution: {resolution_bits} bits")
    print(f"Channels: {channel_names}")
    print(f"Output CSV: {paths.emg_csv}")
    print("Connecting to Biosignalsplux...")

    try:
        with EMGCSVWriter(
            path=paths.emg_csv,
            channel_names=channel_names,
            sampling_rate_hz=sampling_rate_hz,
            flush_interval_samples=flush_interval,
        ) as writer:
            try:
                device = device_class(
                    device_address,
                    writer=writer,
                    session_origin=session_origin,
                    expected_channel_count=expected_channel_count,
                    live_buffer=live_buffer,
                )

                print("Device object created.")
                print("Starting acquisition...")
                print("Press Ctrl+C to stop safely.")

                device.start(
                    sampling_rate_hz,
                    channel_mask,
                    resolution_bits,
                )
                device_started = True
                status_thread.start()

                # The PLUX SDK calls onRawFrame repeatedly from this loop.
                # No protocol duration is used to terminate acquisition.
                device.loop()

                if device.callback_error is not None:
                    raise BiosignalsDeviceError(
                        "The PLUX callback stopped after an error: "
                        f"{device.callback_error}"
                    )

                termination_reason = "device_loop_completed"

            except KeyboardInterrupt:
                termination_reason = "keyboard_interrupt"
                print("\nCtrl+C received. Closing acquisition safely...")

                if device is not None:
                    device.request_stop()

            except Exception as exc:
                termination_reason = "unexpected_exception"
                runtime_error = exc

            finally:
                status_stop_event.set()

                if status_thread.is_alive():
                    status_thread.join(timeout=2.0)

                if device is not None:
                    samples_received = device.samples_received

                writer_stats = writer.stats

                if device is not None and device_started:
                    try:
                        device.stop()
                        print("Device streaming stopped.")
                    except Exception as exc:
                        print(f"Warning: device.stop() failed: {exc}")

                if device is not None:
                    try:
                        device.close()
                        print("Device connection closed.")
                    except Exception as exc:
                        print(f"Warning: device.close() failed: {exc}")

    except Exception as exc:
        termination_reason = "setup_exception"
        runtime_error = exc

    print("\nAcquisition summary")
    print("-------------------")
    print(f"Termination reason: {termination_reason}")
    print(f"Samples received: {samples_received}")

    live_snapshot = live_buffer.snapshot()
    print(
        "Samples copied to live buffer: "
        f"{live_snapshot.total_samples_received}"
    )
    print(
        "Samples currently retained in buffer: "
        f"{live_snapshot.buffered_samples}"
    )

    if device is not None:
        print(
            "Live buffer errors: "
            f"{device.live_buffer_error_count}"
        )

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

    print(f"CSV path: {paths.emg_csv}")

    if runtime_error is not None:
        print(
            "Acquisition failed: "
            f"{type(runtime_error).__name__}: {runtime_error}"
        )
        return 1

    if writer_stats is None or writer_stats.samples_written == 0:
        print("No samples were written; validation was not performed.")
        return 1

    counts_match = (
        samples_received
        == writer_stats.samples_written
        == live_snapshot.total_samples_received
    )

    print(f"Device, CSV, and live-buffer counts match: {counts_match}")

    if not counts_match:
        print("Acquisition failed: inconsistent sample counts.")
        return 1

    if device is not None and device.live_buffer_error_count:
        print("Acquisition failed: live-buffer errors were detected.")
        return 1

    is_valid = print_validation_report(paths.emg_csv)
    return 0 if is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
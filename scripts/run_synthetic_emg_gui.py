#!/usr/bin/env python
"""Run synthetic continuous sEMG acquisition with a live GUI window."""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from PyQt5 import QtWidgets

from biosignalsplux_labview_sync.config import load_config
from biosignalsplux_labview_sync.emg_writer import EMGCSVWriter
from biosignalsplux_labview_sync.live_buffer import EMGLiveBuffer
from biosignalsplux_labview_sync.live_window import EMGLiveWindow
from biosignalsplux_labview_sync.session import (
    create_session_paths,
    generate_session_id,
)
from biosignalsplux_labview_sync.validation import validate_emg_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic two-channel sEMG continuously with a "
            "live visualization window."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/acquisition.local.json"),
        help="Path to the acquisition JSON configuration.",
    )
    parser.add_argument(
        "--prefix",
        default="synthetic_gui",
        help="Prefix used to generate the session identifier.",
    )
    return parser.parse_args()


def synthetic_channel_values(
    elapsed_s: float,
    resolution_bits: int,
) -> list[int]:
    """Generate two bounded synthetic integer channels."""

    maximum_value = (2**resolution_bits) - 1
    midpoint = maximum_value / 2

    noise_0 = random.gauss(0.0, maximum_value * 0.002)
    noise_1 = random.gauss(0.0, maximum_value * 0.002)

    burst = 1.0 if int(elapsed_s) % 4 in (1, 2) else 0.15

    signal_0 = (
        midpoint
        + burst * maximum_value * 0.04 * math.sin(2 * math.pi * 70 * elapsed_s)
        + noise_0
    )
    signal_1 = (
        midpoint
        + burst * maximum_value * 0.03 * math.sin(2 * math.pi * 95 * elapsed_s)
        + noise_1
    )

    return [
        int(max(0, min(maximum_value, signal_0))),
        int(max(0, min(maximum_value, signal_1))),
    ]


def acquisition_worker(
    stop_event: threading.Event,
    live_buffer: EMGLiveBuffer,
    csv_path: Path,
    channel_names: list[str],
    sampling_rate_hz: int,
    resolution_bits: int,
    flush_interval_samples: int,
    stats: dict[str, object],
) -> None:
    """Generate synthetic samples continuously until stop is requested."""

    sample_period_s = 1.0 / sampling_rate_hz
    session_origin = time.perf_counter()
    next_sample_time = session_origin
    device_sequence = 0

    try:
        with EMGCSVWriter(
            path=csv_path,
            channel_names=channel_names,
            sampling_rate_hz=sampling_rate_hz,
            flush_interval_samples=flush_interval_samples,
        ) as writer:
            while not stop_event.is_set():
                now = time.perf_counter()

                if now < next_sample_time:
                    time.sleep(min(next_sample_time - now, sample_period_s))
                    continue

                elapsed_s = now - session_origin
                values = synthetic_channel_values(
                    elapsed_s,
                    resolution_bits,
                )

                writer.append_sample(
                    device_sequence=device_sequence,
                    time_monotonic_s=elapsed_s,
                    timestamp_host=datetime.now(timezone.utc),
                    channel_values=values,
                )

                try:
                    live_buffer.append_sample(
                        time_monotonic_s=elapsed_s,
                        channel_values=values,
                    )
                except Exception as exc:
                    stats["live_buffer_error"] = exc
                    stop_event.set()
                    break

                device_sequence += 1
                next_sample_time += sample_period_s

                if now - next_sample_time > 5 * sample_period_s:
                    next_sample_time = now + sample_period_s

            stats["writer_stats"] = writer.stats

    except Exception as exc:
        stats["runtime_error"] = exc
    finally:
        stats["samples_generated"] = device_sequence


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    session_config = config["session"]
    device_config = config["biosignalsplux"]

    sampling_rate_hz = device_config["sampling_rate_hz"]
    resolution_bits = device_config["resolution_bits"]
    channel_names = device_config["channels"]
    flush_interval = session_config["flush_interval_samples"]

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
    stats: dict[str, object] = {
        "samples_generated": 0,
        "writer_stats": None,
        "runtime_error": None,
        "live_buffer_error": None,
    }

    worker = threading.Thread(
        target=acquisition_worker,
        args=(
            stop_event,
            live_buffer,
            paths.emg_csv,
            channel_names,
            sampling_rate_hz,
            resolution_bits,
            flush_interval,
            stats,
        ),
        name="synthetic-emg-gui-worker",
        daemon=True,
    )

    print(f"Session ID: {session_id}")
    print(f"Sampling rate: {sampling_rate_hz} Hz")
    print(f"Channels: {channel_names}")
    print(f"Output CSV: {paths.emg_csv}")
    print("Opening live GUI window...")

    worker.start()

    app = QtWidgets.QApplication(sys.argv)
    window = EMGLiveWindow(
        live_buffer=live_buffer,
        channel_names=channel_names,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=5.0,
        refresh_interval_ms=50,
        stop_callback=stop_event.set,
    )
    window.show()
    app.exec_()

    stop_event.set()
    worker.join(timeout=5.0)

    print("\nSynthetic GUI acquisition summary")
    print("---------------------------------")
    print(f"Samples generated: {stats['samples_generated']}")

    if stats["live_buffer_error"] is not None:
        print(
            "Live buffer error: "
            f"{type(stats['live_buffer_error']).__name__}: "
            f"{stats['live_buffer_error']}"
        )
        return 1

    if stats["runtime_error"] is not None:
        print(
            "Acquisition failed: "
            f"{type(stats['runtime_error']).__name__}: "
            f"{stats['runtime_error']}"
        )
        return 1

    writer_stats = stats["writer_stats"]
    if writer_stats is None:
        print("No writer statistics were produced.")
        return 1

    print(f"Samples written: {writer_stats.samples_written}")
    print(f"Sequence gap events: {writer_stats.sequence_gap_events}")
    print(f"Missing sequence count: {writer_stats.missing_sequence_count}")
    print(
        "Non-increasing sequences: "
        f"{writer_stats.non_increasing_sequence_count}"
    )

    snapshot = live_buffer.snapshot()
    print(
        "Live buffer total received: "
        f"{snapshot.total_samples_received}"
    )
    print(
        "Live buffer retained samples: "
        f"{snapshot.buffered_samples}"
    )

    if writer_stats.samples_written == 0:
        print("No samples were written.")
        return 1

    report = validate_emg_csv(paths.emg_csv)
    print(f"Validation passed: {report.is_valid}")

    if report.warnings:
        for warning in report.warnings:
            print(f"Warning: {warning}")

    if report.errors:
        for error in report.errors:
            print(f"Error: {error}")

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
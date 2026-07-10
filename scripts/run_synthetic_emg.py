#!/usr/bin/env python
"""Run an indefinite synthetic sEMG acquisition until Ctrl+C."""

from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from biosignalsplux_labview_sync.config import load_config
from biosignalsplux_labview_sync.emg_writer import EMGCSVWriter
from biosignalsplux_labview_sync.session import (
    create_session_paths,
    generate_session_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic two-channel sEMG continuously until Ctrl+C."
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
        default="synthetic_emg",
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

    sample_period_s = 1.0 / sampling_rate_hz
    session_origin = time.perf_counter()
    next_sample_time = session_origin
    last_status_time = session_origin
    device_sequence = 0
    termination_reason = "unknown"

    print(f"Session ID: {session_id}")
    print(f"Sampling rate: {sampling_rate_hz} Hz")
    print(f"Channels: {channel_names}")
    print(f"Output CSV: {paths.emg_csv}")
    print("Synthetic acquisition started.")
    print("Press Ctrl+C to stop safely.")

    try:
        with EMGCSVWriter(
            path=paths.emg_csv,
            channel_names=channel_names,
            sampling_rate_hz=sampling_rate_hz,
            flush_interval_samples=flush_interval,
        ) as writer:
            try:
                while True:
                    now = time.perf_counter()

                    if now < next_sample_time:
                        time.sleep(
                            min(
                                next_sample_time - now,
                                sample_period_s,
                            )
                        )
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

                    device_sequence += 1
                    next_sample_time += sample_period_s

                    # Avoid an uncontrolled catch-up loop if Windows delays
                    # execution for several sampling periods.
                    if now - next_sample_time > 5 * sample_period_s:
                        next_sample_time = now + sample_period_s

                    if now - last_status_time >= 1.0:
                        stats = writer.stats
                        print(
                            "Samples: "
                            f"{stats.samples_written} | "
                            "Missing sequences: "
                            f"{stats.missing_sequence_count}"
                        )
                        last_status_time = now

            except KeyboardInterrupt:
                termination_reason = "keyboard_interrupt"
                print("\nCtrl+C received. Closing acquisition safely...")

            finally:
                final_stats = writer.stats

    except Exception:
        termination_reason = "unexpected_exception"
        raise

    print("Acquisition closed.")
    print(f"Termination reason: {termination_reason}")
    print(f"Samples written: {final_stats.samples_written}")
    print(f"Sequence gap events: {final_stats.sequence_gap_events}")
    print(f"Missing sequences: {final_stats.missing_sequence_count}")
    print(
        "Non-increasing sequences: "
        f"{final_stats.non_increasing_sequence_count}"
    )
    print(f"CSV saved at: {paths.emg_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python
"""Validate ATI calibration, NI-DAQ acquisition, storage, and live display."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from biosignalsplux_labview_sync.config import load_config
from biosignalsplux_labview_sync.force_live_window import ForceLiveWindow
from biosignalsplux_labview_sync.force_runtime import (
    build_force_summary,
    force_settings_from_config,
    prepare_force_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test continuous ATI force/torque acquisition."
    )
    parser.add_argument(
        "--config",
        default="configs/acquisition.local.json",
        help="Local project configuration JSON.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root overriding session.output_root.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional automatic stop duration. Without it, acquisition is indefinite.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Acquire without the PyQtGraph monitor.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    settings = force_settings_from_config(config)
    if settings is None:
        print("[ERROR] force_sensor is missing or disabled in the local config.")
        return 2

    output_root = Path(args.output_root or config["session"]["output_root"])
    session_id = datetime.now(timezone.utc).strftime("force_test_%Y%m%dT%H%M%SZ")
    session_directory = output_root / session_id
    session_directory.mkdir(parents=True, exist_ok=False)
    session_origin = time.perf_counter()

    runtime = prepare_force_runtime(
        settings=settings,
        session_directory=session_directory,
        session_id=session_id,
        session_origin=session_origin,
    )

    print(f"ATI device:       {settings.device}")
    print(f"Channels:         {', '.join(settings.channels)}")
    print(f"Sampling rate:    {settings.sampling_rate_hz} Hz")
    print(f"Bias interval:    {settings.bias_seconds:.3f} s")
    print(f"Calibration file: {settings.calibration_file}")
    print(f"Output CSV:       {runtime.csv_path}")
    print("Keep the sensor unloaded during the initial bias interval.")

    runtime.acquisition.start()

    if args.no_gui:
        try:
            started = time.perf_counter()
            while runtime.acquisition.is_running:
                if args.duration_s is not None and time.perf_counter() - started >= args.duration_s:
                    runtime.acquisition.request_stop()
                time.sleep(0.1)
        except KeyboardInterrupt:
            runtime.acquisition.request_stop()
        runtime.acquisition.join(timeout=10.0)
    else:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        window = ForceLiveWindow(
            live_buffer=runtime.live_buffer,
            sampling_rate_hz=settings.sampling_rate_hz,
            window_seconds=settings.live_window_seconds,
            stop_callback=runtime.acquisition.request_stop,
        )
        window.setWindowTitle(f"ATI F/T · {session_id}")
        timer = QtCore.QTimer(window)

        started = time.perf_counter()

        def check_state() -> None:
            if args.duration_s is not None and time.perf_counter() - started >= args.duration_s:
                runtime.acquisition.request_stop()
            if runtime.acquisition.is_finished:
                timer.stop()
                window.close()
                app.quit()

        timer.timeout.connect(check_state)
        timer.start(100)
        window.show()
        app.exec_()
        if runtime.acquisition.is_running:
            runtime.acquisition.request_stop()
        runtime.acquisition.join(timeout=10.0)

    summary = build_force_summary(runtime)
    summary_path = session_directory / f"force_summary_{session_id}.json"
    summary_path.write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Exception) else value
                for key, value in summary.items()
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("ATI acquisition summary")
    print("-----------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"summary_json: {summary_path}")

    return 1 if summary["runtime_error"] is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())

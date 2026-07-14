#!/usr/bin/env python
"""Run a complete Biosignalsplux-LabVIEW synchronized session."""

from __future__ import annotations

import argparse

from biosignalsplux_labview_sync.synchronized_acquisition_window import (
    run_synchronized_session_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Configure a participant, acquire Biosignalsplux sEMG "
            "at 1000 Hz, and execute a three-reference LabVIEW "
            "trajectory."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/acquisition.local.json",
        help="Local hardware and session configuration JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    return run_synchronized_session_window(
        config_path=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
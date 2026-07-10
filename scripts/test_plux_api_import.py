#!/usr/bin/env python
"""Verify that the external PLUX Python API can be imported."""

from __future__ import annotations

import argparse
from pathlib import Path

from biosignalsplux_labview_sync.config import load_config
from biosignalsplux_labview_sync.plux_api import (
    PluxAPIError,
    load_plux_api,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the local PLUX Python API installation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/acquisition.local.json"),
        help="Path to the local acquisition configuration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    api_path = config["biosignalsplux"]["api_path"]

    print(f"PLUX API directory: {api_path}")
    print("Attempting to import plux...")

    try:
        plux = load_plux_api(api_path)
    except PluxAPIError as exc:
        print(f"PLUX API import failed: {exc}")
        return 1

    print("PLUX API import successful.")
    print(f"Module location: {getattr(plux, '__file__', 'unknown')}")
    print(f"SignalsDev available: {hasattr(plux, 'SignalsDev')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
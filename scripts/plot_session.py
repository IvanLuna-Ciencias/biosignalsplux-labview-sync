#!/usr/bin/env python
"""Plot one synchronized Biosignalsplux-LabVIEW session."""

from __future__ import annotations

import argparse
from pathlib import Path

from biosignalsplux_labview_sync.session_plot import (
    SessionPlotError,
    create_session_plots,
    find_latest_session_directory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw sEMG, calculated Python references, "
            "and synchronized session events."
        )
    )

    parser.add_argument(
        "--session-dir",
        type=Path,
        help=(
            "Specific session directory. If omitted, the newest "
            "session under --output-root is used."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory used to locate the newest session.",
    )

    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Optional output directory for PNG and JSON files.",
    )

    parser.add_argument(
        "--max-emg-points",
        type=int,
        default=200_000,
        help=(
            "Maximum sEMG points drawn per channel. "
            "The original CSV is never modified."
        ),
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening interactive windows.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        session_directory = (
            args.session_dir
            if args.session_dir is not None
            else find_latest_session_directory(
                args.output_root
            )
        )

        summary = create_session_plots(
            session_directory=session_directory,
            output_directory=args.plot_dir,
            max_emg_points=args.max_emg_points,
            show=not args.no_show,
        )

    except SessionPlotError as exc:
        print(f"[ERROR] {exc}")
        return 2

    emg = summary["emg"]
    trajectory = summary["trajectory"]

    print()
    print("Session plot summary")
    print("--------------------")
    print(
        f"Session directory: {summary['session_directory']}"
    )
    print(
        f"sEMG rows:         {emg['rows']}"
    )
    print(
        f"sEMG channels:     {', '.join(emg['channels'])}"
    )
    print(
        f"sEMG duration:     {emg['duration_s']:.3f} s"
    )

    emg_rate = emg["effective_rate_hz"]

    if emg_rate is not None:
        print(
            f"Effective sEMG rate: {emg_rate:.3f} Hz"
        )

    if trajectory is not None:
        print(
            f"Setpoint rows:     {trajectory['rows']}"
        )

        trajectory_rate = trajectory[
            "effective_rate_hz"
        ]

        if trajectory_rate is not None:
            print(
                "Effective setpoint rate: "
                f"{trajectory_rate:.3f} Hz"
            )
    else:
        print(
            "Setpoint data:     not found"
        )

    print(
        f"Events:            {summary['events']['count']}"
    )
    print(
        f"Plots:             {len(summary['plots'])}"
    )

    for plot_path in summary["plots"]:
        print(
            f"  - {plot_path}"
        )

    print(
        f"Summary JSON:      {summary['summary_json']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

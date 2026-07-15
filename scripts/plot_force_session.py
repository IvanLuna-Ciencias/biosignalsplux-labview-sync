#!/usr/bin/env python
"""Plot one ATI force/torque CSV in two synchronized panels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FORCE_COLUMNS = ("fx_n", "fy_n", "fz_n")
TORQUE_COLUMNS = ("mx_nm", "my_nm", "mz_nm")


def latest_force_csv(output_root: Path) -> Path:
    matches = list(output_root.glob("**/force_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No force_*.csv files were found under {output_root}.")
    return max(matches, key=lambda path: path.stat().st_mtime)


def read_force_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = ("time_monotonic_s", *FORCE_COLUMNS, *TORQUE_COLUMNS)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
        data = {name: [] for name in required}
        for row in reader:
            for name in required:
                data[name].append(float(row[name]))
    if not data["time_monotonic_s"]:
        raise ValueError(f"Force CSV is empty: {path}")
    time_values = np.asarray(data.pop("time_monotonic_s"), dtype=float)
    return time_values, {name: np.asarray(values, dtype=float) for name, values in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ATI forces and torques.")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv or latest_force_csv(args.output_root)
    times, channels = read_force_csv(csv_path)

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
    for name in FORCE_COLUMNS:
        axes[0].plot(times, channels[name], linewidth=0.8, label=name)
    for name in TORQUE_COLUMNS:
        axes[1].plot(times, channels[name], linewidth=0.8, label=name)

    axes[0].set_title("Fuerzas ATI")
    axes[0].set_ylabel("Fuerza (N)")
    axes[1].set_title("Torques ATI")
    axes[1].set_ylabel("Torque (N·m)")
    axes[1].set_xlabel("Tiempo de sesión (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()

    figure.suptitle(csv_path.parent.name)
    figure.tight_layout()
    output_path = csv_path.parent / "plots" / "session_force_torque.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    print(f"Force CSV: {csv_path}")
    print(f"Plot:      {output_path}")
    if not args.no_show:
        plt.show()
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

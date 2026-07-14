"""Post-session visualization and integrity summaries."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class SessionPlotError(RuntimeError):
    """Raised when session files cannot be read or plotted."""


@dataclass(frozen=True)
class EventRecord:
    """One synchronized session event."""

    time_s: float
    name: str
    details_json: str


@dataclass(frozen=True)
class SessionPlotFiles:
    """Files discovered inside one session directory."""

    session_directory: Path
    emg_csv: Path
    trajectories_csv: Path | None
    events_csv: Path | None
    sync_report_json: Path | None


MAJOR_EVENTS = {
    "TRAJECTORY_SESSION_STARTED",
    "MOVE_TO_START",
    "TRAJECTORY_STARTED",
    "PAUSE",
    "RESUME",
    "TRAJECTORY_DONE",
    "STOP_REQUESTED",
    "RETURN_STARTED",
    "RETURN_DONE",
    "TRAJECTORY_SESSION_FINISHED",
    "FAULT",
}


def _latest_matching_file(
    directory: Path,
    patterns: tuple[str, ...],
) -> Path | None:
    matches: list[Path] = []

    for pattern in patterns:
        matches.extend(
            path
            for path in directory.glob(pattern)
            if path.is_file()
        )

    if not matches:
        return None

    unique_matches = list(set(matches))

    return max(
        unique_matches,
        key=lambda path: path.stat().st_mtime,
    )


def find_latest_session_directory(
    output_root: str | Path = "outputs",
) -> Path:
    """Find the newest output directory containing an sEMG CSV."""

    root = Path(output_root)

    if not root.exists():
        raise SessionPlotError(
            f"Output directory does not exist: {root}"
        )

    candidates = []

    for directory in root.iterdir():
        if not directory.is_dir():
            continue

        emg_file = _latest_matching_file(
            directory,
            (
                "emg_*.csv",
                "*emg*.csv",
            ),
        )

        if emg_file is not None:
            candidates.append(directory)

    if not candidates:
        raise SessionPlotError(
            f"No session directories with sEMG data were found in {root}."
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def discover_session_files(
    session_directory: str | Path,
) -> SessionPlotFiles:
    """Locate sEMG, trajectory, event, and report files."""

    directory = Path(session_directory)

    if not directory.exists() or not directory.is_dir():
        raise SessionPlotError(
            f"Session directory does not exist: {directory}"
        )

    emg_csv = _latest_matching_file(
        directory,
        (
            "emg_*.csv",
            "*emg*.csv",
        ),
    )

    if emg_csv is None:
        raise SessionPlotError(
            f"No sEMG CSV was found in {directory}."
        )

    trajectories_csv = _latest_matching_file(
        directory,
        (
            "trajectories_*.csv",
        ),
    )

    events_csv = _latest_matching_file(
        directory,
        (
            "events_*.csv",
            "*events*.csv",
        ),
    )

    sync_report_json = _latest_matching_file(
        directory,
        (
            "sync_report_*.json",
            "*sync*report*.json",
        ),
    )

    return SessionPlotFiles(
        session_directory=directory,
        emg_csv=emg_csv,
        trajectories_csv=trajectories_csv,
        events_csv=events_csv,
        sync_report_json=sync_report_json,
    )


def _parse_float(
    value: str | None,
    *,
    field_name: str,
    row_number: int,
    path: Path,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SessionPlotError(
            f"Invalid {field_name!r} at row {row_number} in {path}."
        ) from exc

    if not math.isfinite(number):
        raise SessionPlotError(
            f"Non-finite {field_name!r} at row {row_number} in {path}."
        )

    return number


def read_emg_csv(
    path: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read session-relative time and all emg_* channels."""

    csv_path = Path(path)

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        if "time_monotonic_s" not in fieldnames:
            raise SessionPlotError(
                f"{csv_path} does not contain time_monotonic_s."
            )

        channel_names = [
            name
            for name in fieldnames
            if name.startswith("emg_")
        ]

        if not channel_names:
            raise SessionPlotError(
                f"{csv_path} does not contain emg_* channels."
            )

        times: list[float] = []
        channels: dict[str, list[float]] = {
            name: []
            for name in channel_names
        }

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            times.append(
                _parse_float(
                    row.get("time_monotonic_s"),
                    field_name="time_monotonic_s",
                    row_number=row_number,
                    path=csv_path,
                )
            )

            for channel_name in channel_names:
                channels[channel_name].append(
                    _parse_float(
                        row.get(channel_name),
                        field_name=channel_name,
                        row_number=row_number,
                        path=csv_path,
                    )
                )

    if not times:
        raise SessionPlotError(
            f"The sEMG CSV is empty: {csv_path}"
        )

    return (
        np.asarray(times, dtype=float),
        {
            name: np.asarray(values, dtype=float)
            for name, values in channels.items()
        },
    )


def read_trajectories_csv(
    path: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read the three transmitted references in radians."""

    csv_path = Path(path)

    required_fields = (
        "session_time_s",
        "q_shoulder_ref_rad",
        "q_elbow_ref_rad",
        "q_rotation_ref_rad",
    )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        missing = [
            name
            for name in required_fields
            if name not in fieldnames
        ]

        if missing:
            raise SessionPlotError(
                f"{csv_path} is missing columns: {', '.join(missing)}."
            )

        columns: dict[str, list[float]] = {
            name: []
            for name in required_fields
        }

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            for field_name in required_fields:
                columns[field_name].append(
                    _parse_float(
                        row.get(field_name),
                        field_name=field_name,
                        row_number=row_number,
                        path=csv_path,
                    )
                )

    if not columns["session_time_s"]:
        raise SessionPlotError(
            f"The trajectory CSV is empty: {csv_path}"
        )

    times = np.asarray(
        columns.pop("session_time_s"),
        dtype=float,
    )

    return (
        times,
        {
            name: np.asarray(values, dtype=float)
            for name, values in columns.items()
        },
    )


def read_events_csv(
    path: str | Path,
) -> list[EventRecord]:
    """Read synchronized session events."""

    csv_path = Path(path)
    events: list[EventRecord] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        required = {
            "session_time_s",
            "event",
        }

        if not required.issubset(fieldnames):
            raise SessionPlotError(
                f"{csv_path} does not contain the required event columns."
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            event_name = str(
                row.get("event", "")
            ).strip()

            if not event_name:
                continue

            events.append(
                EventRecord(
                    time_s=_parse_float(
                        row.get("session_time_s"),
                        field_name="session_time_s",
                        row_number=row_number,
                        path=csv_path,
                    ),
                    name=event_name,
                    details_json=str(
                        row.get("details_json", "")
                    ),
                )
            )

    return events


def effective_rate_hz(
    times_s: np.ndarray,
) -> float | None:
    """Estimate average sample rate from the first and last samples."""

    if len(times_s) < 2:
        return None

    duration_s = float(
        times_s[-1] - times_s[0]
    )

    if duration_s <= 0:
        return None

    return float(
        (len(times_s) - 1) / duration_s
    )


def downsample_indices(
    sample_count: int,
    max_points: int,
) -> np.ndarray:
    """Return display-only indices without modifying stored data."""

    if sample_count <= 0:
        return np.asarray([], dtype=int)

    if max_points <= 0:
        raise SessionPlotError(
            "max_points must be greater than zero."
        )

    if sample_count <= max_points:
        return np.arange(
            sample_count,
            dtype=int,
        )

    return np.linspace(
        0,
        sample_count - 1,
        max_points,
        dtype=int,
    )


def _add_event_markers(
    axis: Any,
    events: list[EventRecord],
) -> None:
    """Add synchronized event markers to one plot."""

    for event in events:
        axis.axvline(
            event.time_s,
            linestyle="--",
            linewidth=0.8,
            alpha=0.3,
        )

        if event.name in MAJOR_EVENTS:
            axis.annotate(
                event.name,
                xy=(event.time_s, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -5),
                textcoords="offset points",
                rotation=90,
                verticalalignment="top",
                fontsize=7,
                alpha=0.75,
            )


def create_session_plots(
    *,
    session_directory: str | Path,
    output_directory: str | Path | None = None,
    max_emg_points: int = 200_000,
    show: bool = True,
) -> dict[str, Any]:
    """Create sEMG and trajectory plots plus a JSON summary."""

    import matplotlib.pyplot as plt

    files = discover_session_files(
        session_directory
    )

    plot_directory = (
        Path(output_directory)
        if output_directory is not None
        else files.session_directory / "plots"
    )
    plot_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    events = (
        read_events_csv(files.events_csv)
        if files.events_csv is not None
        else []
    )

    emg_time, emg_channels = read_emg_csv(
        files.emg_csv
    )

    emg_indices = downsample_indices(
        len(emg_time),
        max_emg_points,
    )

    figures = []
    output_plots: list[str] = []

    emg_figure = plt.figure(
        figsize=(14, 6)
    )
    emg_axis = emg_figure.add_subplot(111)

    for channel_name, values in emg_channels.items():
        emg_axis.plot(
            emg_time[emg_indices],
            values[emg_indices],
            linewidth=0.7,
            label=channel_name,
        )

    _add_event_markers(
        emg_axis,
        events,
    )

    emg_axis.set_title(
        "sEMG cruda de la sesión"
    )
    emg_axis.set_xlabel(
        "Tiempo de sesión (s)"
    )
    emg_axis.set_ylabel(
        "Amplitud cruda (ADC)"
    )
    emg_axis.grid(
        True,
        alpha=0.25,
    )
    emg_axis.legend()
    emg_figure.tight_layout()

    emg_plot_path = (
        plot_directory
        / "session_emg.png"
    )
    emg_figure.savefig(
        emg_plot_path,
        dpi=170,
        bbox_inches="tight",
    )
    figures.append(emg_figure)
    output_plots.append(
        str(emg_plot_path)
    )

    trajectory_summary: dict[str, Any] | None = None
    trajectory_time: np.ndarray | None = None
    references: dict[str, np.ndarray] | None = None

    if files.trajectories_csv is not None:
        trajectory_time, references = (
            read_trajectories_csv(
                files.trajectories_csv
            )
        )

        trajectory_figure = plt.figure(
            figsize=(14, 6)
        )
        trajectory_axis = (
            trajectory_figure.add_subplot(111)
        )

        reference_labels = {
            "q_shoulder_ref_rad": "Hombro",
            "q_elbow_ref_rad": "Codo",
            "q_rotation_ref_rad": "Rotación",
        }

        for column_name, values_rad in references.items():
            trajectory_axis.plot(
                trajectory_time,
                np.degrees(values_rad),
                linewidth=1.2,
                label=reference_labels[column_name],
            )

        _add_event_markers(
            trajectory_axis,
            events,
        )

        trajectory_axis.set_title(
            "Referencias enviadas a LabVIEW"
        )
        trajectory_axis.set_xlabel(
            "Tiempo de sesión (s)"
        )
        trajectory_axis.set_ylabel(
            "Referencia angular (°)"
        )
        trajectory_axis.grid(
            True,
            alpha=0.25,
        )
        trajectory_axis.legend()
        trajectory_figure.tight_layout()

        trajectory_plot_path = (
            plot_directory
            / "session_references.png"
        )
        trajectory_figure.savefig(
            trajectory_plot_path,
            dpi=170,
            bbox_inches="tight",
        )
        figures.append(trajectory_figure)
        output_plots.append(
            str(trajectory_plot_path)
        )

        trajectory_summary = {
            "rows": int(len(trajectory_time)),
            "start_time_s": float(
                trajectory_time[0]
            ),
            "end_time_s": float(
                trajectory_time[-1]
            ),
            "duration_s": float(
                trajectory_time[-1]
                - trajectory_time[0]
            ),
            "effective_rate_hz": effective_rate_hz(
                trajectory_time
            ),
        }


    if (
        trajectory_time is not None
        and references is not None
    ):
        active_reference_name = max(
            references,
            key=lambda name: float(
                np.ptp(references[name])
            ),
        )

        active_reference_deg = np.degrees(
            references[active_reference_name]
        )

        reference_labels = {
            "q_shoulder_ref_rad": "Hombro",
            "q_elbow_ref_rad": "Codo",
            "q_rotation_ref_rad": "Rotaci?n",
        }

        active_joint_label = reference_labels.get(
            active_reference_name,
            active_reference_name,
        )

        comparison_figure, comparison_axes = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(14, 10),
        )

        comparison_axes[0].plot(
            trajectory_time,
            active_reference_deg,
            linewidth=1.2,
        )
        comparison_axes[0].set_title(
            f"Trayectoria calculada por Python: "
            f"{active_joint_label}"
        )
        comparison_axes[0].set_ylabel(
            "Referencia (?)"
        )

        emg_names = list(
            emg_channels.keys()
        )

        for index in range(2):
            axis = comparison_axes[index + 1]

            if index < len(emg_names):
                channel_name = emg_names[index]
                values = emg_channels[channel_name]

                axis.plot(
                    emg_time[emg_indices],
                    values[emg_indices],
                    linewidth=0.65,
                )
                axis.set_title(
                    channel_name
                )
                axis.set_ylabel(
                    "ADC"
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "Canal no disponible",
                    transform=axis.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        for axis in comparison_axes:
            _add_event_markers(
                axis,
                events,
            )
            axis.grid(
                True,
                alpha=0.25,
            )

        comparison_axes[-1].set_xlabel(
            "Tiempo de sesi?n (s)"
        )

        comparison_figure.suptitle(
            "Comparaci?n sincronizada: "
            "trayectoria calculada y sEMG",
            fontsize=14,
        )
        comparison_figure.tight_layout()

        comparison_plot_path = (
            plot_directory
            / "session_synchronized_comparison.png"
        )

        comparison_figure.savefig(
            comparison_plot_path,
            dpi=170,
            bbox_inches="tight",
        )

        figures.append(
            comparison_figure
        )
        output_plots.append(
            str(comparison_plot_path)
        )

    sync_report = None

    if files.sync_report_json is not None:
        try:
            sync_report = json.loads(
                files.sync_report_json.read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            sync_report = {
                "warning": (
                    "The synchronization report could not be read."
                )
            }

    summary = {
        "session_directory": str(
            files.session_directory
        ),
        "emg_csv": str(
            files.emg_csv
        ),
        "trajectories_csv": (
            str(files.trajectories_csv)
            if files.trajectories_csv is not None
            else None
        ),
        "events_csv": (
            str(files.events_csv)
            if files.events_csv is not None
            else None
        ),
        "emg": {
            "rows": int(len(emg_time)),
            "channels": list(
                emg_channels.keys()
            ),
            "start_time_s": float(
                emg_time[0]
            ),
            "end_time_s": float(
                emg_time[-1]
            ),
            "duration_s": float(
                emg_time[-1]
                - emg_time[0]
            ),
            "effective_rate_hz": effective_rate_hz(
                emg_time
            ),
            "points_displayed": int(
                len(emg_indices)
            ),
        },
        "trajectory": trajectory_summary,
        "events": {
            "count": len(events),
            "names": [
                event.name
                for event in events
            ],
        },
        "plots": output_plots,
        "sync_report": sync_report,
    }

    summary_path = (
        plot_directory
        / "session_plot_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary["summary_json"] = str(
        summary_path
    )

    if show:
        plt.show()

    for figure in figures:
        plt.close(figure)

    return summary

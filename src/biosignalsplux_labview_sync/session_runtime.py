"""Preparation of one synchronized acquisition runtime."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biosignalsplux_labview_sync.acquisition_session import (
    BiosignalsAcquisitionSession,
)
from biosignalsplux_labview_sync.live_buffer import EMGLiveBuffer
from biosignalsplux_labview_sync.plux_api import load_plux_api
from biosignalsplux_labview_sync.session import (
    SessionPaths,
    create_session_paths,
)
from biosignalsplux_labview_sync.session_metadata import (
    SessionMetadata,
    write_metadata_json,
)


class SessionRuntimeError(RuntimeError):
    """Raised when runtime resources cannot be prepared."""


@dataclass(frozen=True)
class PreparedAcquisitionRuntime:
    """Resources required by one continuous sEMG acquisition."""

    metadata: SessionMetadata
    paths: SessionPaths
    live_buffer: EMGLiveBuffer
    acquisition: BiosignalsAcquisitionSession
    session_origin: float


def _validate_positive_number(
    value: object,
    field_name: str,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise SessionRuntimeError(
            f"{field_name} must be a finite number greater than zero."
        )

    return float(value)


def _validate_session_origin(
    session_origin: object,
) -> float:
    if (
        not isinstance(session_origin, (int, float))
        or isinstance(session_origin, bool)
        or not math.isfinite(float(session_origin))
        or float(session_origin) < 0
    ):
        raise SessionRuntimeError(
            "session_origin must be a finite non-negative number."
        )

    return float(session_origin)


def prepare_acquisition_runtime(
    *,
    metadata: SessionMetadata,
    output_root: str | Path,
    api_path: str | Path,
    flush_interval_samples: int = 1000,
    live_window_seconds: float = 5.0,
    session_origin: float | None = None,
    plux_loader: Callable[[str | Path], Any] = load_plux_api,
    acquisition_factory: type[
        BiosignalsAcquisitionSession
    ] = BiosignalsAcquisitionSession,
) -> PreparedAcquisitionRuntime:
    """Create files, PLUX resources, buffer, and acquisition controller."""

    if not isinstance(metadata, SessionMetadata):
        raise SessionRuntimeError(
            "metadata must be a SessionMetadata instance."
        )

    if (
        not isinstance(flush_interval_samples, int)
        or isinstance(flush_interval_samples, bool)
        or flush_interval_samples <= 0
    ):
        raise SessionRuntimeError(
            "flush_interval_samples must be a positive integer."
        )

    window_seconds = _validate_positive_number(
        live_window_seconds,
        "live_window_seconds",
    )

    if session_origin is None:
        validated_origin = time.perf_counter()
    else:
        validated_origin = _validate_session_origin(
            session_origin
        )

    output_path = Path(output_root)

    if not str(output_path).strip():
        raise SessionRuntimeError(
            "output_root cannot be empty."
        )

    api_path_value = Path(api_path)

    if not str(api_path_value).strip():
        raise SessionRuntimeError(
            "api_path cannot be empty."
        )

    paths = create_session_paths(
        output_path,
        metadata.session_id,
    )

    # Save the exact configuration snapshot before opening hardware.
    write_metadata_json(
        paths.metadata_json,
        metadata,
    )

    plux = plux_loader(api_path_value)

    sampling_rate_hz = (
        metadata.biosignalsplux.sampling_rate_hz
    )
    channel_names = tuple(
        metadata.biosignalsplux.channels
    )

    capacity_samples = max(
        1,
        int(round(
            sampling_rate_hz * window_seconds
        )),
    )

    live_buffer = EMGLiveBuffer(
        channel_count=len(channel_names),
        capacity_samples=capacity_samples,
    )

    acquisition = acquisition_factory(
        plux=plux,
        device_address=(
            metadata.biosignalsplux.device_address
        ),
        sampling_rate_hz=sampling_rate_hz,
        resolution_bits=(
            metadata.biosignalsplux.resolution_bits
        ),
        channel_mask=(
            metadata.biosignalsplux.channel_mask
        ),
        channel_names=channel_names,
        flush_interval_samples=flush_interval_samples,
        csv_path=paths.emg_csv,
        live_buffer=live_buffer,
        session_origin=validated_origin,
    )

    return PreparedAcquisitionRuntime(
        metadata=metadata,
        paths=paths,
        live_buffer=live_buffer,
        acquisition=acquisition,
        session_origin=validated_origin,
    )


def build_acquisition_summary(
    runtime: PreparedAcquisitionRuntime,
) -> dict[str, object]:
    """Return a GUI-friendly summary of the completed acquisition."""

    results = runtime.acquisition.results
    writer_stats = results.writer_stats

    return {
        "session_id": runtime.metadata.session_id,
        "termination_reason": results.termination_reason,
        "samples_received": results.samples_received,
        "samples_written": (
            writer_stats.samples_written
            if writer_stats is not None
            else 0
        ),
        "sequence_gap_events": (
            writer_stats.sequence_gap_events
            if writer_stats is not None
            else 0
        ),
        "missing_sequence_count": (
            writer_stats.missing_sequence_count
            if writer_stats is not None
            else 0
        ),
        "non_increasing_sequence_count": (
            writer_stats.non_increasing_sequence_count
            if writer_stats is not None
            else 0
        ),
        "runtime_error": results.runtime_error,
        "request_stop_error": results.request_stop_error,
        "stop_error": results.stop_error,
        "close_error": results.close_error,
        "emg_csv": str(runtime.paths.emg_csv),
        "metadata_json": str(runtime.paths.metadata_json),
    }
"""Configuration and preparation helpers for optional ATI acquisition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from biosignalsplux_labview_sync.force_acquisition_session import (
    ATIForceAcquisitionSession,
)
from biosignalsplux_labview_sync.live_buffer import EMGLiveBuffer


class ForceRuntimeError(ValueError):
    """Raised when the optional force-sensor configuration is invalid."""


@dataclass(frozen=True)
class ForceSensorSettings:
    enabled: bool
    calibration_file: Path
    device: str
    channels: tuple[str, ...]
    sampling_rate_hz: int
    bias_seconds: float
    flush_interval_samples: int
    live_window_seconds: float

    @property
    def bias_samples(self) -> int:
        return max(1, int(round(self.sampling_rate_hz * self.bias_seconds)))


@dataclass(frozen=True)
class PreparedForceRuntime:
    settings: ForceSensorSettings
    csv_path: Path
    live_buffer: EMGLiveBuffer
    acquisition: ATIForceAcquisitionSession


def force_settings_from_config(config: dict[str, Any]) -> ForceSensorSettings | None:
    section = config.get("force_sensor")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ForceRuntimeError("force_sensor must be a JSON object.")

    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ForceRuntimeError("force_sensor.enabled must be true or false.")
    if not enabled:
        return None

    calibration_file = section.get("calibration_file")
    device = section.get("device", "Dev1")
    channels = section.get("channels")
    sampling_rate_hz = section.get("sampling_rate_hz", 1000)
    bias_seconds = section.get("bias_seconds", 1.0)
    flush_interval_samples = section.get("flush_interval_samples", 1000)
    live_window_seconds = section.get("live_window_seconds", 5.0)

    if not isinstance(calibration_file, str) or not calibration_file.strip():
        raise ForceRuntimeError("force_sensor.calibration_file must be a path string.")
    if not isinstance(device, str) or not device.strip():
        raise ForceRuntimeError("force_sensor.device must be a non-empty string.")
    if channels is None:
        channels = [f"{device}/ai{index}" for index in range(6)]
    if (
        not isinstance(channels, list)
        or len(channels) != 6
        or not all(isinstance(channel, str) and channel.strip() for channel in channels)
    ):
        raise ForceRuntimeError("force_sensor.channels must contain six channel names.")
    if (
        not isinstance(sampling_rate_hz, int)
        or isinstance(sampling_rate_hz, bool)
        or sampling_rate_hz <= 0
    ):
        raise ForceRuntimeError("force_sensor.sampling_rate_hz must be positive.")
    if (
        not isinstance(bias_seconds, (int, float))
        or isinstance(bias_seconds, bool)
        or not math.isfinite(float(bias_seconds))
        or float(bias_seconds) <= 0
    ):
        raise ForceRuntimeError("force_sensor.bias_seconds must be greater than zero.")
    if (
        not isinstance(flush_interval_samples, int)
        or isinstance(flush_interval_samples, bool)
        or flush_interval_samples <= 0
    ):
        raise ForceRuntimeError("force_sensor.flush_interval_samples must be positive.")
    if (
        not isinstance(live_window_seconds, (int, float))
        or isinstance(live_window_seconds, bool)
        or not math.isfinite(float(live_window_seconds))
        or float(live_window_seconds) <= 0
    ):
        raise ForceRuntimeError("force_sensor.live_window_seconds must be positive.")

    return ForceSensorSettings(
        enabled=True,
        calibration_file=Path(calibration_file),
        device=device.strip(),
        channels=tuple(channel.strip() for channel in channels),
        sampling_rate_hz=sampling_rate_hz,
        bias_seconds=float(bias_seconds),
        flush_interval_samples=flush_interval_samples,
        live_window_seconds=float(live_window_seconds),
    )


def prepare_force_runtime(
    *,
    settings: ForceSensorSettings,
    session_directory: str | Path,
    session_id: str,
    session_origin: float,
    acquisition_factory: Callable[..., ATIForceAcquisitionSession] = ATIForceAcquisitionSession,
) -> PreparedForceRuntime:
    directory = Path(session_directory)
    csv_path = directory / f"force_{session_id}.csv"
    capacity = max(
        1,
        int(round(settings.sampling_rate_hz * settings.live_window_seconds)),
    )
    live_buffer = EMGLiveBuffer(channel_count=6, capacity_samples=capacity)
    acquisition = acquisition_factory(
        calibration_file=settings.calibration_file,
        device=settings.device,
        channels=settings.channels,
        sampling_rate_hz=settings.sampling_rate_hz,
        bias_samples=settings.bias_samples,
        flush_interval_samples=settings.flush_interval_samples,
        csv_path=csv_path,
        live_buffer=live_buffer,
        session_origin=session_origin,
    )
    return PreparedForceRuntime(
        settings=settings,
        csv_path=csv_path,
        live_buffer=live_buffer,
        acquisition=acquisition,
    )


def build_force_summary(runtime: PreparedForceRuntime) -> dict[str, object]:
    results = runtime.acquisition.results
    return {
        "enabled": True,
        "termination_reason": results.termination_reason,
        "samples_received": results.samples_received,
        "samples_written": (
            results.writer_stats.samples_written
            if results.writer_stats is not None
            else 0
        ),
        "bias_samples_used": results.bias_samples_used,
        "bias_volts": results.bias_volts,
        "device_start_session_s": results.device_start_session_s,
        "runtime_error": results.runtime_error,
        "stop_error": results.stop_error,
        "live_buffer_error_count": results.live_buffer_error_count,
        "last_live_buffer_error": results.last_live_buffer_error,
        "force_csv": str(runtime.csv_path),
    }

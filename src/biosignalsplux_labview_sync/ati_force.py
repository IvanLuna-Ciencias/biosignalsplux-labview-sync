"""ATI six-axis force/torque acquisition through NI-DAQmx.

This module intentionally keeps hardware access separate from session control.
It loads the same 6x6 calibration matrix used by the laboratory reference
script and exposes raw voltage blocks for the acquisition layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from xml.etree import ElementTree as ET

import numpy as np


AXIS_NAMES = ("fx_n", "fy_n", "fz_n", "mx_nm", "my_nm", "mz_nm")
RAW_CHANNEL_NAMES = tuple(f"ai{index}_v" for index in range(6))


class ATIForceError(RuntimeError):
    """Raised when ATI calibration or NI-DAQ acquisition fails."""


@dataclass(frozen=True)
class ATICalibration:
    """Parsed ATI calibration metadata and 6x6 conversion matrix."""

    path: Path
    serial: str
    body_style: str
    force_units: str
    torque_units: str
    matrix: np.ndarray


def load_ati_calibration(path: str | Path) -> ATICalibration:
    """Load the first six Axis rows from an ATI XML .cal file."""

    cal_path = Path(path)
    if not cal_path.is_file():
        raise ATIForceError(
            f"ATI calibration file was not found: {cal_path}"
        )

    try:
        root = ET.parse(cal_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ATIForceError(
            f"Could not parse ATI calibration file {cal_path}: {exc}"
        ) from exc

    calibration_element = root.find(".//Calibration")
    if calibration_element is None:
        raise ATIForceError(
            f"No Calibration element was found in {cal_path}."
        )

    axes = calibration_element.findall(".//Axis")
    if len(axes) < 6:
        raise ATIForceError(
            f"Expected at least six Axis rows in {cal_path}; found {len(axes)}."
        )

    matrix = np.zeros((6, 6), dtype=np.float64)
    for row_index, axis in enumerate(axes[:6]):
        values = axis.get("values", "").split()
        if len(values) != 6:
            raise ATIForceError(
                f"Axis row {row_index} in {cal_path} does not contain six values."
            )
        try:
            matrix[row_index, :] = [float(value) for value in values]
        except ValueError as exc:
            raise ATIForceError(
                f"Axis row {row_index} in {cal_path} contains a non-numeric value."
            ) from exc

    if not np.isfinite(matrix).all():
        raise ATIForceError(
            f"Calibration matrix in {cal_path} contains non-finite values."
        )

    return ATICalibration(
        path=cal_path,
        serial=str(root.get("Serial", "unknown")),
        body_style=str(root.get("BodyStyle", "unknown")),
        force_units=str(calibration_element.get("ForceUnits", "N")),
        torque_units=str(calibration_element.get("TorqueUnits", "N-m")),
        matrix=matrix,
    )


def normalize_daq_block(
    raw: Any,
    *,
    channel_count: int = 6,
) -> np.ndarray:
    """Normalize NI-DAQmx read output to shape ``(samples, channels)``."""

    array = np.asarray(raw, dtype=np.float64)

    if array.size == 0:
        return np.empty((0, channel_count), dtype=np.float64)

    if array.ndim == 1:
        if channel_count == 1:
            array = array.reshape(-1, 1)
        elif array.size == channel_count:
            array = array.reshape(1, channel_count)
        else:
            raise ATIForceError(
                "One-dimensional DAQ data cannot be mapped to the configured "
                f"{channel_count} channels."
            )
    elif array.ndim == 2:
        # nidaqmx returns one list per channel for a multi-channel task.
        if array.shape[0] == channel_count:
            array = array.T
        elif array.shape[1] != channel_count:
            raise ATIForceError(
                f"Unexpected DAQ block shape {array.shape}; expected six channels."
            )
    else:
        raise ATIForceError(
            f"Unexpected DAQ block dimensions: {array.ndim}."
        )

    if array.shape[1] != channel_count:
        raise ATIForceError(
            f"DAQ block contains {array.shape[1]} channels; expected {channel_count}."
        )

    if not np.isfinite(array).all():
        raise ATIForceError("DAQ block contains non-finite voltage values.")

    return np.ascontiguousarray(array, dtype=np.float64)


def calibrate_force_block(
    raw_volts: np.ndarray,
    *,
    bias_volts: np.ndarray,
    calibration_matrix: np.ndarray,
) -> np.ndarray:
    """Convert a voltage block into ``Fx,Fy,Fz,Mx,My,Mz`` values."""

    raw_array = np.asarray(raw_volts, dtype=np.float64)
    bias_array = np.asarray(bias_volts, dtype=np.float64)
    matrix = np.asarray(calibration_matrix, dtype=np.float64)

    if raw_array.ndim != 2 or raw_array.shape[1] != 6:
        raise ATIForceError("raw_volts must have shape (samples, 6).")
    if bias_array.shape != (6,):
        raise ATIForceError("bias_volts must contain exactly six values.")
    if matrix.shape != (6, 6):
        raise ATIForceError("calibration_matrix must have shape (6, 6).")

    calibrated = (matrix @ (raw_array - bias_array).T).T
    if not np.isfinite(calibrated).all():
        raise ATIForceError("Calibrated force/torque values are non-finite.")
    return calibrated


class ATIDAQDevice:
    """Own one continuous NI-DAQmx task for the complete recording session."""

    def __init__(
        self,
        *,
        device: str = "Dev1",
        channels: Sequence[str] | None = None,
        sampling_rate_hz: int = 1000,
        read_timeout_s: float = 0.1,
        task_factory: Callable[[], Any] | None = None,
        read_all_available: Any | None = None,
        acquisition_type_continuous: Any | None = None,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ATIForceError("device must be a non-empty string.")
        if (
            not isinstance(sampling_rate_hz, int)
            or isinstance(sampling_rate_hz, bool)
            or sampling_rate_hz <= 0
        ):
            raise ATIForceError("sampling_rate_hz must be a positive integer.")
        if (
            not isinstance(read_timeout_s, (int, float))
            or isinstance(read_timeout_s, bool)
            or not math.isfinite(float(read_timeout_s))
            or float(read_timeout_s) <= 0
        ):
            raise ATIForceError("read_timeout_s must be greater than zero.")

        self.device = device.strip()
        self.channels = tuple(
            channels
            if channels is not None
            else (f"{self.device}/ai{index}" for index in range(6))
        )
        if len(self.channels) != 6 or any(
            not isinstance(channel, str) or not channel.strip()
            for channel in self.channels
        ):
            raise ATIForceError("Exactly six valid analog-input channels are required.")

        self.sampling_rate_hz = sampling_rate_hz
        self.read_timeout_s = float(read_timeout_s)
        self._task_factory = task_factory
        self._read_all_available = read_all_available
        self._continuous = acquisition_type_continuous
        self._task: Any | None = None

    def _resolve_nidaqmx(self) -> None:
        if self._task_factory is not None:
            return
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
        except ImportError as exc:
            raise ATIForceError(
                "nidaqmx is not installed. Install the project with the force extra."
            ) from exc

        self._task_factory = nidaqmx.Task
        self._read_all_available = READ_ALL_AVAILABLE
        self._continuous = AcquisitionType.CONTINUOUS

    def verify(self) -> None:
        """Open a temporary task and validate all configured channels."""

        self._resolve_nidaqmx()
        assert self._task_factory is not None
        task = self._task_factory()
        try:
            for channel in self.channels:
                task.ai_channels.add_ai_voltage_chan(channel)
        except Exception as exc:
            raise ATIForceError(
                f"Could not open ATI DAQ channels {self.channels}: {exc}"
            ) from exc
        finally:
            try:
                task.close()
            except Exception:
                pass

    def start(self) -> None:
        """Create and start one continuous hardware-clocked task."""

        if self._task is not None:
            raise ATIForceError("ATI DAQ task is already running.")

        self._resolve_nidaqmx()
        assert self._task_factory is not None

        task = self._task_factory()
        try:
            for channel in self.channels:
                task.ai_channels.add_ai_voltage_chan(channel)
            task.timing.cfg_samp_clk_timing(
                rate=self.sampling_rate_hz,
                sample_mode=self._continuous,
            )
            task.start()
        except Exception as exc:
            try:
                task.close()
            except Exception:
                pass
            raise ATIForceError(f"Could not start ATI DAQ acquisition: {exc}") from exc

        self._task = task

    def read_available(self) -> np.ndarray:
        """Read every sample currently available from the DAQ buffer."""

        if self._task is None:
            raise ATIForceError("ATI DAQ task has not been started.")

        try:
            raw = self._task.read(
                self._read_all_available,
                timeout=self.read_timeout_s,
            )
        except Exception as exc:
            raise ATIForceError(f"ATI DAQ read failed: {exc}") from exc

        return normalize_daq_block(raw, channel_count=6)

    def stop(self) -> None:
        """Stop and close the task. Safe to call more than once."""

        task = self._task
        self._task = None
        if task is None:
            return

        stop_error: Exception | None = None
        try:
            task.stop()
        except Exception as exc:
            stop_error = exc
        try:
            task.close()
        except Exception as exc:
            if stop_error is None:
                stop_error = exc

        if stop_error is not None:
            raise ATIForceError(f"Could not close ATI DAQ task: {stop_error}")

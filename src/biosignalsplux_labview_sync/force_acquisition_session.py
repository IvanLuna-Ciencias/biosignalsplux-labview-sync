"""Continuous ATI force/torque acquisition in a background thread."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from biosignalsplux_labview_sync.ati_force import (
    ATICalibration,
    ATIDAQDevice,
    calibrate_force_block,
    load_ati_calibration,
)
from biosignalsplux_labview_sync.force_writer import (
    ForceCSVWriter,
    ForceWriteStats,
)
from biosignalsplux_labview_sync.live_buffer import EMGLiveBuffer


class ForceAcquisitionError(RuntimeError):
    """Raised when a force acquisition session cannot be managed."""


@dataclass
class ForceAcquisitionResults:
    """Final and diagnostic information from one ATI acquisition."""

    termination_reason: str = "unknown"
    session_origin: float | None = None
    device_start_session_s: float | None = None
    runtime_error: Exception | None = None
    stop_error: Exception | None = None
    samples_received: int = 0
    writer_stats: ForceWriteStats | None = None
    bias_samples_used: int = 0
    bias_volts: tuple[float, ...] | None = None
    live_buffer_error_count: int = 0
    last_live_buffer_error: Exception | None = None


class ATIForceAcquisitionSession:
    """Acquire ATI data until an explicit stop or a hardware error."""

    def __init__(
        self,
        *,
        calibration_file: str | Path,
        device: str,
        channels: Sequence[str],
        sampling_rate_hz: int,
        bias_samples: int,
        flush_interval_samples: int,
        csv_path: str | Path,
        live_buffer: EMGLiveBuffer,
        session_origin: float | None = None,
        read_idle_sleep_s: float = 0.001,
        device_factory: Callable[..., ATIDAQDevice] = ATIDAQDevice,
        calibration_loader: Callable[[str | Path], ATICalibration] = load_ati_calibration,
        writer_factory: Callable[..., ForceCSVWriter] = ForceCSVWriter,
    ) -> None:
        if (
            not isinstance(sampling_rate_hz, int)
            or isinstance(sampling_rate_hz, bool)
            or sampling_rate_hz <= 0
        ):
            raise ForceAcquisitionError("sampling_rate_hz must be positive.")
        if (
            not isinstance(bias_samples, int)
            or isinstance(bias_samples, bool)
            or bias_samples <= 0
        ):
            raise ForceAcquisitionError("bias_samples must be positive.")
        if (
            not isinstance(flush_interval_samples, int)
            or isinstance(flush_interval_samples, bool)
            or flush_interval_samples <= 0
        ):
            raise ForceAcquisitionError("flush_interval_samples must be positive.")
        if live_buffer.channel_count != 6:
            raise ForceAcquisitionError("force live buffer must contain six channels.")
        if session_origin is not None and (
            not isinstance(session_origin, (int, float))
            or isinstance(session_origin, bool)
            or not math.isfinite(float(session_origin))
            or float(session_origin) < 0
        ):
            raise ForceAcquisitionError("session_origin must be non-negative.")

        self.calibration_file = Path(calibration_file)
        self.device = device
        self.channels = tuple(channels)
        self.sampling_rate_hz = sampling_rate_hz
        self.bias_samples = bias_samples
        self.flush_interval_samples = flush_interval_samples
        self.csv_path = Path(csv_path)
        self.live_buffer = live_buffer
        self._provided_session_origin = (
            None if session_origin is None else float(session_origin)
        )
        self.read_idle_sleep_s = float(read_idle_sleep_s)
        self._device_factory = device_factory
        self._calibration_loader = calibration_loader
        self._writer_factory = writer_factory

        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.bias_ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.results = ForceAcquisitionResults()

    @property
    def is_started(self) -> bool:
        return self._thread is not None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_finished(self) -> bool:
        return self.finished_event.is_set()

    @property
    def bias_ready(self) -> bool:
        return self.bias_ready_event.is_set()

    def start(self) -> None:
        if self._thread is not None:
            raise ForceAcquisitionError("This force acquisition has already started.")
        self._thread = threading.Thread(
            target=self._run_worker,
            name="ati-force-acquisition",
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> bool:
        if self._thread is None:
            raise ForceAcquisitionError("Cannot join an acquisition that was not started.")
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def stop_and_join(self, timeout: float = 10.0) -> bool:
        self.request_stop()
        return self.join(timeout=timeout)

    def _write_block(
        self,
        *,
        writer: ForceCSVWriter,
        raw_block: np.ndarray,
        ft_block: np.ndarray,
        first_sample_index: int,
        device_start_session_s: float,
        device_start_utc: datetime,
    ) -> None:
        for row_offset, (raw_row, ft_row) in enumerate(zip(raw_block, ft_block)):
            sample_index = first_sample_index + row_offset
            elapsed_device_s = sample_index / self.sampling_rate_hz
            session_time_s = device_start_session_s + elapsed_device_s
            timestamp = device_start_utc + timedelta(seconds=elapsed_device_s)
            writer.append_sample(
                device_sample_index=sample_index,
                time_monotonic_s=session_time_s,
                timestamp_host=timestamp,
                raw_volts=raw_row,
                force_torque=ft_row,
            )
            try:
                self.live_buffer.append_sample(session_time_s, ft_row)
            except Exception as exc:
                self.results.live_buffer_error_count += 1
                self.results.last_live_buffer_error = exc

    def _run_worker(self) -> None:
        session_origin = (
            self._provided_session_origin
            if self._provided_session_origin is not None
            else time.perf_counter()
        )
        self.results.session_origin = session_origin
        termination_reason = "unknown"
        daq: ATIDAQDevice | None = None
        pending_raw: list[np.ndarray] = []
        pending_count = 0
        next_sample_index = 0

        try:
            calibration = self._calibration_loader(self.calibration_file)
            daq = self._device_factory(
                device=self.device,
                channels=self.channels,
                sampling_rate_hz=self.sampling_rate_hz,
            )
            daq.verify()

            with self._writer_factory(
                path=self.csv_path,
                sampling_rate_hz=self.sampling_rate_hz,
                flush_interval_samples=self.flush_interval_samples,
            ) as writer:
                if self.stop_event.is_set():
                    termination_reason = "stop_before_start"
                else:
                    daq.start()
                    device_start_perf = time.perf_counter()
                    device_start_utc = datetime.now(timezone.utc)
                    device_start_session_s = max(
                        0.0,
                        device_start_perf - session_origin,
                    )
                    self.results.device_start_session_s = device_start_session_s

                    while not self.stop_event.is_set():
                        block = daq.read_available()
                        if block.shape[0] == 0:
                            time.sleep(self.read_idle_sleep_s)
                            continue

                        self.results.samples_received += int(block.shape[0])

                        if not self.bias_ready_event.is_set():
                            pending_raw.append(block)
                            pending_count += int(block.shape[0])
                            if pending_count < self.bias_samples:
                                continue

                            stacked = np.vstack(pending_raw)
                            bias = np.mean(stacked[: self.bias_samples], axis=0)
                            self.results.bias_samples_used = self.bias_samples
                            self.results.bias_volts = tuple(float(value) for value in bias)
                            calibrated = calibrate_force_block(
                                stacked,
                                bias_volts=bias,
                                calibration_matrix=calibration.matrix,
                            )
                            self._write_block(
                                writer=writer,
                                raw_block=stacked,
                                ft_block=calibrated,
                                first_sample_index=next_sample_index,
                                device_start_session_s=device_start_session_s,
                                device_start_utc=device_start_utc,
                            )
                            next_sample_index += int(stacked.shape[0])
                            pending_raw.clear()
                            pending_count = 0
                            self.bias_ready_event.set()
                            continue

                        assert self.results.bias_volts is not None
                        bias = np.asarray(self.results.bias_volts, dtype=np.float64)
                        calibrated = calibrate_force_block(
                            block,
                            bias_volts=bias,
                            calibration_matrix=calibration.matrix,
                        )
                        self._write_block(
                            writer=writer,
                            raw_block=block,
                            ft_block=calibrated,
                            first_sample_index=next_sample_index,
                            device_start_session_s=device_start_session_s,
                            device_start_utc=device_start_utc,
                        )
                        next_sample_index += int(block.shape[0])

                    # Preserve a short recording even if STOP occurs before the
                    # requested bias interval is complete.
                    if pending_count > 0:
                        stacked = np.vstack(pending_raw)
                        bias = np.mean(stacked, axis=0)
                        self.results.bias_samples_used = int(stacked.shape[0])
                        self.results.bias_volts = tuple(float(value) for value in bias)
                        calibrated = calibrate_force_block(
                            stacked,
                            bias_volts=bias,
                            calibration_matrix=calibration.matrix,
                        )
                        self._write_block(
                            writer=writer,
                            raw_block=stacked,
                            ft_block=calibrated,
                            first_sample_index=next_sample_index,
                            device_start_session_s=device_start_session_s,
                            device_start_utc=device_start_utc,
                        )
                        next_sample_index += int(stacked.shape[0])
                        self.bias_ready_event.set()

                    termination_reason = "stop_requested"

                self.results.writer_stats = writer.stats

        except Exception as exc:
            termination_reason = "unexpected_exception"
            self.results.runtime_error = exc
        finally:
            if daq is not None:
                try:
                    daq.stop()
                except Exception as exc:
                    self.results.stop_error = exc
            self.results.termination_reason = termination_reason
            self.finished_event.set()

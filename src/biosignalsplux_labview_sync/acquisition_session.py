"""Reusable continuous Biosignalsplux acquisition session."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biosignalsplux_labview_sync.biosignals_device import (
    BiosignalsDeviceError,
    create_biosignals_device_class,
)
from biosignalsplux_labview_sync.emg_writer import (
    EMGCSVWriter,
    EMGWriteStats,
)
from biosignalsplux_labview_sync.live_buffer import EMGLiveBuffer


class AcquisitionSessionError(RuntimeError):
    """Raised when an acquisition session cannot be managed safely."""


@dataclass
class AcquisitionResults:
    """Final and diagnostic information from one acquisition."""

    termination_reason: str = "unknown"
    session_origin: float | None = None
    runtime_error: Exception | None = None
    writer_stats: EMGWriteStats | None = None
    samples_received: int = 0
    live_buffer_error_count: int = 0
    last_live_buffer_error: Exception | None = None
    request_stop_error: Exception | None = None
    stop_error: Exception | None = None
    close_error: Exception | None = None


class BiosignalsAcquisitionSession:
    """Run continuous raw sEMG acquisition in a background thread.

    Acquisition remains active until request_stop() is called, the device loop
    finishes unexpectedly, or an acquisition error occurs.
    """

    def __init__(
        self,
        *,
        plux: Any | None,
        device_address: str,
        sampling_rate_hz: int,
        resolution_bits: int,
        channel_mask: int,
        channel_names: list[str] | tuple[str, ...],
        flush_interval_samples: int,
        csv_path: str | Path,
        live_buffer: EMGLiveBuffer,
        session_origin: float | None = None,
        device_class: type | None = None,
    ) -> None:
        if not isinstance(device_address, str) or not device_address.strip():
            raise AcquisitionSessionError(
                "device_address must be a non-empty string."
            )

        if (
            not isinstance(sampling_rate_hz, int)
            or isinstance(sampling_rate_hz, bool)
            or sampling_rate_hz <= 0
        ):
            raise AcquisitionSessionError(
                "sampling_rate_hz must be a positive integer."
            )

        if (
            not isinstance(resolution_bits, int)
            or isinstance(resolution_bits, bool)
            or resolution_bits <= 0
        ):
            raise AcquisitionSessionError(
                "resolution_bits must be a positive integer."
            )

        if (
            not isinstance(channel_mask, int)
            or isinstance(channel_mask, bool)
            or channel_mask <= 0
        ):
            raise AcquisitionSessionError(
                "channel_mask must be a positive integer."
            )

        validated_channels = tuple(
            str(channel).strip()
            for channel in channel_names
        )

        if not validated_channels or any(
            not channel for channel in validated_channels
        ):
            raise AcquisitionSessionError(
                "channel_names must contain at least one valid name."
            )

        if (
            not isinstance(flush_interval_samples, int)
            or isinstance(flush_interval_samples, bool)
            or flush_interval_samples <= 0
        ):
            raise AcquisitionSessionError(
                "flush_interval_samples must be a positive integer."
            )

        if session_origin is not None and (
            not isinstance(session_origin, (int, float))
            or isinstance(session_origin, bool)
            or not math.isfinite(float(session_origin))
            or float(session_origin) < 0
        ):
            raise AcquisitionSessionError(
                "session_origin must be a finite non-negative number."
            )

        if device_class is None and plux is None:
            raise AcquisitionSessionError(
                "Either plux or device_class must be provided."
            )

        self.plux = plux
        self.device_address = device_address.strip()
        self.sampling_rate_hz = sampling_rate_hz
        self.resolution_bits = resolution_bits
        self.channel_mask = channel_mask
        self.channel_names = validated_channels
        self.flush_interval_samples = flush_interval_samples
        self.csv_path = Path(csv_path)
        self.live_buffer = live_buffer
        self._provided_session_origin = (
            None
            if session_origin is None
            else float(session_origin)
        )
        self._device_class = device_class

        self.stop_event = threading.Event()
        self.finished_event = threading.Event()

        self._device_lock = threading.Lock()
        self._device: object | None = None
        self._thread: threading.Thread | None = None

        self.results = AcquisitionResults()

    @property
    def is_started(self) -> bool:
        return self._thread is not None

    @property
    def is_running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
        )

    @property
    def is_finished(self) -> bool:
        return self.finished_event.is_set()

    def start(self) -> None:
        """Start acquisition in a background thread."""

        if self._thread is not None:
            raise AcquisitionSessionError(
                "This acquisition session has already been started."
            )

        self._thread = threading.Thread(
            target=self._run_worker,
            name="biosignalsplux-acquisition",
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        """Request explicit termination at the next device frame."""

        self.stop_event.set()

        with self._device_lock:
            device = self._device

        if device is not None:
            try:
                device.request_stop()
            except Exception as exc:
                self.results.request_stop_error = exc

    def join(self, timeout: float | None = None) -> bool:
        """Wait for acquisition and return True when the thread stopped."""

        if self._thread is None:
            raise AcquisitionSessionError(
                "Cannot join an acquisition that has not started."
            )

        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def stop_and_join(self, timeout: float = 10.0) -> bool:
        """Request a stop and wait for a clean acquisition shutdown."""

        self.request_stop()
        return self.join(timeout=timeout)

    def _resolve_device_class(self) -> type:
        if self._device_class is not None:
            return self._device_class

        return create_biosignals_device_class(self.plux)

    def _run_worker(self) -> None:
        device = None
        device_started = False
        termination_reason = "unknown"

        session_origin = (
            self._provided_session_origin
            if self._provided_session_origin is not None
            else time.perf_counter()
        )
        self.results.session_origin = session_origin

        try:
            device_class = self._resolve_device_class()

            with EMGCSVWriter(
                path=self.csv_path,
                channel_names=self.channel_names,
                sampling_rate_hz=self.sampling_rate_hz,
                flush_interval_samples=self.flush_interval_samples,
            ) as writer:
                try:
                    device = device_class(
                        self.device_address,
                        writer=writer,
                        session_origin=session_origin,
                        expected_channel_count=len(
                            self.channel_names
                        ),
                        live_buffer=self.live_buffer,
                    )

                    with self._device_lock:
                        self._device = device

                    if self.stop_event.is_set():
                        termination_reason = "stop_before_start"
                    else:
                        device.start(
                            self.sampling_rate_hz,
                            self.channel_mask,
                            self.resolution_bits,
                        )
                        device_started = True

                        if self.stop_event.is_set():
                            device.request_stop()

                        # Blocks until request_stop(), callback error,
                        # disconnection, or another device-level termination.
                        device.loop()

                        if device.callback_error is not None:
                            raise BiosignalsDeviceError(
                                "The PLUX callback stopped after an error: "
                                f"{device.callback_error}"
                            )

                        termination_reason = (
                            "stop_requested"
                            if self.stop_event.is_set()
                            else "device_loop_completed"
                        )

                except Exception as exc:
                    termination_reason = "unexpected_exception"
                    self.results.runtime_error = exc

                finally:
                    self.results.writer_stats = writer.stats

                    if device is not None:
                        self.results.samples_received = int(
                            device.samples_received
                        )
                        self.results.live_buffer_error_count = int(
                            device.live_buffer_error_count
                        )
                        self.results.last_live_buffer_error = (
                            device.last_live_buffer_error
                        )

                    if device is not None and device_started:
                        try:
                            device.stop()
                        except Exception as exc:
                            self.results.stop_error = exc

                    if device is not None:
                        try:
                            device.close()
                        except Exception as exc:
                            self.results.close_error = exc

        except Exception as exc:
            termination_reason = "setup_exception"
            self.results.runtime_error = exc

        finally:
            with self._device_lock:
                self._device = None

            self.results.termination_reason = termination_reason
            self.finished_event.set()
"""Biosignalsplux device adapter for continuous raw sEMG acquisition."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import ModuleType
from typing import Callable

from biosignalsplux_labview_sync.emg_writer import EMGCSVWriter
from biosignalsplux_labview_sync.live_buffer import EMGLiveBuffer


class BiosignalsDeviceError(RuntimeError):
    """Raised when a Biosignalsplux frame cannot be processed safely."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_biosignals_device_class(
    plux: ModuleType,
) -> type:
    """Create a SignalsDev subclass using the dynamically loaded PLUX API."""

    class BiosignalsEMGDevice(plux.SignalsDev):
        """Continuously receive and store Biosignalsplux sEMG frames."""

        def __init__(
            self,
            address: str,
            writer: EMGCSVWriter,
            session_origin: float,
            expected_channel_count: int,
            live_buffer: EMGLiveBuffer | None = None,
            monotonic_clock: Callable[[], float] = time.perf_counter,
            wall_clock: Callable[[], datetime] = _utc_now,
        ) -> None:
            # This unusual initialization form follows the official
            # PLUX Python examples distributed with the local SDK.
            plux.MemoryDev.__init__(address)

            if not isinstance(address, str) or not address.strip():
                raise BiosignalsDeviceError(
                    "The Biosignalsplux address must be a non-empty string."
                )

            if (
                not isinstance(expected_channel_count, int)
                or isinstance(expected_channel_count, bool)
                or expected_channel_count <= 0
            ):
                raise BiosignalsDeviceError(
                    "expected_channel_count must be a positive integer."
                )

            self.address = address
            self.writer = writer
            self.session_origin = session_origin
            self.expected_channel_count = expected_channel_count
            self.live_buffer = live_buffer
            self.monotonic_clock = monotonic_clock
            self.wall_clock = wall_clock

            self.samples_received = 0
            self.stop_requested = False
            self.callback_error: Exception | None = None

            self.live_buffer_error_count = 0
            self.last_live_buffer_error: Exception | None = None

        def request_stop(self) -> None:
            """Request termination at the next acquisition frame."""

            self.stop_requested = True

        def onRawFrame(self, nSeq, data):  # PLUX callback name
            """Store one frame and return True only when acquisition must stop."""

            if self.stop_requested:
                return True

            try:
                values = tuple(data[: self.expected_channel_count])

                if len(values) != self.expected_channel_count:
                    raise BiosignalsDeviceError(
                        "Received fewer channels than expected: "
                        f"expected {self.expected_channel_count}, "
                        f"received {len(values)}."
                    )

                elapsed_s = self.monotonic_clock() - self.session_origin

                if elapsed_s < 0:
                    raise BiosignalsDeviceError(
                        "The monotonic acquisition time became negative."
                    )

                # Raw storage has priority. A storage failure must stop
                # acquisition to avoid silently losing experimental data.
                self.writer.append_sample(
                    device_sequence=int(nSeq),
                    time_monotonic_s=elapsed_s,
                    timestamp_host=self.wall_clock(),
                    channel_values=values,
                )

                # Live visualization is secondary. A graphics or buffer
                # problem must not interrupt raw acquisition.
                if self.live_buffer is not None:
                    try:
                        self.live_buffer.append_sample(
                            time_monotonic_s=elapsed_s,
                            channel_values=values,
                        )
                    except Exception as exc:
                        self.live_buffer_error_count += 1
                        self.last_live_buffer_error = exc

                self.samples_received += 1

            except Exception as exc:
                # Avoid allowing an exception to escape repeatedly through
                # the compiled PLUX callback.
                self.callback_error = exc
                self.stop_requested = True
                return True

            return self.stop_requested

    return BiosignalsEMGDevice
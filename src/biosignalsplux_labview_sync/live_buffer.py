"""Thread-safe circular buffer for live sEMG visualization."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterable


class EMGLiveBufferError(ValueError):
    """Raised when live-buffer settings or samples are invalid."""


@dataclass(frozen=True)
class EMGBufferSnapshot:
    """Immutable copy of the samples currently stored in the buffer."""

    time_monotonic_s: tuple[float, ...]
    channels: tuple[tuple[float, ...], ...]
    total_samples_received: int

    @property
    def buffered_samples(self) -> int:
        """Return the number of samples contained in this snapshot."""

        return len(self.time_monotonic_s)


class EMGLiveBuffer:
    """Store only the most recent sEMG samples for live consumers."""

    def __init__(
        self,
        channel_count: int,
        capacity_samples: int,
    ) -> None:
        if (
            not isinstance(channel_count, int)
            or isinstance(channel_count, bool)
            or channel_count <= 0
        ):
            raise EMGLiveBufferError(
                "channel_count must be a positive integer."
            )

        if (
            not isinstance(capacity_samples, int)
            or isinstance(capacity_samples, bool)
            or capacity_samples <= 0
        ):
            raise EMGLiveBufferError(
                "capacity_samples must be a positive integer."
            )

        self.channel_count = channel_count
        self.capacity_samples = capacity_samples

        self._time_values: deque[float] = deque(
            maxlen=capacity_samples
        )
        self._channel_values: tuple[deque[float], ...] = tuple(
            deque(maxlen=capacity_samples)
            for _ in range(channel_count)
        )

        self._total_samples_received = 0
        self._lock = threading.Lock()

    def append_sample(
        self,
        time_monotonic_s: float,
        channel_values: Iterable[int | float],
    ) -> None:
        """Append one sample without allowing the buffer to grow indefinitely."""

        if (
            not isinstance(time_monotonic_s, (int, float))
            or isinstance(time_monotonic_s, bool)
            or time_monotonic_s < 0
        ):
            raise EMGLiveBufferError(
                "time_monotonic_s must be a non-negative number."
            )

        values = tuple(channel_values)

        if len(values) != self.channel_count:
            raise EMGLiveBufferError(
                "The number of channel values must match channel_count."
            )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            for value in values
        ):
            raise EMGLiveBufferError(
                "Every channel value must be numeric."
            )

        with self._lock:
            self._time_values.append(float(time_monotonic_s))

            for channel_buffer, value in zip(
                self._channel_values,
                values,
            ):
                channel_buffer.append(float(value))

            self._total_samples_received += 1

    def snapshot(self) -> EMGBufferSnapshot:
        """Return a consistent copy for visualization or quality checks."""

        with self._lock:
            return EMGBufferSnapshot(
                time_monotonic_s=tuple(self._time_values),
                channels=tuple(
                    tuple(channel)
                    for channel in self._channel_values
                ),
                total_samples_received=self._total_samples_received,
            )

    def clear(self) -> None:
        """Remove buffered samples without resetting the lifetime counter."""

        with self._lock:
            self._time_values.clear()

            for channel in self._channel_values:
                channel.clear()

    @property
    def total_samples_received(self) -> int:
        """Return the lifetime number of samples passed to the buffer."""

        with self._lock:
            return self._total_samples_received

    @property
    def buffered_samples(self) -> int:
        """Return the number of samples currently retained."""

        with self._lock:
            return len(self._time_values)
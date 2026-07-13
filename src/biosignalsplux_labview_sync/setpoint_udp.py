"""UDP transmission of three joint references to LabVIEW."""

from __future__ import annotations

import math
import socket
from collections.abc import Callable
from dataclasses import dataclass


class SetpointUDPError(RuntimeError):
    """Raised when a LabVIEW setpoint cannot be formatted or sent."""


@dataclass(frozen=True)
class SetpointTransmission:
    """Information about one transmitted UDP setpoint."""

    sequence: int
    payload: bytes
    bytes_sent: int
    destination: tuple[str, int]


def _finite_reference(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise SetpointUDPError(
            f"{field_name} must be a finite number."
        )

    return float(value)


def format_setpoint_payload(
    q_shoulder_rad: float,
    q_elbow_rad: float,
    q_rotation_rad: float,
) -> bytes:
    """Format three references using the existing LabVIEW CSV protocol."""

    shoulder = _finite_reference(
        q_shoulder_rad,
        "q_shoulder_rad",
    )
    elbow = _finite_reference(
        q_elbow_rad,
        "q_elbow_rad",
    )
    rotation = _finite_reference(
        q_rotation_rad,
        "q_rotation_rad",
    )

    message = (
        f"{shoulder:.6f},"
        f"{elbow:.6f},"
        f"{rotation:.6f}\n"
    )

    return message.encode("utf-8")


class LabVIEWSetpointSender:
    """Send three joint-position references to LabVIEW over UDP."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        socket_factory: Callable[..., socket.socket] = socket.socket,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise SetpointUDPError(
                "host must be a non-empty string."
            )

        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
        ):
            raise SetpointUDPError(
                "port must be an integer between 1 and 65535."
            )

        self.host = host.strip()
        self.port = port
        self._sequence = 0
        self._closed = False

        try:
            self._socket = socket_factory(
                socket.AF_INET,
                socket.SOCK_DGRAM,
            )
        except OSError as exc:
            raise SetpointUDPError(
                f"Could not create UDP socket: {exc}"
            ) from exc

    @property
    def next_sequence(self) -> int:
        """Sequence number that will be assigned to the next packet."""

        return self._sequence

    def send(
        self,
        q_shoulder_rad: float,
        q_elbow_rad: float,
        q_rotation_rad: float,
    ) -> SetpointTransmission:
        """Send one three-reference UDP datagram."""

        if self._closed:
            raise SetpointUDPError(
                "Cannot send using a closed UDP sender."
            )

        payload = format_setpoint_payload(
            q_shoulder_rad,
            q_elbow_rad,
            q_rotation_rad,
        )

        destination = (self.host, self.port)

        try:
            bytes_sent = self._socket.sendto(
                payload,
                destination,
            )
        except OSError as exc:
            raise SetpointUDPError(
                f"UDP setpoint transmission failed: {exc}"
            ) from exc

        if bytes_sent != len(payload):
            raise SetpointUDPError(
                "UDP send reported an incomplete datagram."
            )

        transmission = SetpointTransmission(
            sequence=self._sequence,
            payload=payload,
            bytes_sent=bytes_sent,
            destination=destination,
        )

        self._sequence += 1
        return transmission

    def close(self) -> None:
        """Close the UDP socket safely."""

        if self._closed:
            return

        try:
            self._socket.close()
        except OSError as exc:
            raise SetpointUDPError(
                f"Could not close UDP socket: {exc}"
            ) from exc

        self._closed = True

    def __enter__(self) -> "LabVIEWSetpointSender":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
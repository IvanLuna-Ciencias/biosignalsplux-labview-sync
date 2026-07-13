"""Tests for three-reference LabVIEW UDP transmission."""

import math

import pytest

from biosignalsplux_labview_sync.setpoint_udp import (
    LabVIEWSetpointSender,
    SetpointUDPError,
    format_setpoint_payload,
)


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def sendto(
        self,
        data: bytes,
        destination: tuple[str, int],
    ) -> int:
        self.sent.append((data, destination))
        return len(data)

    def close(self) -> None:
        self.closed = True


def make_socket_factory(fake_socket: FakeSocket):
    def factory(address_family, socket_type):
        return fake_socket

    return factory


def test_payload_matches_existing_labview_format() -> None:
    payload = format_setpoint_payload(
        0.0,
        math.pi / 2.0,
        -math.pi / 4.0,
    )

    assert payload == b"0.000000,1.570796,-0.785398\n"


def test_payload_rejects_non_finite_reference() -> None:
    with pytest.raises(
        SetpointUDPError,
        match="q_elbow_rad must be a finite number",
    ):
        format_setpoint_payload(
            0.0,
            float("nan"),
            0.0,
        )


def test_sender_transmits_to_expected_destination() -> None:
    fake_socket = FakeSocket()

    sender = LabVIEWSetpointSender(
        host="172.22.11.2",
        port=5005,
        socket_factory=make_socket_factory(fake_socket),
    )

    first = sender.send(0.0, math.pi / 2.0, 0.0)
    second = sender.send(0.1, 0.2, 0.3)

    assert first.sequence == 0
    assert second.sequence == 1
    assert sender.next_sequence == 2

    assert fake_socket.sent[0] == (
        b"0.000000,1.570796,0.000000\n",
        ("172.22.11.2", 5005),
    )


def test_sender_closes_socket_idempotently() -> None:
    fake_socket = FakeSocket()

    sender = LabVIEWSetpointSender(
        host="127.0.0.1",
        port=5005,
        socket_factory=make_socket_factory(fake_socket),
    )

    sender.close()
    sender.close()

    assert fake_socket.closed


def test_closed_sender_rejects_transmission() -> None:
    fake_socket = FakeSocket()

    sender = LabVIEWSetpointSender(
        host="127.0.0.1",
        port=5005,
        socket_factory=make_socket_factory(fake_socket),
    )

    sender.close()

    with pytest.raises(
        SetpointUDPError,
        match="closed UDP sender",
    ):
        sender.send(0.0, 0.0, 0.0)
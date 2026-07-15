"""Tests for the LabVIEW UDP session coordination channel."""

from __future__ import annotations

import socket
from collections import deque

import pytest

from biosignalsplux_labview_sync.labview_session_udp import (
    LabVIEWSessionUDPClient,
    SessionProtocolError,
    SessionState,
    SessionStateError,
    SessionTimeoutError,
    decode_response,
    encode_command,
    validate_session_id,
)


class FakeSocket:
    def __init__(self, receive_plan) -> None:
        self.receive_plan = deque(receive_plan)
        self.sent: list[bytes] = []
        self.closed = False

    def bind(self, address) -> None:
        self.bound_address = address

    def connect(self, address) -> None:
        self.remote_address = address

    def send(self, payload: bytes) -> int:
        self.sent.append(payload)
        return len(payload)

    def recv(self, size: int) -> bytes:
        del size
        if not self.receive_plan:
            raise socket.timeout("simulated timeout")
        item = self.receive_plan.popleft()
        if isinstance(item, BaseException):
            raise item
        return item

    def settimeout(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s

    def getsockname(self):
        return ("192.168.1.20", 43125)

    def close(self) -> None:
        self.closed = True


class FakeSocketFactory:
    def __init__(self, fake_socket: FakeSocket) -> None:
        self.fake_socket = fake_socket
        self.calls = 0

    def __call__(self, family, socket_type) -> FakeSocket:
        assert family == socket.AF_INET
        assert socket_type == socket.SOCK_DGRAM
        self.calls += 1
        return self.fake_socket


def make_client(receive_plan, *, retries: int = 2):
    fake_socket = FakeSocket(receive_plan)
    factory = FakeSocketFactory(fake_socket)
    client = LabVIEWSessionUDPClient(
        host="172.22.11.2",
        port=5006,
        timeout_s=0.1,
        retries=retries,
        socket_factory=factory,
    )
    return client, fake_socket, factory


@pytest.mark.parametrize(
    "session_id",
    ["P-ABC123_elbow_20260715_120000", "TEST_001", "participant.01", "A"],
)
def test_validate_session_id_accepts_safe_values(session_id) -> None:
    assert validate_session_id(session_id) == session_id


@pytest.mark.parametrize(
    "session_id",
    ["", "contains space", "contains,comma", "contains\nnewline", " leading", "trailing ", "/invalid/path"],
)
def test_validate_session_id_rejects_unsafe_values(session_id) -> None:
    with pytest.raises(ValueError):
        validate_session_id(session_id)


def test_encode_command_adds_newline() -> None:
    assert encode_command("prepare", "TEST_001") == b"PREPARE,TEST_001\n"


def test_decode_response() -> None:
    response = decode_response(b"READY,TEST_001\n")
    assert response.response_type == "READY"
    assert response.session_id == "TEST_001"


def test_decode_response_rejects_invalid_message() -> None:
    with pytest.raises(SessionProtocolError):
        decode_response(b"READY,TEST_001,EXTRA\n")


def test_start_before_ready_is_rejected() -> None:
    client, _, _ = make_client([])
    with pytest.raises(SessionStateError):
        client.start()
    client.close()


def test_full_flow_uses_one_socket() -> None:
    client, fake_socket, factory = make_client(
        [b"READY,TEST_001\n", b"DONE,TEST_001\n"]
    )
    ready = client.prepare("TEST_001")
    client.start()
    done = client.stop()
    assert ready.response_type == "READY"
    assert done.response_type == "DONE"
    assert client.state is SessionState.DONE
    assert fake_socket.sent == [
        b"PREPARE,TEST_001\n",
        b"START,TEST_001\n",
        b"STOP,TEST_001\n",
    ]
    assert factory.calls == 1
    client.close()
    assert fake_socket.closed is True


def test_prepare_retries_after_timeout() -> None:
    client, fake_socket, _ = make_client(
        [socket.timeout("first attempt timed out"), b"READY,TEST_001\n"],
        retries=1,
    )
    response = client.prepare("TEST_001")
    assert response.response_type == "READY"
    assert fake_socket.sent == [b"PREPARE,TEST_001\n", b"PREPARE,TEST_001\n"]
    client.close()


def test_wrong_session_response_is_ignored() -> None:
    client, fake_socket, _ = make_client(
        [b"READY,OTHER_SESSION\n", b"READY,TEST_001\n"]
    )
    response = client.prepare("TEST_001")
    assert response.session_id == "TEST_001"
    assert fake_socket.sent == [b"PREPARE,TEST_001\n"]
    client.close()


def test_stop_timeout_is_reported() -> None:
    client, fake_socket, _ = make_client(
        [b"READY,TEST_001\n", socket.timeout(), socket.timeout(), socket.timeout()],
        retries=2,
    )
    client.prepare("TEST_001")
    client.start()
    with pytest.raises(SessionTimeoutError):
        client.stop()
    assert client.state is SessionState.STOP_UNCONFIRMED
    assert fake_socket.sent.count(b"STOP,TEST_001\n") == 3
    client.close()


def test_context_manager_closes_socket() -> None:
    client, fake_socket, _ = make_client([])
    with client:
        assert client.local_address == ("192.168.1.20", 43125)
    assert client.state is SessionState.CLOSED
    assert fake_socket.closed is True

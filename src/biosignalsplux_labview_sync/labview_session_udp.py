"""UDP session coordination with a continuously running LabVIEW VI.

Independent from the UDP setpoint channel on port 5005.
"""

from __future__ import annotations

import re
import socket
from dataclasses import dataclass
from enum import Enum
from typing import Callable

DEFAULT_SESSION_PORT = 5006
MAX_DATAGRAM_BYTES = 4096
_VALID_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_COMMANDS = frozenset({"PREPARE", "START", "STOP"})
_RESPONSES = frozenset({"READY", "DONE"})


class LabVIEWSessionUDPError(RuntimeError):
    """Base error for the LabVIEW session channel."""


class SessionProtocolError(LabVIEWSessionUDPError):
    """A datagram does not follow the session protocol."""


class SessionStateError(LabVIEWSessionUDPError):
    """A command is not valid in the current client state."""


class SessionTimeoutError(TimeoutError, LabVIEWSessionUDPError):
    """LabVIEW did not return the expected acknowledgment."""


class SessionState(str, Enum):
    CREATED = "CREATED"
    READY = "READY"
    STARTED = "STARTED"
    DONE = "DONE"
    STOP_UNCONFIRMED = "STOP_UNCONFIRMED"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class SessionResponse:
    response_type: str
    session_id: str
    raw_text: str


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a string.")
    if not _VALID_SESSION_ID.fullmatch(session_id):
        raise ValueError(
            "session_id must contain 1 to 128 characters using only "
            "letters, digits, dots, underscores, or hyphens."
        )
    return session_id


def encode_command(command: str, session_id: str) -> bytes:
    normalized = str(command).upper()
    if normalized not in _COMMANDS:
        raise ValueError(f"Unsupported LabVIEW session command: {command!r}.")
    return f"{normalized},{validate_session_id(session_id)}\n".encode("utf-8")


def decode_response(payload: bytes | str) -> SessionResponse:
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SessionProtocolError("LabVIEW response is not valid UTF-8.") from exc
    elif isinstance(payload, str):
        text = payload
    else:
        raise SessionProtocolError("LabVIEW response must be bytes or text.")

    raw_text = text
    text = text.rstrip("\r\n")
    if not text:
        raise SessionProtocolError("LabVIEW returned an empty response.")
    if "\n" in text or "\r" in text:
        raise SessionProtocolError("LabVIEW response contains multiple lines.")
    if text != text.strip():
        raise SessionProtocolError(
            "LabVIEW response contains leading or trailing spaces."
        )

    fields = text.split(",")
    if len(fields) != 2:
        raise SessionProtocolError("Expected RESPONSE,session_id.")

    response_type, session_id = fields
    response_type = response_type.upper()
    if response_type not in _RESPONSES:
        raise SessionProtocolError(
            f"Unsupported LabVIEW response: {response_type!r}."
        )
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise SessionProtocolError(
            f"Invalid session_id in LabVIEW response: {session_id!r}."
        ) from exc

    return SessionResponse(response_type, session_id, raw_text)


class LabVIEWSessionUDPClient:
    """Coordinate one LabVIEW recording session over UDP port 5006."""

    def __init__(
        self,
        *,
        host: str,
        port: int = DEFAULT_SESSION_PORT,
        timeout_s: float = 1.0,
        retries: int = 2,
        bind_host: str = "",
        bind_port: int = 0,
        socket_factory: Callable[..., socket.socket] = socket.socket,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError("LabVIEW host cannot be empty.")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("LabVIEW port must be between 1 and 65535.")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero.")
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("retries must be a non-negative integer.")
        if not isinstance(bind_port, int) or isinstance(bind_port, bool) or not 0 <= bind_port <= 65535:
            raise ValueError("bind_port must be between 0 and 65535.")

        self.host = host.strip()
        self.port = port
        self.timeout_s = float(timeout_s)
        self.retries = retries
        self._socket = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            if bind_host or bind_port:
                self._socket.bind((bind_host, bind_port))
            self._socket.connect((self.host, self.port))
        except Exception:
            self._socket.close()
            raise

        self._state = SessionState.CREATED
        self._session_id: str | None = None

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def local_address(self) -> tuple[str, int]:
        host, port = self._socket.getsockname()
        return str(host), int(port)

    def _require_state(self, expected: SessionState, operation: str) -> None:
        if self._state is not expected:
            raise SessionStateError(
                f"{operation} requires state {expected.value}; "
                f"current state is {self._state.value}."
            )

    def _send(self, command: str, session_id: str) -> None:
        if self._state is SessionState.CLOSED:
            raise SessionStateError("The LabVIEW session socket is closed.")
        self._socket.send(encode_command(command, session_id))

    def _wait_for(self, *, expected_response: str, session_id: str) -> SessionResponse:
        while True:
            self._socket.settimeout(self.timeout_s)
            try:
                payload = self._socket.recv(MAX_DATAGRAM_BYTES)
            except socket.timeout as exc:
                raise SessionTimeoutError(
                    f"Timed out waiting for {expected_response},{session_id}."
                ) from exc

            try:
                response = decode_response(payload)
            except SessionProtocolError:
                continue
            if response.session_id != session_id:
                continue
            if response.response_type != expected_response:
                continue
            return response

    def _exchange_with_retries(
        self,
        *,
        command: str,
        expected_response: str,
        session_id: str,
    ) -> SessionResponse:
        attempts = self.retries + 1
        last_error: SessionTimeoutError | None = None
        for _ in range(attempts):
            self._send(command, session_id)
            try:
                return self._wait_for(
                    expected_response=expected_response,
                    session_id=session_id,
                )
            except SessionTimeoutError as exc:
                last_error = exc

        raise SessionTimeoutError(
            f"No valid {expected_response} response for session {session_id!r} "
            f"after {attempts} attempt(s). Last error: {last_error}"
        ) from last_error

    def prepare(self, session_id: str) -> SessionResponse:
        self._require_state(SessionState.CREATED, "PREPARE")
        valid = validate_session_id(session_id)
        response = self._exchange_with_retries(
            command="PREPARE",
            expected_response="READY",
            session_id=valid,
        )
        self._session_id = valid
        self._state = SessionState.READY
        return response

    def start(self) -> None:
        self._require_state(SessionState.READY, "START")
        assert self._session_id is not None
        self._send("START", self._session_id)
        self._state = SessionState.STARTED

    def stop(self) -> SessionResponse:
        self._require_state(SessionState.STARTED, "STOP")
        assert self._session_id is not None
        try:
            response = self._exchange_with_retries(
                command="STOP",
                expected_response="DONE",
                session_id=self._session_id,
            )
        except SessionTimeoutError:
            self._state = SessionState.STOP_UNCONFIRMED
            raise
        self._state = SessionState.DONE
        return response

    def close(self) -> None:
        if self._state is SessionState.CLOSED:
            return
        try:
            self._socket.close()
        finally:
            self._state = SessionState.CLOSED

    def __enter__(self) -> "LabVIEWSessionUDPClient":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

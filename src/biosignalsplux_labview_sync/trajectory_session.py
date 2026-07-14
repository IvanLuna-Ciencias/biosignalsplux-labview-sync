"""Background execution and logging of LabVIEW reference trajectories."""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from biosignalsplux_labview_sync.setpoint_udp import (
    LabVIEWSetpointSender,
    SetpointUDPError,
)
from biosignalsplux_labview_sync.trajectory import (
    CosineTrajectory,
    JointReferences,
    smooth_transition_deg,
)


class TrajectorySessionError(RuntimeError):
    """Raised when a trajectory session cannot be executed safely."""


ReferenceTuple = tuple[float, float, float]


@dataclass
class TrajectorySessionResults:
    """Summary and diagnostics for one trajectory execution."""

    termination_reason: str = "unknown"
    runtime_error: Exception | None = None
    close_error: Exception | None = None
    setpoints_generated: int = 0
    udp_packets_sent: int = 0
    pause_count: int = 0
    final_state: str = "IDLE"


class TrajectoryExecutionSession:
    """Execute one trajectory while acquisition remains independent."""

    def __init__(
        self,
        *,
        trajectory: CosineTrajectory,
        setpoint_rate_hz: float,
        move_to_start_s: float,
        return_duration_s: float,
        neutral_hold_s: float,
        trajectories_csv: str | Path,
        events_csv: str | Path,
        session_origin: float,
        labview_host: str,
        labview_port: int,
        udp_enabled: bool,
        sender_factory: type[
            LabVIEWSetpointSender
        ] = LabVIEWSetpointSender,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(trajectory, CosineTrajectory):
            raise TrajectorySessionError(
                "trajectory must be a CosineTrajectory instance."
            )

        self.trajectory = trajectory
        self.setpoint_rate_hz = self._positive_number(
            setpoint_rate_hz,
            "setpoint_rate_hz",
        )
        self.move_to_start_s = self._positive_number(
            move_to_start_s,
            "move_to_start_s",
        )
        self.return_duration_s = self._positive_number(
            return_duration_s,
            "return_duration_s",
        )
        self.neutral_hold_s = self._non_negative_number(
            neutral_hold_s,
            "neutral_hold_s",
        )

        if (
            not isinstance(session_origin, (int, float))
            or isinstance(session_origin, bool)
            or not math.isfinite(float(session_origin))
            or float(session_origin) < 0
        ):
            raise TrajectorySessionError(
                "session_origin must be a finite non-negative number."
            )

        if not isinstance(udp_enabled, bool):
            raise TrajectorySessionError(
                "udp_enabled must be a boolean."
            )

        self.trajectories_csv = Path(trajectories_csv)
        self.events_csv = Path(events_csv)
        self.session_origin = float(session_origin)

        self.labview_host = labview_host
        self.labview_port = labview_port
        self.udp_enabled = udp_enabled

        self._sender_factory = sender_factory
        self._clock = clock
        self._sleeper = sleeper

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._finished_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._state_lock = threading.Lock()
        self._state = "IDLE"
        self._current_references_deg = self._neutral_references()

        self.results = TrajectorySessionResults()

    @staticmethod
    def _positive_number(
        value: object,
        field_name: str,
    ) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise TrajectorySessionError(
                f"{field_name} must be greater than zero."
            )

        return float(value)

    @staticmethod
    def _non_negative_number(
        value: object,
        field_name: str,
    ) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise TrajectorySessionError(
                f"{field_name} cannot be negative."
            )

        return float(value)

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
        return self._finished_event.is_set()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def current_references_deg(self) -> ReferenceTuple:
        with self._state_lock:
            return self._current_references_deg

    def _set_state(self, state: str) -> None:
        with self._state_lock:
            self._state = state
            self.results.final_state = state

    def _set_current_references(
        self,
        references_deg: ReferenceTuple,
    ) -> None:
        with self._state_lock:
            self._current_references_deg = references_deg

    def start(self) -> None:
        """Start trajectory execution in a background thread."""

        if self._thread is not None:
            raise TrajectorySessionError(
                "This trajectory session has already been started."
            )

        self._thread = threading.Thread(
            target=self._run,
            name="labview-trajectory",
            daemon=True,
        )
        self._thread.start()

    def toggle_pause(self) -> bool:
        """Pause or resume RUNNING trajectory time."""

        if self.state not in {"RUNNING", "PAUSED"}:
            return False

        if self._pause_event.is_set():
            self._pause_event.clear()
        else:
            self._pause_event.set()

        return self._pause_event.is_set()

    def request_stop(self) -> None:
        """Request a normal controlled return to neutral."""

        self._stop_event.set()
        self._pause_event.clear()

    def join(self, timeout: float | None = None) -> bool:
        if self._thread is None:
            raise TrajectorySessionError(
                "Cannot join a trajectory that has not started."
            )

        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def _neutral_references(self) -> ReferenceTuple:
        neutral = self.trajectory.neutral_deg

        return (
            float(neutral["shoulder"]),
            float(neutral["elbow"]),
            float(neutral["rotation"]),
        )

    @staticmethod
    def _to_radians(
        references_deg: ReferenceTuple,
    ) -> ReferenceTuple:
        return tuple(
            math.radians(value)
            for value in references_deg
        )

    @staticmethod
    def _transition_references(
        initial_deg: ReferenceTuple,
        target_deg: ReferenceTuple,
        elapsed_s: float,
        duration_s: float,
    ) -> ReferenceTuple:
        return tuple(
            smooth_transition_deg(
                initial_deg=initial,
                target_deg=target,
                elapsed_s=elapsed_s,
                duration_s=duration_s,
            )
            for initial, target in zip(
                initial_deg,
                target_deg,
            )
        )

    def _session_time_s(self) -> float:
        return max(
            0.0,
            self._clock() - self.session_origin,
        )

    def _wait_for_tick(
        self,
        next_tick: float,
        period_s: float,
    ) -> tuple[float, float]:
        now = self._clock()

        if next_tick > now:
            self._sleeper(next_tick - now)

        now = self._clock()
        next_tick += period_s

        if now - next_tick > 5.0 * period_s:
            next_tick = now + period_s

        return now, next_tick

    def _write_event(
        self,
        writer: csv.writer,
        event: str,
        details: dict[str, object] | None = None,
    ) -> None:
        writer.writerow(
            [
                f"{self._session_time_s():.9f}",
                event,
                json.dumps(
                    details or {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ]
        )

    def _emit_setpoint(
        self,
        *,
        sender: LabVIEWSetpointSender | None,
        writer: csv.writer,
        sample_index: int,
        trajectory_time_s: float,
        references_deg: ReferenceTuple,
    ) -> None:
        references_rad = self._to_radians(
            references_deg
        )

        udp_sequence: int | str = ""

        if sender is not None:
            transmission = sender.send(
                *references_rad
            )
            udp_sequence = transmission.sequence
            self.results.udp_packets_sent += 1

        writer.writerow(
            [
                sample_index,
                f"{self._session_time_s():.9f}",
                f"{trajectory_time_s:.9f}",
                self.state,
                self.trajectory.joint,
                f"{references_rad[0]:.9f}",
                f"{references_rad[1]:.9f}",
                f"{references_rad[2]:.9f}",
                udp_sequence,
            ]
        )

        self.results.setpoints_generated += 1
        self._set_current_references(
            references_deg
        )

    def _run_transition(
        self,
        *,
        sender: LabVIEWSetpointSender | None,
        setpoint_writer: csv.writer,
        initial_deg: ReferenceTuple,
        target_deg: ReferenceTuple,
        duration_s: float,
        trajectory_time_s: float,
        sample_index: int,
        interruptible: bool,
    ) -> tuple[ReferenceTuple, int, bool]:
        period_s = 1.0 / self.setpoint_rate_hz
        started_at = self._clock()
        next_tick = started_at
        current = initial_deg

        while True:
            now, next_tick = self._wait_for_tick(
                next_tick,
                period_s,
            )
            if interruptible and self._stop_event.is_set():
                return current, sample_index, True

            elapsed_s = min(
                now - started_at,
                duration_s,
            )

            current = self._transition_references(
                initial_deg,
                target_deg,
                elapsed_s,
                duration_s,
            )

            self._emit_setpoint(
                sender=sender,
                writer=setpoint_writer,
                sample_index=sample_index,
                trajectory_time_s=trajectory_time_s,
                references_deg=current,
            )
            sample_index += 1

            if elapsed_s >= duration_s:
                return current, sample_index, False

    def _run_active_trajectory(
        self,
        *,
        sender: LabVIEWSetpointSender | None,
        setpoint_writer: csv.writer,
        event_writer: csv.writer,
        sample_index: int,
    ) -> tuple[ReferenceTuple, int, bool]:
        period_s = 1.0 / self.setpoint_rate_hz
        started_at = self._clock()
        next_tick = started_at

        pause_started_at = 0.0
        total_paused_s = 0.0
        pause_was_active = False
        trajectory_time_s = 0.0

        current = self.trajectory.references_deg(
            0.0
        ).as_degrees_tuple()

        while True:
            now, next_tick = self._wait_for_tick(
                next_tick,
                period_s,
            )

            if self._stop_event.is_set():
                self._write_event(
                    event_writer,
                    "STOP_REQUESTED",
                    {
                        "trajectory_time_s": trajectory_time_s,
                    },
                )
                return current, sample_index, True

            paused = self._pause_event.is_set()

            if paused and not pause_was_active:
                pause_started_at = now
                pause_was_active = True
                self.results.pause_count += 1
                self._set_state("PAUSED")
                self._write_event(
                    event_writer,
                    "PAUSE",
                    {
                        "trajectory_time_s": trajectory_time_s,
                    },
                )

            elif not paused and pause_was_active:
                total_paused_s += (
                    now - pause_started_at
                )
                pause_was_active = False
                self._set_state("RUNNING")
                self._write_event(
                    event_writer,
                    "RESUME",
                    {
                        "trajectory_time_s": trajectory_time_s,
                    },
                )

            if not paused:
                trajectory_time_s = min(
                    now
                    - started_at
                    - total_paused_s,
                    self.trajectory.duration_s,
                )

                current = self.trajectory.references_deg(
                    trajectory_time_s
                ).as_degrees_tuple()

            self._emit_setpoint(
                sender=sender,
                writer=setpoint_writer,
                sample_index=sample_index,
                trajectory_time_s=trajectory_time_s,
                references_deg=current,
            )
            sample_index += 1

            if (
                not paused
                and trajectory_time_s
                >= self.trajectory.duration_s
            ):
                return current, sample_index, False

    def _run(self) -> None:
        sender: LabVIEWSetpointSender | None = None
        setpoint_file = None
        event_file = None

        current = self._neutral_references()
        sample_index = 0

        try:
            self.trajectories_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.events_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            setpoint_file = self.trajectories_csv.open(
                "w",
                newline="",
                encoding="utf-8",
            )
            event_file = self.events_csv.open(
                "w",
                newline="",
                encoding="utf-8",
            )

            setpoint_writer = csv.writer(
                setpoint_file
            )
            event_writer = csv.writer(
                event_file
            )

            setpoint_writer.writerow(
                [
                    "sample_index",
                    "session_time_s",
                    "trajectory_time_s",
                    "state",
                    "active_joint",
                    "q_shoulder_ref_rad",
                    "q_elbow_ref_rad",
                    "q_rotation_ref_rad",
                    "udp_sequence",
                ]
            )
            event_writer.writerow(
                [
                    "session_time_s",
                    "event",
                    "details_json",
                ]
            )

            if self.udp_enabled:
                sender = self._sender_factory(
                    host=self.labview_host,
                    port=self.labview_port,
                )

            self._write_event(
                event_writer,
                "TRAJECTORY_SESSION_STARTED",
                {
                    "joint": self.trajectory.joint,
                    "udp_enabled": self.udp_enabled,
                },
            )

            start_references = (
                self.trajectory.references_deg(
                    0.0
                ).as_degrees_tuple()
            )

            self._set_state("MOVE_TO_START")
            self._write_event(
                event_writer,
                "MOVE_TO_START",
            )

            (
                current,
                sample_index,
                transition_stopped,
            ) = self._run_transition(
                sender=sender,
                setpoint_writer=setpoint_writer,
                initial_deg=current,
                target_deg=start_references,
                duration_s=self.move_to_start_s,
                trajectory_time_s=0.0,
                sample_index=sample_index,
                interruptible=True,
            )

            stopped_early = (
                transition_stopped
                or self._stop_event.is_set()
            )

            if stopped_early:
                self._write_event(
                    event_writer,
                    "STOP_REQUESTED",
                    {"phase": "MOVE_TO_START"},
                )

            if not stopped_early:
                self._set_state("RUNNING")
                self._write_event(
                    event_writer,
                    "TRAJECTORY_STARTED",
                )

                (
                    current,
                    sample_index,
                    stopped_early,
                ) = self._run_active_trajectory(
                    sender=sender,
                    setpoint_writer=setpoint_writer,
                    event_writer=event_writer,
                    sample_index=sample_index,
                )

                if not stopped_early:
                    self._write_event(
                        event_writer,
                        "TRAJECTORY_DONE",
                    )

            self._set_state("RETURNING")
            self._write_event(
                event_writer,
                "RETURN_STARTED",
            )

            (
                current,
                sample_index,
                _,
            ) = self._run_transition(
                sender=sender,
                setpoint_writer=setpoint_writer,
                initial_deg=current,
                target_deg=self._neutral_references(),
                duration_s=self.return_duration_s,
                trajectory_time_s=(
                    self.trajectory.duration_s
                    if not stopped_early
                    else 0.0
                ),
                sample_index=sample_index,
                interruptible=False,
            )

            self._set_state("HOLDING_NEUTRAL")
            self._write_event(
                event_writer,
                "RETURN_DONE",
            )

            if self.neutral_hold_s > 0:
                hold_started_at = self._clock()
                period_s = 1.0 / self.setpoint_rate_hz
                next_tick = hold_started_at

                while True:
                    now, next_tick = self._wait_for_tick(
                        next_tick,
                        period_s,
                    )

                    self._emit_setpoint(
                        sender=sender,
                        writer=setpoint_writer,
                        sample_index=sample_index,
                        trajectory_time_s=0.0,
                        references_deg=(
                            self._neutral_references()
                        ),
                    )
                    sample_index += 1

                    if (
                        now - hold_started_at
                        >= self.neutral_hold_s
                    ):
                        break

            self._set_state("STOPPED")
            self._write_event(
                event_writer,
                "TRAJECTORY_SESSION_FINISHED",
            )

            self.results.termination_reason = (
                "stop_requested"
                if stopped_early
                else "trajectory_completed"
            )

        except Exception as exc:
            self.results.runtime_error = exc
            self.results.termination_reason = "runtime_error"
            self._set_state("FAULT")

            if event_file is not None:
                try:
                    csv.writer(event_file).writerow(
                        [
                            f"{self._session_time_s():.9f}",
                            "FAULT",
                            json.dumps(
                                {"error": str(exc)},
                                ensure_ascii=False,
                            ),
                        ]
                    )
                except Exception:
                    pass

        finally:
            if setpoint_file is not None:
                try:
                    setpoint_file.flush()
                    setpoint_file.close()
                except Exception:
                    pass

            if event_file is not None:
                try:
                    event_file.flush()
                    event_file.close()
                except Exception:
                    pass

            if sender is not None:
                try:
                    sender.close()
                except SetpointUDPError as exc:
                    self.results.close_error = exc

            self.results.final_state = self.state
            self._finished_event.set()
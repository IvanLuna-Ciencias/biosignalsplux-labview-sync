"""Tests for background LabVIEW trajectory execution."""

from __future__ import annotations

import csv

from biosignalsplux_labview_sync.trajectory import (
    CosineTrajectory,
)
from biosignalsplux_labview_sync.trajectory_session import (
    TrajectoryExecutionSession,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, duration_s: float) -> None:
        self.value += max(0.0, duration_s)


class FakeTransmission:
    def __init__(self, sequence: int) -> None:
        self.sequence = sequence


class FakeSender:
    instances = []

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.sent = []
        self.closed = False
        type(self).instances.append(self)

    def send(self, shoulder, elbow, rotation):
        sequence = len(self.sent)
        self.sent.append(
            (shoulder, elbow, rotation)
        )
        return FakeTransmission(sequence)

    def close(self) -> None:
        self.closed = True


def make_trajectory() -> CosineTrajectory:
    return CosineTrajectory(
        joint="elbow",
        start_deg=0.0,
        end_deg=20.0,
        frequency_hz=1.0,
        cycles=1,
    )


def test_trajectory_session_writes_setpoints_and_events(
    tmp_path,
) -> None:
    FakeSender.instances.clear()
    clock = FakeClock()

    session = TrajectoryExecutionSession(
        trajectory=make_trajectory(),
        setpoint_rate_hz=20.0,
        move_to_start_s=0.1,
        return_duration_s=0.1,
        neutral_hold_s=0.1,
        trajectories_csv=tmp_path / "trajectories.csv",
        events_csv=tmp_path / "events.csv",
        session_origin=100.0,
        labview_host="127.0.0.1",
        labview_port=5005,
        udp_enabled=True,
        sender_factory=FakeSender,
        clock=clock,
        sleeper=clock.sleep,
    )

    session.start()

    assert session.join(timeout=2.0)
    assert session.is_finished
    assert session.state == "STOPPED"
    assert (
        session.results.termination_reason
        == "trajectory_completed"
    )
    assert session.results.setpoints_generated > 0
    assert session.results.udp_packets_sent > 0

    sender = FakeSender.instances[0]

    assert sender.host == "127.0.0.1"
    assert sender.port == 5005
    assert sender.closed
    assert len(sender.sent) == (
        session.results.udp_packets_sent
    )

    with (
        tmp_path / "trajectories.csv"
    ).open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows
    assert rows[0]["active_joint"] == "elbow"
    assert rows[-1]["state"] == "HOLDING_NEUTRAL"
    assert float(
        rows[-1]["q_elbow_ref_rad"]
    ) == 0.0

    with (
        tmp_path / "events.csv"
    ).open(encoding="utf-8") as file:
        event_rows = list(csv.DictReader(file))

    events = {
        row["event"]
        for row in event_rows
    }

    assert "TRAJECTORY_STARTED" in events
    assert "TRAJECTORY_DONE" in events
    assert "RETURN_STARTED" in events
    assert "RETURN_DONE" in events


def test_dry_run_logs_without_creating_udp_sender(
    tmp_path,
) -> None:
    FakeSender.instances.clear()
    clock = FakeClock()

    session = TrajectoryExecutionSession(
        trajectory=make_trajectory(),
        setpoint_rate_hz=20.0,
        move_to_start_s=0.1,
        return_duration_s=0.1,
        neutral_hold_s=0.0,
        trajectories_csv=tmp_path / "trajectories.csv",
        events_csv=tmp_path / "events.csv",
        session_origin=100.0,
        labview_host="127.0.0.1",
        labview_port=5005,
        udp_enabled=False,
        sender_factory=FakeSender,
        clock=clock,
        sleeper=clock.sleep,
    )

    session.start()

    assert session.join(timeout=2.0)
    assert session.results.udp_packets_sent == 0
    assert session.results.setpoints_generated > 0
    assert FakeSender.instances == []


def test_trajectory_session_cannot_start_twice(
    tmp_path,
) -> None:
    clock = FakeClock()

    session = TrajectoryExecutionSession(
        trajectory=make_trajectory(),
        setpoint_rate_hz=20.0,
        move_to_start_s=0.1,
        return_duration_s=0.1,
        neutral_hold_s=0.0,
        trajectories_csv=tmp_path / "trajectories.csv",
        events_csv=tmp_path / "events.csv",
        session_origin=100.0,
        labview_host="127.0.0.1",
        labview_port=5005,
        udp_enabled=False,
        clock=clock,
        sleeper=clock.sleep,
    )

    session.start()

    try:
        session.start()
    except Exception as exc:
        assert "already been started" in str(exc)
    else:
        raise AssertionError(
            "Trajectory session started twice."
        )

    assert session.join(timeout=2.0)
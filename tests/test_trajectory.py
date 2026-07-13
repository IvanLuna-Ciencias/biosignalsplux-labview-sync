"""Tests for three-DOF cosine trajectory generation."""

import math

import pytest

from biosignalsplux_labview_sync.trajectory import (
    CosineTrajectory,
    TrajectoryError,
    smooth_transition_deg,
)


def make_elbow_trajectory() -> CosineTrajectory:
    return CosineTrajectory(
        joint="elbow",
        start_deg=0.0,
        end_deg=90.0,
        frequency_hz=0.1,
        cycles=5,
    )


def test_duration_velocity_and_acceleration() -> None:
    trajectory = make_elbow_trajectory()

    assert trajectory.duration_s == pytest.approx(50.0)
    assert trajectory.peak_velocity_deg_s == pytest.approx(
        math.pi * 0.1 * 90.0
    )
    assert trajectory.peak_acceleration_deg_s2 == pytest.approx(
        2.0 * math.pi**2 * 0.1**2 * 90.0
    )


def test_cosine_trajectory_reaches_end_at_half_cycle() -> None:
    trajectory = make_elbow_trajectory()

    assert trajectory.active_position_deg(0.0) == pytest.approx(0.0)
    assert trajectory.active_position_deg(5.0) == pytest.approx(90.0)
    assert trajectory.active_position_deg(10.0) == pytest.approx(0.0)


def test_trajectory_finishes_at_start_position() -> None:
    trajectory = make_elbow_trajectory()

    assert trajectory.active_position_deg(
        trajectory.duration_s
    ) == pytest.approx(0.0)


def test_elbow_trajectory_preserves_other_neutral_joints() -> None:
    trajectory = make_elbow_trajectory()

    references = trajectory.references_deg(5.0)

    assert references.shoulder_deg == pytest.approx(0.0)
    assert references.elbow_deg == pytest.approx(90.0)
    assert references.rotation_deg == pytest.approx(0.0)


def test_radian_references_follow_labview_order() -> None:
    trajectory = make_elbow_trajectory()

    shoulder, elbow, rotation = trajectory.references_rad(5.0)

    assert shoulder == pytest.approx(0.0)
    assert elbow == pytest.approx(math.pi / 2.0)
    assert rotation == pytest.approx(0.0)


def test_invalid_joint_is_rejected() -> None:
    with pytest.raises(
        TrajectoryError,
        match="joint must be one of",
    ):
        CosineTrajectory(
            joint="wrist",
            start_deg=0.0,
            end_deg=20.0,
            frequency_hz=0.1,
            cycles=1,
        )


def test_position_outside_joint_limits_is_rejected() -> None:
    with pytest.raises(
        TrajectoryError,
        match="end_deg is outside",
    ):
        CosineTrajectory(
            joint="elbow",
            start_deg=0.0,
            end_deg=120.0,
            frequency_hz=0.1,
            cycles=1,
        )


def test_smooth_transition_has_expected_endpoints() -> None:
    assert smooth_transition_deg(
        initial_deg=60.0,
        target_deg=0.0,
        elapsed_s=0.0,
        duration_s=3.0,
    ) == pytest.approx(60.0)

    assert smooth_transition_deg(
        initial_deg=60.0,
        target_deg=0.0,
        elapsed_s=1.5,
        duration_s=3.0,
    ) == pytest.approx(30.0)

    assert smooth_transition_deg(
        initial_deg=60.0,
        target_deg=0.0,
        elapsed_s=3.0,
        duration_s=3.0,
    ) == pytest.approx(0.0)
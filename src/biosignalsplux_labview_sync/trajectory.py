"""Three-DOF reference generation for LabVIEW trajectory tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


JOINT_NAMES = ("shoulder", "elbow", "rotation")

# Initial software limits inherited from the validated 3-DOF demo.
# They are not a substitute for LabVIEW/cRIO safety limits.
DEFAULT_LIMITS_DEG = {
    "shoulder": (-20.0, 140.0),
    "elbow": (0.0, 100.0),
    "rotation": (-40.0, 40.0),
}

DEFAULT_NEUTRAL_DEG = {
    "shoulder": 0.0,
    "elbow": 0.0,
    "rotation": 0.0,
}


class TrajectoryError(ValueError):
    """Raised when trajectory parameters are invalid."""


@dataclass(frozen=True)
class JointReferences:
    """Three joint references expressed in degrees."""

    shoulder_deg: float
    elbow_deg: float
    rotation_deg: float

    def as_degrees_tuple(self) -> tuple[float, float, float]:
        return (
            float(self.shoulder_deg),
            float(self.elbow_deg),
            float(self.rotation_deg),
        )

    def as_radians_tuple(self) -> tuple[float, float, float]:
        return tuple(
            math.radians(value)
            for value in self.as_degrees_tuple()
        )


def _finite_number(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise TrajectoryError(
            f"{field_name} must be a finite number."
        )

    return float(value)


def _validate_named_values(
    values: Mapping[str, object],
    field_name: str,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TrajectoryError(
            f"{field_name} must be a mapping."
        )

    missing = set(JOINT_NAMES) - set(values)
    extra = set(values) - set(JOINT_NAMES)

    if missing:
        raise TrajectoryError(
            f"{field_name} is missing: {', '.join(sorted(missing))}."
        )

    if extra:
        raise TrajectoryError(
            f"{field_name} has unsupported joints: "
            f"{', '.join(sorted(extra))}."
        )

    return {
        joint: _finite_number(
            values[joint],
            f"{field_name}.{joint}",
        )
        for joint in JOINT_NAMES
    }


def _validate_limits(
    limits_deg: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    if not isinstance(limits_deg, Mapping):
        raise TrajectoryError(
            "limits_deg must be a mapping."
        )

    missing = set(JOINT_NAMES) - set(limits_deg)
    extra = set(limits_deg) - set(JOINT_NAMES)

    if missing:
        raise TrajectoryError(
            "limits_deg is missing: "
            + ", ".join(sorted(missing))
            + "."
        )

    if extra:
        raise TrajectoryError(
            "limits_deg has unsupported joints: "
            + ", ".join(sorted(extra))
            + "."
        )

    validated: dict[str, tuple[float, float]] = {}

    for joint in JOINT_NAMES:
        limits = limits_deg[joint]

        if (
            not isinstance(limits, (tuple, list))
            or len(limits) != 2
        ):
            raise TrajectoryError(
                f"limits_deg.{joint} must contain [min, max]."
            )

        minimum = _finite_number(
            limits[0],
            f"limits_deg.{joint}.minimum",
        )
        maximum = _finite_number(
            limits[1],
            f"limits_deg.{joint}.maximum",
        )

        if minimum >= maximum:
            raise TrajectoryError(
                f"limits_deg.{joint} minimum must be less than maximum."
            )

        validated[joint] = (minimum, maximum)

    return validated


@dataclass(frozen=True)
class CosineTrajectory:
    """Cosine trajectory for one active joint while retaining three references."""

    joint: str
    start_deg: float
    end_deg: float
    frequency_hz: float
    cycles: int
    neutral_deg: Mapping[str, float] | None = None
    limits_deg: Mapping[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.joint not in JOINT_NAMES:
            raise TrajectoryError(
                f"joint must be one of: {', '.join(JOINT_NAMES)}."
            )

        start_deg = _finite_number(
            self.start_deg,
            "start_deg",
        )
        end_deg = _finite_number(
            self.end_deg,
            "end_deg",
        )
        frequency_hz = _finite_number(
            self.frequency_hz,
            "frequency_hz",
        )

        if frequency_hz <= 0:
            raise TrajectoryError(
                "frequency_hz must be greater than zero."
            )

        if (
            not isinstance(self.cycles, int)
            or isinstance(self.cycles, bool)
            or self.cycles <= 0
        ):
            raise TrajectoryError(
                "cycles must be a positive integer."
            )

        if start_deg == end_deg:
            raise TrajectoryError(
                "start_deg and end_deg must be different."
            )

        neutral = _validate_named_values(
            self.neutral_deg or DEFAULT_NEUTRAL_DEG,
            "neutral_deg",
        )
        limits = _validate_limits(
            self.limits_deg or DEFAULT_LIMITS_DEG
        )

        for joint in JOINT_NAMES:
            minimum, maximum = limits[joint]
            neutral_value = neutral[joint]

            if not minimum <= neutral_value <= maximum:
                raise TrajectoryError(
                    f"neutral_deg.{joint} is outside its limits."
                )

        active_minimum, active_maximum = limits[self.joint]

        if not active_minimum <= start_deg <= active_maximum:
            raise TrajectoryError(
                "start_deg is outside the selected joint limits."
            )

        if not active_minimum <= end_deg <= active_maximum:
            raise TrajectoryError(
                "end_deg is outside the selected joint limits."
            )

        object.__setattr__(self, "start_deg", start_deg)
        object.__setattr__(self, "end_deg", end_deg)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "neutral_deg", neutral)
        object.__setattr__(self, "limits_deg", limits)

    @property
    def duration_s(self) -> float:
        """Total duration of all complete cycles."""

        return self.cycles / self.frequency_hz

    @property
    def peak_velocity_deg_s(self) -> float:
        """Maximum theoretical active-joint speed."""

        excursion_deg = abs(self.end_deg - self.start_deg)
        return math.pi * self.frequency_hz * excursion_deg

    @property
    def peak_acceleration_deg_s2(self) -> float:
        """Maximum theoretical active-joint acceleration."""

        excursion_deg = abs(self.end_deg - self.start_deg)
        return (
            2.0
            * math.pi**2
            * self.frequency_hz**2
            * excursion_deg
        )

    def active_position_deg(self, trajectory_time_s: float) -> float:
        """Return active-joint position at trajectory time."""

        time_s = _finite_number(
            trajectory_time_s,
            "trajectory_time_s",
        )

        if time_s < 0:
            raise TrajectoryError(
                "trajectory_time_s cannot be negative."
            )

        bounded_time_s = min(time_s, self.duration_s)

        phase = (
            2.0
            * math.pi
            * self.frequency_hz
            * bounded_time_s
        )

        return self.start_deg + (
            self.end_deg - self.start_deg
        ) * (1.0 - math.cos(phase)) / 2.0

    def references_deg(
        self,
        trajectory_time_s: float,
    ) -> JointReferences:
        """Return shoulder, elbow, and rotation references."""

        active_position = self.active_position_deg(
            trajectory_time_s
        )

        values = dict(self.neutral_deg)
        values[self.joint] = active_position

        return JointReferences(
            shoulder_deg=values["shoulder"],
            elbow_deg=values["elbow"],
            rotation_deg=values["rotation"],
        )

    def references_rad(
        self,
        trajectory_time_s: float,
    ) -> tuple[float, float, float]:
        """Return the three references in LabVIEW-compatible radians."""

        return self.references_deg(
            trajectory_time_s
        ).as_radians_tuple()


def smooth_transition_deg(
    *,
    initial_deg: float,
    target_deg: float,
    elapsed_s: float,
    duration_s: float,
) -> float:
    """Return a smooth half-cosine transition between two positions."""

    initial = _finite_number(
        initial_deg,
        "initial_deg",
    )
    target = _finite_number(
        target_deg,
        "target_deg",
    )
    elapsed = _finite_number(
        elapsed_s,
        "elapsed_s",
    )
    duration = _finite_number(
        duration_s,
        "duration_s",
    )

    if elapsed < 0:
        raise TrajectoryError(
            "elapsed_s cannot be negative."
        )

    if duration <= 0:
        raise TrajectoryError(
            "duration_s must be greater than zero."
        )

    progress = min(elapsed / duration, 1.0)

    return initial + (
        target - initial
    ) * (1.0 - math.cos(math.pi * progress)) / 2.0
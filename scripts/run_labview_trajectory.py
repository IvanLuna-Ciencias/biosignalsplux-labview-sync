#!/usr/bin/env python
"""Run a three-reference cosine trajectory for the existing LabVIEW VI."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence

try:
    import msvcrt
except ImportError as exc:
    raise RuntimeError(
        "This interactive trajectory script currently requires Windows."
    ) from exc

from biosignalsplux_labview_sync.setpoint_udp import (
    LabVIEWSetpointSender,
    SetpointUDPError,
)
from biosignalsplux_labview_sync.trajectory import (
    CosineTrajectory,
    JointReferences,
    TrajectoryError,
    smooth_transition_deg,
)


ReferenceTuple = tuple[float, float, float]


def positive_float(value: str) -> float:
    parsed = float(value)

    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be a finite number greater than zero."
        )

    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)

    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            "Value must be a finite number greater than or equal to zero."
        )

    return parsed


def positive_integer(value: str) -> int:
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be a positive integer."
        )

    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one active-joint cosine trajectory while continuously "
            "sending shoulder, elbow, and rotation references to LabVIEW."
        )
    )

    parser.add_argument(
        "--joint",
        required=True,
        choices=("shoulder", "elbow", "rotation"),
        help="Joint that executes the trajectory.",
    )
    parser.add_argument(
        "--start-deg",
        required=True,
        type=float,
        help="Initial active-joint position in degrees.",
    )
    parser.add_argument(
        "--end-deg",
        required=True,
        type=float,
        help="Opposite trajectory endpoint in degrees.",
    )
    parser.add_argument(
        "--frequency-hz",
        required=True,
        type=positive_float,
        help="Cosine trajectory frequency.",
    )
    parser.add_argument(
        "--cycles",
        required=True,
        type=positive_integer,
        help="Number of complete cosine cycles.",
    )
    parser.add_argument(
        "--host",
        default="172.22.11.2",
        help="LabVIEW UDP destination address.",
    )
    parser.add_argument(
        "--port",
        default=5005,
        type=int,
        help="LabVIEW UDP destination port.",
    )
    parser.add_argument(
        "--rate-hz",
        default=50.0,
        type=positive_float,
        help="Setpoint transmission rate. Default: 50 Hz.",
    )
    parser.add_argument(
        "--move-to-start-s",
        default=3.0,
        type=positive_float,
        help="Smooth transition duration from neutral to start.",
    )
    parser.add_argument(
        "--return-duration-s",
        default=3.0,
        type=positive_float,
        help="Controlled return duration to neutral.",
    )
    parser.add_argument(
        "--neutral-hold-s",
        default=0.5,
        type=non_negative_float,
        help="Time to continue transmitting neutral before closing.",
    )
    parser.add_argument(
        "--neutral-shoulder-deg",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--neutral-elbow-deg",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--neutral-rotation-deg",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate the trajectory without transmitting UDP packets.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive START confirmation.",
    )

    return parser


def poll_keyboard() -> str | None:
    """Return pause or stop when a supported key has been pressed."""

    command = None

    while msvcrt.kbhit():
        key = msvcrt.getwch()

        if key == " ":
            command = "pause"
        elif key == "\x1b":
            command = "stop"

    return command


def transition_references(
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
        for initial, target in zip(initial_deg, target_deg)
    )


def references_to_radians(
    references_deg: ReferenceTuple,
) -> ReferenceTuple:
    return tuple(math.radians(value) for value in references_deg)


def send_references(
    sender: LabVIEWSetpointSender | None,
    references_deg: ReferenceTuple,
) -> None:
    if sender is None:
        return

    shoulder, elbow, rotation = references_to_radians(
        references_deg
    )
    sender.send(shoulder, elbow, rotation)


def wait_for_tick(
    *,
    next_tick: float,
    period_s: float,
) -> tuple[float, float]:
    now = time.perf_counter()

    if next_tick > now:
        time.sleep(next_tick - now)

    now = time.perf_counter()
    next_tick += period_s

    # Recover gracefully if Windows suspended the process for too long.
    if now - next_tick > 5.0 * period_s:
        next_tick = now + period_s

    return now, next_tick


def print_reference_status(
    *,
    state: str,
    references_deg: ReferenceTuple,
    elapsed_s: float,
) -> None:
    shoulder, elbow, rotation = references_deg

    print(
        f"\r[{state:<13}] "
        f"t={elapsed_s:7.2f} s | "
        f"shoulder={shoulder:8.3f} deg | "
        f"elbow={elbow:8.3f} deg | "
        f"rotation={rotation:8.3f} deg",
        end="",
        flush=True,
    )


def run_transition(
    *,
    sender: LabVIEWSetpointSender | None,
    initial_deg: ReferenceTuple,
    target_deg: ReferenceTuple,
    duration_s: float,
    rate_hz: float,
    state: str,
    allow_escape: bool,
) -> tuple[ReferenceTuple, bool, int]:
    period_s = 1.0 / rate_hz
    started_at = time.perf_counter()
    next_tick = started_at
    last_print_s = -1.0
    packets = 0
    current = initial_deg

    while True:
        now, next_tick = wait_for_tick(
            next_tick=next_tick,
            period_s=period_s,
        )
        elapsed_s = min(now - started_at, duration_s)

        if allow_escape and poll_keyboard() == "stop":
            print("\n[STOP] Controlled return requested.")
            return current, True, packets

        current = transition_references(
            initial_deg,
            target_deg,
            elapsed_s,
            duration_s,
        )
        send_references(sender, current)
        packets += 1

        if elapsed_s - last_print_s >= 0.5:
            print_reference_status(
                state=state,
                references_deg=current,
                elapsed_s=elapsed_s,
            )
            last_print_s = elapsed_s

        if elapsed_s >= duration_s:
            print()
            return current, False, packets


def run_trajectory(
    *,
    sender: LabVIEWSetpointSender | None,
    trajectory: CosineTrajectory,
    rate_hz: float,
) -> tuple[ReferenceTuple, bool, int]:
    period_s = 1.0 / rate_hz
    started_at = time.perf_counter()
    next_tick = started_at

    paused = False
    pause_started_at = 0.0
    total_paused_s = 0.0
    trajectory_time_s = 0.0
    last_print_s = -1.0
    packets = 0

    current = trajectory.references_deg(
        0.0
    ).as_degrees_tuple()

    while True:
        now, next_tick = wait_for_tick(
            next_tick=next_tick,
            period_s=period_s,
        )

        command = poll_keyboard()

        if command == "stop":
            print("\n[STOP] Controlled return requested.")
            return current, True, packets

        if command == "pause":
            if paused:
                total_paused_s += now - pause_started_at
                paused = False
                print("\n[RESUME] Trajectory resumed.")
            else:
                paused = True
                pause_started_at = now
                print("\n[PAUSE] Reference held. Press Space to resume.")

        if not paused:
            trajectory_time_s = (
                now
                - started_at
                - total_paused_s
            )
            trajectory_time_s = min(
                trajectory_time_s,
                trajectory.duration_s,
            )

            current = trajectory.references_deg(
                trajectory_time_s
            ).as_degrees_tuple()

        # While paused, the same reference is still transmitted so that
        # LabVIEW continues receiving packets and its watchdog stays active.
        send_references(sender, current)
        packets += 1

        state = "PAUSED" if paused else "RUNNING"

        if (
            trajectory_time_s - last_print_s >= 0.5
            or paused
        ):
            print_reference_status(
                state=state,
                references_deg=current,
                elapsed_s=trajectory_time_s,
            )

            if not paused:
                last_print_s = trajectory_time_s

        if (
            not paused
            and trajectory_time_s >= trajectory.duration_s
        ):
            print()
            return current, False, packets


def hold_neutral(
    *,
    sender: LabVIEWSetpointSender | None,
    neutral_deg: ReferenceTuple,
    duration_s: float,
    rate_hz: float,
) -> int:
    if duration_s <= 0:
        send_references(sender, neutral_deg)
        return 1

    period_s = 1.0 / rate_hz
    started_at = time.perf_counter()
    next_tick = started_at
    packets = 0

    while True:
        now, next_tick = wait_for_tick(
            next_tick=next_tick,
            period_s=period_s,
        )
        elapsed_s = now - started_at

        send_references(sender, neutral_deg)
        packets += 1

        if elapsed_s >= duration_s:
            return packets


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    neutral_deg = {
        "shoulder": args.neutral_shoulder_deg,
        "elbow": args.neutral_elbow_deg,
        "rotation": args.neutral_rotation_deg,
    }

    try:
        trajectory = CosineTrajectory(
            joint=args.joint,
            start_deg=args.start_deg,
            end_deg=args.end_deg,
            frequency_hz=args.frequency_hz,
            cycles=args.cycles,
            neutral_deg=neutral_deg,
        )
    except TrajectoryError as exc:
        print(f"[ERROR] Invalid trajectory: {exc}", file=sys.stderr)
        return 2

    neutral_references = JointReferences(
        shoulder_deg=neutral_deg["shoulder"],
        elbow_deg=neutral_deg["elbow"],
        rotation_deg=neutral_deg["rotation"],
    ).as_degrees_tuple()

    start_references = trajectory.references_deg(
        0.0
    ).as_degrees_tuple()

    print()
    print("LabVIEW three-DOF trajectory")
    print("---------------------------")
    print(f"Active joint:       {trajectory.joint}")
    print(
        f"Range:              "
        f"{trajectory.start_deg:.3f} to "
        f"{trajectory.end_deg:.3f} deg"
    )
    print(f"Frequency:          {trajectory.frequency_hz:.4f} Hz")
    print(f"Cycles:             {trajectory.cycles}")
    print(f"Trajectory duration:{trajectory.duration_s:9.3f} s")
    print(
        f"Peak velocity:      "
        f"{trajectory.peak_velocity_deg_s:.3f} deg/s"
    )
    print(
        f"Peak acceleration:  "
        f"{trajectory.peak_acceleration_deg_s2:.3f} deg/s^2"
    )
    print(f"Setpoint rate:      {args.rate_hz:.3f} Hz")
    print(
        f"Destination:        "
        f"{args.host}:{args.port}"
        if not args.dry_run
        else "Destination:        DRY RUN — UDP disabled"
    )
    print()
    print("Space = pause/resume")
    print("Esc   = controlled stop and return to neutral")
    print("Ctrl+C= controlled stop and return to neutral")
    print()
    print(
        "Space is a protocol pause, not an emergency stop. "
        "The physical emergency stop remains in LabVIEW/cRIO."
    )

    if not args.dry_run and not args.yes:
        confirmation = input(
            '\nType "START" to begin transmitting: '
        ).strip()

        if confirmation != "START":
            print("[INFO] Cancelled before UDP transmission.")
            return 0

    sender: LabVIEWSetpointSender | None = None
    total_packets = 0
    current_references = neutral_references
    controlled_stop = False

    try:
        if not args.dry_run:
            sender = LabVIEWSetpointSender(
                host=args.host,
                port=args.port,
            )

        print("\n[MOVE_TO_START] Smooth transition to initial position.")

        (
            current_references,
            controlled_stop,
            packets,
        ) = run_transition(
            sender=sender,
            initial_deg=neutral_references,
            target_deg=start_references,
            duration_s=args.move_to_start_s,
            rate_hz=args.rate_hz,
            state="MOVE_TO_START",
            allow_escape=True,
        )
        total_packets += packets

        if not controlled_stop:
            print(
                "[RUNNING] Trajectory started. "
                "Space pauses without stopping UDP."
            )

            (
                current_references,
                controlled_stop,
                packets,
            ) = run_trajectory(
                sender=sender,
                trajectory=trajectory,
                rate_hz=args.rate_hz,
            )
            total_packets += packets

            if not controlled_stop:
                print("[DONE] Requested cycles completed.")

    except KeyboardInterrupt:
        controlled_stop = True
        print("\n[STOP] Ctrl+C received. Starting controlled return.")

    except (SetpointUDPError, OSError) as exc:
        print(
            f"\n[ERROR] UDP trajectory failed: {exc}",
            file=sys.stderr,
        )
        print(
            "[WARNING] Verify the independent LabVIEW/cRIO "
            "watchdog and safety state.",
            file=sys.stderr,
        )
        return 3

    finally:
        try:
            if sender is not None or args.dry_run:
                print("[RETURNING] Smooth return to neutral.")

                current_references, _, packets = run_transition(
                    sender=sender,
                    initial_deg=current_references,
                    target_deg=neutral_references,
                    duration_s=args.return_duration_s,
                    rate_hz=args.rate_hz,
                    state="RETURNING",
                    allow_escape=False,
                )
                total_packets += packets

                packets = hold_neutral(
                    sender=sender,
                    neutral_deg=neutral_references,
                    duration_s=args.neutral_hold_s,
                    rate_hz=args.rate_hz,
                )
                total_packets += packets

                print("[NEUTRAL] Neutral reference reached and held.")

        except (SetpointUDPError, OSError) as exc:
            print(
                f"\n[ERROR] Controlled return transmission failed: {exc}",
                file=sys.stderr,
            )
            print(
                "[WARNING] LabVIEW/cRIO must enter its local safe state.",
                file=sys.stderr,
            )

        finally:
            if sender is not None:
                try:
                    sender.close()
                except SetpointUDPError as exc:
                    print(
                        f"[WARNING] UDP socket close failed: {exc}",
                        file=sys.stderr,
                    )

    print(f"[OK] Session finished. Packets generated: {total_packets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
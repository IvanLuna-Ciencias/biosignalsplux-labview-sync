#!/usr/bin/env python
"""Manually test LabVIEW PREPARE/START/STOP on UDP port 5006."""

from __future__ import annotations

import argparse
from datetime import datetime

from biosignalsplux_labview_sync.labview_session_udp import (
    LabVIEWSessionUDPClient,
    SessionTimeoutError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test the independent LabVIEW session coordination channel "
            "without starting sEMG, ATI, or trajectories."
        )
    )
    parser.add_argument("--host", required=True, help="IP address of the LabVIEW computer.")
    parser.add_argument("--port", type=int, default=5006, help="Remote LabVIEW session port.")
    parser.add_argument(
        "--session-id",
        default="TEST_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Safe identifier used for the manual test.",
    )
    parser.add_argument("--timeout-s", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-prompt", action="store_true")
    return parser


def wait_for_enter(message: str, *, no_prompt: bool) -> None:
    if not no_prompt:
        input(message)


def main() -> int:
    args = build_parser().parse_args()
    client = LabVIEWSessionUDPClient(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        retries=args.retries,
    )
    start_sent = False

    try:
        print(f"Local UDP socket: {client.local_address[0]}:{client.local_address[1]}")
        print(f"LabVIEW target:   {args.host}:{args.port}")
        print(f"Session ID:       {args.session_id}")
        print()
        print(f"Sending PREPARE,{args.session_id}")

        try:
            ready = client.prepare(args.session_id)
        except SessionTimeoutError as exc:
            print("[ERROR] LabVIEW did not confirm READY.")
            print(exc)
            print("The acquisition must not be started.")
            return 2

        print("Received:", ready.raw_text.rstrip())
        wait_for_enter("Press Enter to send START...", no_prompt=args.no_prompt)
        client.start()
        start_sent = True
        print(f"Sent START,{args.session_id}")
        wait_for_enter("Press Enter to send STOP...", no_prompt=args.no_prompt)
        print(f"Sending STOP,{args.session_id}")

        try:
            done = client.stop()
        except SessionTimeoutError as exc:
            print("[WARNING] LabVIEW did not confirm DONE.")
            print(exc)
            print(
                "Python-side data would still be preserved, but the "
                "LabVIEW recording closure is unconfirmed."
            )
            return 3

        print("Received:", done.raw_text.rstrip())
        print("Manual session protocol test completed successfully.")
        return 0

    except KeyboardInterrupt:
        print()
        print("Manual test interrupted.")
        if start_sent:
            print("Attempting STOP before closing the socket...")
            try:
                done = client.stop()
                print("Received:", done.raw_text.rstrip())
            except Exception as exc:
                print("[WARNING] STOP could not be confirmed:", exc)
        return 130
    finally:
        client.close()
        print("UDP session socket closed.")


if __name__ == "__main__":
    raise SystemExit(main())

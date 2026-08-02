"""Send a world-frame target to an interactive MuJoCo navigation session."""

from __future__ import annotations

import argparse
import json
import socket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float, nargs="?", default=0.0)
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    payload = json.dumps([args.x, args.y, args.z]).encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.sendto(payload, ("127.0.0.1", args.port))
    print(f"Sent MuJoCo world goal: ({args.x:.3f}, {args.y:.3f}, {args.z:.3f})")


if __name__ == "__main__":
    main()

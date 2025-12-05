"""Debug utility for verifying STM32 serial protocol.

This script can be executed on the host PC to send sample pan/tilt
commands to the STM32 Nucleo board and print any responses. It is meant
to validate the ASCII protocol expected by ``uart_comm.c`` before
running the full OpenCV/YOLO pipeline.

Usage examples:

    python debug_serial_test.py --port /dev/ttyACM0 --pan 120 --tilt 80
    python debug_serial_test.py --port COM6 --sweep

"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import serial


def send_command(connection: serial.Serial, pan: int, tilt: int) -> Optional[str]:
    """Send a single C,<pan>,<tilt> command and return any response."""
    command = f"C,{pan},{tilt}\n"
    connection.write(command.encode("ascii"))
    time.sleep(0.05)
    if connection.in_waiting:
        return connection.readline().decode(errors="ignore").strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="STM32 UART debug helper")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM6 or /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate, defaults to 115200")
    parser.add_argument("--pan", type=int, default=90, help="Pan angle to send when not sweeping")
    parser.add_argument("--tilt", type=int, default=90, help="Tilt angle to send when not sweeping")
    parser.add_argument("--sweep", action="store_true", help="Sweep angles 60→120 degrees for quick motion test")
    parser.add_argument("--iterations", type=int, default=5, help="Number of sweep iterations")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between commands during sweep (seconds)")

    args = parser.parse_args()

    try:
        with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
            print(f"[INFO] Connected to {args.port} at {args.baud} baud")

            if args.sweep:
                angles = [60, 90, 120, 90]
                for i in range(args.iterations):
                    for angle in angles:
                        response = send_command(ser, angle, angle)
                        print(f"[TX] C,{angle},{angle} | [RX] {response}")
                        time.sleep(args.delay)
            else:
                response = send_command(ser, args.pan, args.tilt)
                print(f"[TX] C,{args.pan},{args.tilt} | [RX] {response}")

    except serial.SerialException as exc:
        print(f"[ERROR] Could not open serial port: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


"""Receive ten PGM files from the Pico's USB serial export firmware on macOS."""
import argparse
import os
import re
import select
import termios
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BEGIN = re.compile(rb"<<<BEGIN (digit_[0-9]\.pgm)>>>\r?\n$")
END = re.compile(rb"<<<END (digit_[0-9]\.pgm)>>>\r?\n$")


def configure_port(fd):
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = termios.B115200
    attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 1
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)


def receive(port, timeout):
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        configure_port(fd)
        files = {}
        active_name = None
        active_data = bytearray()
        pending = bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.25)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            if not chunk:
                continue
            pending.extend(chunk)
            while b"\n" in pending:
                raw_line, _, remaining = pending.partition(b"\n")
                pending = bytearray(remaining)
                line = bytes(raw_line).rstrip(b"\r")
                begin = BEGIN.fullmatch(line + b"\n")
                end = END.fullmatch(line + b"\n")
                if line == b"PICO_PGM_EXPORT_DONE":
                    if active_name is not None:
                        raise RuntimeError("Export ended in the middle of a file")
                    return files
                if line.startswith(b"EXPORT_ERROR"):
                    raise RuntimeError(line.decode("ascii", "replace"))
                if begin:
                    if active_name is not None:
                        raise RuntimeError("Nested file marker in serial stream")
                    active_name = begin.group(1).decode("ascii")
                    active_data = bytearray()
                    continue
                if end:
                    name = end.group(1).decode("ascii")
                    if active_name != name:
                        raise RuntimeError("Mismatched end marker for " + name)
                    files[name] = bytes(active_data)
                    active_name = None
                    active_data = bytearray()
                    continue
                if active_name is not None:
                    active_data.extend(raw_line)
                    active_data.extend(b"\n")
        raise TimeoutError("Timed out waiting for PICO_PGM_EXPORT_DONE")
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="macOS USB serial path, e.g. /dev/cu.usbmodem1401")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--refresh-own-digits", action="store_true",
                        help="replace data/own_digits while preserving the first raw SD export")
    args = parser.parse_args()

    files = receive(args.port, args.timeout)
    expected = {f"digit_{digit}.pgm" for digit in range(10)}
    if set(files) != expected:
        raise RuntimeError("Expected 10 digit PGM files; received: " + ", ".join(sorted(files)))
    for name, data in files.items():
        if not data.startswith(b"P2\n280 240\n255\n"):
            raise RuntimeError(name + " does not have the expected PGM header")

    raw_dir = ROOT / "data" / "raw_sd_card"
    digit_dir = ROOT / "data" / "own_digits"
    destinations = [] if args.refresh_own_digits else [raw_dir / name for name in expected] + [digit_dir / name for name in expected]
    conflicts = [path for path in destinations if path.exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite existing files: " + ", ".join(map(str, conflicts)))

    raw_dir.mkdir(parents=True, exist_ok=True)
    digit_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        if not args.refresh_own_digits:
            (raw_dir / name).write_bytes(data)
        (digit_dir / name).write_bytes(data)
    if args.refresh_own_digits:
        print("Updated data/own_digits/; the first raw capture archive is unchanged.")
    else:
        print("Received and copied 10 PGM images to data/raw_sd_card/ and data/own_digits/.")


if __name__ == "__main__":
    main()

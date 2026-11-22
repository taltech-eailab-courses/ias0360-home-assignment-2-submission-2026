"""Record IMU gestures from the Pico over USB serial into data/own/<class>/.

Pairs with hw2/data_collection_serial (flash its main.uf2 first).

    python tools/record_serial.py ring            # 5 recordings of "ring"
    python tools/record_serial.py wing -n 8
    python tools/record_serial.py negative --port /dev/ttyACM0
    python tools/record_serial.py --ping          # connection check only

Each recording: terminal countdown, LED blue (1 s, get ready), LED green
(2.1 s, do the gesture), LED yellow (sending). The file is written as
data/own/<class>/own_<timestamp>.txt with the same "ax,ay,az" CSV format as the
shared dataset. After each one you can keep it (Enter), redo it (r) or quit (q).
Use --auto to skip the question and just record n in a row.
"""

import argparse
import sys
import time
from pathlib import Path

import serial

CLASSES = ["negative", "ring", "slope", "wing"]
HW2_DIR = Path(__file__).resolve().parent.parent
OWN_DIR = HW2_DIR / "data" / "own"


def open_port(port: str) -> serial.Serial:
    ser = serial.Serial(port, 115200, timeout=0.5)
    time.sleep(0.3)
    ser.reset_input_buffer()
    return ser


def ping(ser: serial.Serial) -> bool:
    ser.reset_input_buffer()
    ser.write(b"p")
    deadline = time.time() + 2
    while time.time() < deadline:
        line = ser.readline().decode(errors="replace").strip()
        if line == "PONG":
            return True
        if line.startswith("ERROR"):
            print(line)
            return False
    return False


def record_one(ser: serial.Serial) -> tuple[list[str], int, int]:
    ser.reset_input_buffer()
    ser.write(b"r")
    deadline = time.time() + 15
    lines: list[str] = []
    n = duration = -1
    started = False
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode(errors="replace").strip()
        if not started:
            if line.startswith("BEGIN"):
                _, n_s, d_s = line.split()
                n, duration = int(n_s), int(d_s)
                started = True
            continue
        if line == "END":
            break
        lines.append(line)
    else:
        raise TimeoutError("no END from the Pico within 15 s")

    if not lines or lines[0] != "ax,ay,az":
        raise ValueError(f"unexpected header: {lines[:1]}")
    if len(lines) - 1 != n:
        raise ValueError(f"expected {n} rows, got {len(lines) - 1}")
    return lines, n, duration


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cls", nargs="?", choices=CLASSES)
    ap.add_argument("-n", type=int, default=5, help="recordings to keep (default 5)")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--auto", action="store_true",
                    help="no keep/redo question, 3 s pause between recordings")
    ap.add_argument("--ping", action="store_true", help="only check the connection")
    args = ap.parse_args()

    ser = open_port(args.port)
    if not ping(ser):
        print(f"No answer from the Pico on {args.port}. Is data_collection_serial flashed?")
        return 1
    print(f"Pico OK on {args.port}")
    if args.ping:
        return 0
    if args.cls is None:
        ap.error("give a class: " + ", ".join(CLASSES))

    out_dir = OWN_DIR / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    while kept < args.n:
        print(f"\n[{args.cls} {kept + 1}/{args.n}] ", end="", flush=True)
        for s in (3, 2, 1):
            print(f"{s}..", end="", flush=True)
            time.sleep(0.7 if not args.auto else 1.0)
        print(" blue = get ready, GREEN = GO", flush=True)

        try:
            lines, n, duration = record_one(ser)
        except (TimeoutError, ValueError) as e:
            print(f"  failed: {e}")
            continue
        rate = n / (duration / 1e6)
        flag = "" if abs(n - 4200) <= 210 else "  (row count >5% off 4200, preprocessor will reject)"
        print(f"  {n} rows, {duration / 1e6:.2f} s, {rate:.0f} Hz{flag}")

        if not args.auto:
            ans = input("  keep [Enter] / redo [r] / quit [q]: ").strip().lower()
            if ans == "q":
                break
            if ans == "r":
                continue

        path = out_dir / f"own_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        path.write_text("\n".join(lines) + "\n")
        kept += 1
        print(f"  saved {path.relative_to(HW2_DIR)}")

    total = len(list(out_dir.glob("*.txt")))
    print(f"\n{kept} new, {total} total in data/own/{args.cls}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

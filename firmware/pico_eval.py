#!/usr/bin/env python3
"""Send the 28x28 own digits to main_uart_inference, report on-device accuracy and time."""
import argparse
import json
import time
from pathlib import Path

import serial

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/pico")
    parser.add_argument("--digits", type=Path, default=BASE_DIR / "data" / "own_digits_28x28.csv")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "output" / "pico_results.json")
    args = parser.parse_args()

    rows = [line.split(",", 1) for line in args.digits.read_text().splitlines() if line]
    results = []
    with serial.Serial(args.port, 115200, timeout=10) as ser:
        time.sleep(3)
        ser.reset_input_buffer()
        for label, pixels in rows:
            ser.write((pixels + "\n").encode("ascii"))
            line = ""
            while not line.startswith(("OK,", "ERROR,")):
                raw = ser.readline()
                if not raw:
                    raise SystemExit("no answer from the Pico")
                line = raw.decode("ascii", "replace").strip()
            if line.startswith("ERROR,"):
                raise SystemExit(line)
            fields = line.split(",")
            scores = [float(s) for s in fields[3:]]
            results.append({"label": int(label), "predicted": int(fields[1]), "inference_us": int(fields[2]),
                            "scores": scores})
            top = sorted(range(len(scores)), key=lambda d: -scores[d])[:3]
            shown = ", ".join(f"{d}: {scores[d] * 100:.1f}%" for d in top)
            print(f"digit {label}: predicted {fields[1]}, {int(fields[2]) / 1000:.2f} ms ({shown})")

    correct = sum(r["label"] == r["predicted"] for r in results)
    avg_ms = sum(r["inference_us"] for r in results) / len(results) / 1000
    print(f"on-device accuracy {correct}/{len(results)}, average {avg_ms:.2f} ms per digit")
    args.out.write_text(json.dumps({"correct": correct, "total": len(results), "avg_ms": round(avg_ms, 2),
                                    "digits": results}, indent=1))


if __name__ == "__main__":
    main()

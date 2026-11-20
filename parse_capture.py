import sys
import numpy as np
from PIL import Image

rows, digits = [], []
for line in open(sys.argv[1], errors="ignore"):
    line = line.rstrip("\r\n")
    if not line.strip():
        continue                               # blank separator line
    if len(line) == 84 and line.replace(" ", "").isdigit():
        rows.append([int(line[i:i + 3]) for i in range(0, 84, 3)])
        if len(rows) == 28:
            digits.append(np.array(rows, dtype="uint8"))
            rows = []
    else:
        rows = []                              # e.g. "Login button pressed."

kept = 0
for d in digits:
    if d.max() == 0:                           # empty box
        continue
    Image.fromarray(d).save(f"touch_{kept}.png")
    kept += 1
print(len(digits), "blocks found,", kept, "non-empty saved")
"""Prüft die auf dem Pico laufende Inferenz gegen unveränderte CPU-Referenzen."""
import fcntl
import glob
import json
import os
from pathlib import Path
import re
import select
import struct
import sys
import termios
import time

folder = Path(sys.argv[1])
ports = glob.glob('/dev/cu.usbmodem*')
assert len(ports) == 1, ports
fd = os.open(ports[0], os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
buffer = bytearray()

def send(data):
    while data:
        _, writable, _ = select.select([], [fd], [], 5)
        assert writable, 'USB write timeout'
        try:
            n = os.write(fd, data)
            data = data[n:]
        except BlockingIOError:
            pass

def line(prefix, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        while b'\n' in buffer:
            raw, _, rest = buffer.partition(b'\n')
            buffer[:] = rest
            text = raw.decode(errors='replace').strip()
            if text.startswith(prefix):
                return text
            if 'ERROR' in text:
                raise RuntimeError(text)
        ready, _, _ = select.select([fd], [], [], 0.5)
        if ready:
            try:
                buffer.extend(os.read(fd, 8192))
            except BlockingIOError:
                pass
    raise TimeoutError(f'No {prefix} response; {buffer!r}')

try:
    attrs = termios.tcgetattr(fd)
    attrs[0] = attrs[1] = attrs[3] = 0
    attrs[2] = (attrs[2] & ~(termios.PARENB | termios.CSTOPB | termios.CSIZE)) | termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[4] = attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    fcntl.ioctl(fd, termios.TIOCMBIS, struct.pack('I', termios.TIOCM_DTR))
    time.sleep(0.2)
    send(b'?')
    print(line('H2 DEMO'), flush=True)
    references = json.loads((folder / 'reference.json').read_text())
    frames = (folder / 'inputs.bin').read_bytes()
    results = []
    for index, record in enumerate(references):
        send(b'Q' + frames[index * 784:(index + 1) * 784])
        reply = line('RESULT')
        match = re.search(r'digit=(\d+).*inference_ms=(\d+) scores=([-\d,]+)', reply)
        assert match, reply
        digit, ms = map(int, match.group(1, 2))
        scores = list(map(int, match[3].split(',')))
        error = max(abs(a - b) for a, b in zip(scores, record['scores']))
        result = dict(**record, actual=digit, inference_ms=ms, max_score_error=error)
        results.append(result)
        print(f'{index + 1:02}: {record["name"]} Pico={digit} CPU={record["expected"]} ms={ms} score_delta={error}', flush=True)
        assert digit == record['expected'], result
        assert error <= 2, result
    (folder / 'hardware_results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'PASS: {len(results)} board predictions match CPU; own VAL correct={sum(r["actual"] == r["label"] for r in results)}; inference_ms={min(r["inference_ms"] for r in results)}..{max(r["inference_ms"] for r in results)}')
finally:
    send(b'C?')
    print(line('H2 DEMO'), flush=True)
    os.close(fd)

"""Print the Pico's USB serial output for a while: python tools/serial_log.py [seconds]"""
import sys
import time

import serial

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60
end = time.time() + seconds
while True:
    try:
        port = serial.Serial("/dev/ttyACM0", 115200, timeout=0.5)
        break
    except serial.SerialException:
        if time.time() > end:
            sys.exit("no serial port")
        time.sleep(0.5)
while time.time() < end:
    line = port.readline().decode(errors="replace").rstrip()
    if line:
        print(line, flush=True)

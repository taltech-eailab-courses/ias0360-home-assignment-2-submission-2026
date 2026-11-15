#!/usr/bin/env bash
# Demo: flashes the digit reader. Draw a digit on the LCD, the Pico reads it by itself.
# On the Mac the board sits in the VM, so the script starts the VM and runs itself there.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UF2="$HERE/firmware/inference/build/main.uf2"

if [ ! -f "$UF2" ]; then
    echo "no $UF2 yet, build it first (README, section 3)." >&2
    exit 1
fi

if [ "$(uname -s)" = "Darwin" ]; then
    VM=ias0360-ubuntu
    SSH=(ssh -p 2222 -i "$HOME/.ssh/id_ed25519_ias0360" -o BatchMode=yes)
    VBoxManage list runningvms | grep -q "\"$VM\"" || VBoxManage startvm "$VM" --type headless
    echo ">> waiting for the VM"
    until "${SSH[@]}" -o ConnectTimeout=3 joosep@127.0.0.1 true 2>/dev/null; do sleep 2; done
    exec "${SSH[@]}" -t joosep@127.0.0.1 "/media/sf_repo/submission/$(basename "$HERE")/start.sh"
fi

for attempt in 1 2 3; do
    picotool load "$UF2" -f -x && break
    [ "$attempt" -lt 3 ] || { echo "flash failed, is the board on the data USB port?" >&2; exit 1; }
    sleep 2
done

PORT=/dev/pico
[ -e "$PORT" ] || PORT=/dev/ttyACM0
sleep 3
until [ -e "$PORT" ]; do sleep 0.2; done

echo
echo ">> Running on the Pico. Draw a digit, stop, and it reads it after a short countdown."
echo ">> Ctrl-C leaves this view, the Pico keeps running."
stty -F "$PORT" raw 115200 -echo -hupcl
cat "$PORT" | tee -a "$HERE/output/demo_log.txt" | awk -F, '
    /^Model initialized/ { print; fflush() }
    /^READ,/ { printf "file %d: read as %d, %.1f%% sure, %.1f ms\n", $2, $3, $4, $5 / 1000; fflush() }'

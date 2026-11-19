#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$submission_dir/firmware/deps/course"
revision=9f1560597cf8cb864c0b04025b5faeeb792aeadc
if [[ -e "$destination" ]]; then
    echo "Already exists: $destination (left unchanged)"
    exit 1
fi
git clone --filter=blob:none --no-checkout \
    https://github.com/taltech-eailab-courses/ias0360-lab-excercises-2026.git "$destination"
git -C "$destination" sparse-checkout init --cone
git -C "$destination" sparse-checkout set lab_2_4/pico_sw/lcd/lib/pico-tflmicro
git -C "$destination" checkout --detach "$revision"

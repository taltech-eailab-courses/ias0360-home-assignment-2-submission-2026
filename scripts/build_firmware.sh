#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
program="${1:-demo}"
case "$program" in capture|demo) ;; *) echo 'Usage: bash scripts/build_firmware.sh capture|demo'; exit 1;; esac
exec docker run --rm --network none \
    --mount "type=bind,source=$submission_dir,target=/submission" \
    --workdir /submission --entrypoint /bin/bash ias0360-2026:latest -lc \
    "cmake -S firmware/$program -B firmware/$program/build -DCMAKE_BUILD_TYPE=Release && cmake --build firmware/$program/build -j 6"

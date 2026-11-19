#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$submission_dir/cache"
exec docker run --rm \
    --mount "type=bind,source=$submission_dir,target=/submission" \
    --workdir /submission -e KERAS_HOME=/submission/cache/keras \
    --entrypoint /opt/venv/bin/python ias0360-2026:latest -u "$@"

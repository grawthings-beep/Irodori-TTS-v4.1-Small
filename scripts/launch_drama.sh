#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
nohup bash scripts/runpod_drama_start.sh >> logs/drama.log 2>&1 < /dev/null &
echo "Drama starts in the background. Ctrl+C closes only this log view."
tail -n 80 -F logs/drama.log

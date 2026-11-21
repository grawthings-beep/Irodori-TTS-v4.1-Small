#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p .runtime logs outputs/audio .cache .tools
exec 9>.runtime/server.lock
if ! flock -n 9; then
    echo "Irodori is already starting or running. See logs/runpod.log."
    exit 0
fi
trap 'echo "Startup failed at line $LINENO. See logs/runpod.log." >&2' ERR

if ! command -v nvidia-smi >/dev/null; then
    echo "Use a RunPod GPU Pod with NVIDIA drivers." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Install only missing OS packages; use a separate uv venv for Python packages.
packages=(git curl ca-certificates ffmpeg libsndfile1 build-essential pkg-config)
missing=()
for package in "${packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
        missing+=("$package")
    fi
done
if ((${#missing[@]})); then
    if ((EUID != 0)); then
        echo "Missing OS packages: ${missing[*]}. Install them as root and rerun." >&2
        exit 1
    fi
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
fi

export PATH="$ROOT/.tools:$PATH"
if ! command -v uv >/dev/null; then
    curl -fsSL --retry 3 https://astral.sh/uv/install.sh -o .runtime/install-uv.sh
    UV_INSTALL_DIR="$ROOT/.tools" UV_NO_MODIFY_PATH=1 sh .runtime/install-uv.sh
fi
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.tools/python"
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export HF_HOME="$ROOT/.cache/huggingface"
export GRADIO_TEMP_DIR="$ROOT/.runtime/gradio"
export GRADIO_ANALYTICS_ENABLED=False
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1

# Keep the official lock file and the CUDA 12.8 backend. Never sync without it.
uv sync --frozen --extra cu128 --python 3.10
uv run --no-sync python scripts/preflight.py
exec uv run --no-sync python voice_clone_app.py --server-name 0.0.0.0 --server-port "${IRODORI_PORT:-7860}"

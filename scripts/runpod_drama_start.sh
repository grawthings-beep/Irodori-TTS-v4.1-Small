#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STATE="$ROOT/.runtime/drama"
mkdir -p "$STATE" "$ROOT/.tools" "$ROOT/logs" "$ROOT/outputs/drama"
exec 9>"$STATE/server.lock"
if ! flock -n 9; then
    echo "Drama is already starting or running. See logs/drama.log."
    exit 0
fi
trap 'echo "Drama startup failed at line $LINENO. See logs/drama.log." >&2' ERR
command -v nvidia-smi >/dev/null || { echo "Use an NVIDIA GPU Pod." >&2; exit 1; }
command -v nvcc >/dev/null || { echo "CUDA compiler missing. Use the RunPod PyTorch CUDA devel image." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvcc --version

missing=()
for package in git curl ca-certificates ffmpeg build-essential cmake ninja-build; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
        missing+=("$package")
    fi
done
if ((${#missing[@]})); then
    ((EUID == 0)) || { echo "Install these OS packages as root: ${missing[*]}" >&2; exit 1; }
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
fi
export PATH="$ROOT/.tools:$PATH"
if ! command -v uv >/dev/null; then
    curl -fsSL --retry 3 https://astral.sh/uv/install.sh -o "$STATE/install-uv.sh"
    UV_INSTALL_DIR="$ROOT/.tools" UV_NO_MODIFY_PATH=1 sh "$STATE/install-uv.sh"
fi
export UV_CACHE_DIR="$ROOT/.cache/drama-uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.tools/python"
export UV_PROJECT_ENVIRONMENT="$STATE/venv"
export GRADIO_TEMP_DIR="$STATE/gradio"
export GRADIO_ANALYTICS_ENABLED=False
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
# Preserve intentional pauses by default. Set 18 explicitly to try the pack's pause clamp.
export SPEECH_DRAMA3_MAX_PAUSE_FRAMES="${DRAMA_MAX_PAUSE_FRAMES:-0}"
export SPEECH_DRAMA3_SILENCE_CODES="$STATE/assets/silence_codes.json"

uv sync --project drama --frozen --python 3.11
"$STATE/venv/bin/python" -m drama.bootstrap
cmake -S "$STATE/speech.cpp" -B "$STATE/speech.cpp/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_VULKAN=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build "$STATE/speech.cpp/build" --target speech-cli -j "${DRAMA_BUILD_JOBS:-2}"
"$STATE/speech.cpp/build/speech" devices --json
"$STATE/speech.cpp/build/speech" info "$STATE/assets/Drama-35B-A3B-Japanese-v3-small-IQ3_XXS.gguf" --json
exec "$STATE/venv/bin/python" -m drama.app --server-name 0.0.0.0 --server-port "${DRAMA_PORT:-7861}"

"""A serial, persistent speech.cpp worker and reusable reference voices."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".runtime" / "drama"
PINS = json.loads(Path(__file__).with_name("pins.json").read_text(encoding="utf-8"))
MODEL = STATE / "assets" / PINS["model_file"]
BINARY = STATE / "speech.cpp" / "build" / "speech"
OUTPUTS = ROOT / "outputs" / "drama"


class Worker:
    def __init__(self, command, log_path, timeout=600):
        self.command = command
        self.log_path = Path(log_path)
        self.timeout = timeout
        self.process = None
        self.log = None
        self.messages = queue.Queue()
        self.voices = set()
        self.sample_rate = 44100

    def _receive(self, stream, inbox):
        try:
            for line in stream:
                inbox.put(json.loads(line))
        except Exception as exc:
            inbox.put(exc)
        finally:
            inbox.put(EOFError("Drama worker exited. See logs/drama-worker.log."))

    def _next(self, deadline):
        try:
            event = self.messages.get(timeout=max(0.001, deadline - time.monotonic()))
        except queue.Empty as exc:
            raise TimeoutError(
                "Dramaの応答がタイムアウトしました。ログを確認してください。"
            ) from exc
        if isinstance(event, Exception):
            raise event
        if not isinstance(event, dict):
            raise RuntimeError("Invalid worker protocol response.")
        if event.get("type") == "fatal":
            raise RuntimeError(str(event.get("error")))
        return event

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a", encoding="utf-8")
        self.messages = queue.Queue()
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            threading.Thread(
                target=self._receive, args=(self.process.stdout, self.messages), daemon=True
            ).start()
            event = self._next(time.monotonic() + self.timeout)
            if event.get("type") != "ready" or event.get("protocol") != 4:
                raise RuntimeError(
                    "Unsupported speech.cpp worker protocol; expected ready/protocol 4."
                )
            if event.get("model", {}).get("architecture") != "drama3":
                raise RuntimeError("The loaded model is not Drama 3.")
            self.sample_rate = int(event["model"]["sample_rate"])
            if self.sample_rate != 44100:
                raise RuntimeError("Unexpected Drama sample rate.")
        except Exception:
            self.close()
            raise

    def _request(self, data):
        self.start()
        request_id = uuid4().hex
        data = {**data, "id": request_id}
        try:
            self.process.stdin.write(json.dumps(data, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            while True:
                event = self._next(deadline)
                if event.get("id") != request_id:
                    raise RuntimeError("Unexpected request ID from Drama worker.")
                kind = event.get("type")
                if kind in {"error", "cancelled"}:
                    raise RuntimeError(str(event.get("error", "Generation cancelled.")))
                if kind not in {"chunk", "progress", "end"}:
                    raise RuntimeError(f"Unexpected worker event: {kind}")
                yield event
                if kind == "end":
                    return
        except BaseException:
            # A timed-out or abandoned request must never leave stale audio in the next request.
            self.close()
            raise

    def add_voice(self, name, path):
        self.start()
        if name not in self.voices:
            for _ in self._request({"type": "add_voice", "name": name, "path": str(path)}):
                pass
            self.voices.add(name)

    def synthesize(self, request, target, progress=None):
        self.start()
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial.wav")
        samples, sequence, terminal = 0, 0, None
        try:
            with wave.open(str(partial), "wb") as wav:
                wav.setparams((1, 2, self.sample_rate, 0, "NONE", "not compressed"))
                for event in self._request({"type": "synthesize", **request}):
                    if event["type"] == "chunk":
                        if event.get("seq") != sequence:
                            raise RuntimeError("Out-of-order audio chunk.")
                        pcm = base64.b64decode(event["pcm"], validate=True)
                        if len(pcm) % 2:
                            raise RuntimeError("Invalid 16-bit audio chunk.")
                        samples += len(pcm) // 2
                        if samples > self.sample_rate * 600:
                            raise RuntimeError("Output exceeded the 10-minute storage limit.")
                        wav.writeframesraw(pcm)
                        sequence += 1
                    elif event["type"] == "progress" and progress:
                        progress(event.get("done", 0))
                    elif event["type"] == "end":
                        terminal = event
            if terminal is None or samples == 0 or terminal.get("samples") != samples:
                raise RuntimeError(
                    "音声が空か、受信データが不完全です。文章・Seedを変えて再生成してください。"
                )
            partial.replace(target)
            return terminal, samples / self.sample_rate
        except BaseException:
            self.close()
            partial.unlink(missing_ok=True)
            raise

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        if self.log:
            self.log.close()
            self.log = None
        self.voices.clear()


def make_worker():
    if not BINARY.is_file() or not MODEL.is_file():
        raise RuntimeError(
            "Dramaのセットアップが未完了です。scripts/runpod_drama_start.shを実行してください。"
        )
    device = os.environ.get("DRAMA_DEVICE", "CUDA0")
    devices = json.loads(subprocess.check_output([str(BINARY), "devices", "--json"], text=True))
    selected = next((d for d in devices["devices"] if d["name"] == device), None)
    if not selected or not device.startswith("CUDA"):
        raise RuntimeError(
            f"CUDAデバイス {device} が見つかりません。CUDA版ビルドとGPUを確認してください。"
        )
    if selected.get("memory_free", 0) < 12 * 1024**3:
        raise RuntimeError(
            "GPUの空きメモリが12GiB未満です。Irodori画面でGPUを解放するか、他のGPUアプリを終了してください。"
        )
    return Worker(
        [
            str(BINARY),
            "worker",
            str(MODEL),
            "--device",
            device,
            "--threads",
            "4",
            "--no-warmup",
            "--experts-on-cpu=false",
        ],
        ROOT / "logs" / "drama-worker.log",
    )


def prepare_voice(reference, transcript):
    """Encode only the codec on CPU; never load another 35B model beside the worker."""
    voices = STATE / "voices"
    voices.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=voices) as scratch:
        wav_path = Path(scratch) / "reference.wav"
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(reference),
                "-t",
                "30.1",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode:
            raise ValueError("参考音声を読み込めません。WAV・MP3・FLACなどの音声を選んでください。")
        with wave.open(str(wav_path)) as wav:
            seconds = wav.getnframes() / wav.getframerate()
        if not 0.5 <= seconds <= 30:
            raise ValueError(
                "参考音声は0.5秒以上30秒以内にしてください。長い録音は事前に切り出してください。"
            )
        key = hashlib.sha256(
            wav_path.read_bytes()
            + transcript.encode("utf-8")
            + PINS["pack_revision"].encode("ascii")
        ).hexdigest()
        target = voices / f"{key}.voice.gguf"
        if not target.exists():
            partial = Path(scratch) / "new.voice.gguf"
            result = subprocess.run(
                [
                    str(BINARY),
                    "voice",
                    str(MODEL),
                    str(wav_path),
                    str(partial),
                    "--transcript",
                    transcript,
                    "--device",
                    "cpu",
                    "--threads",
                    "4",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=300,
            )
            if result.returncode or not partial.is_file():
                raise RuntimeError(f"参考音声の登録に失敗しました。\n{result.stderr[-2500:]}")
            partial.replace(target)
        return f"ref_{key}", target

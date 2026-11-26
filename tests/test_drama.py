from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from drama import app, runtime
from scripts import stop_ui


class DramaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref = self.root / "reference.wav"
        with wave.open(str(self.ref), "wb") as wav:
            wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x00\x00" * 44100)

    def values(self, **overrides):
        values = {
            "mode": app.CLONE,
            "reference": str(self.ref),
            "transcript": "こんにちは。",
            "text": "[囁き]おやすみなさい。",
            "description": "",
            "seed": "0",
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 30,
        }
        values.update(overrides)
        return values

    def test_clone_requires_reference_and_exact_transcript(self):
        for change in ({"reference": None}, {"transcript": " "}, {"reference": "missing.wav"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                app.validate(**self.values(**change))

    def test_request_preserves_tags_seed_and_uses_supported_options(self):
        request, transcript = app.validate(**self.values(description="落ち着いた声"))
        self.assertEqual(request["text"], "[落ち着いた声] [囁き]おやすみなさい。")
        self.assertEqual(request["seed"], 0)
        self.assertEqual(transcript, "こんにちは。")
        self.assertNotIn("instructions", request)
        self.assertFalse(request["experts_on_cpu"])

    def test_invalid_text_and_numbers_fail_before_loading(self):
        for change in (
            {"text": ""},
            {"text": "あ" * 1001},
            {"text": "<|speaker:1|> test"},
            {"description": "[高い声]"},
            {"seed": "1.2"},
            {"seed": "-1"},
            {"seed": str(2**53)},
            {"temperature": float("nan")},
            {"top_p": 0},
            {"top_k": 1.2},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                app.validate(**self.values(**change))

    def test_design_ignores_previous_reference_and_reports_truncation(self):
        worker = Mock()
        worker.synthesize.return_value = ({"seed": 42, "stop": "model_limit"}, 1.0)
        with patch.object(app, "WORKER", worker), patch.object(app, "prepare_voice") as encode:
            events = list(app.generate(**self.values(mode=app.DESIGN), progress=Mock()))
            encode.assert_not_called()
            worker.add_voice.assert_not_called()
            self.assertEqual(worker.synthesize.call_args.args[0]["voice"], "none")
            self.assertEqual(events[0][:2], (None, None))
            self.assertIn("生成上限", events[-1][2])

    def test_clone_passes_encoded_voice_and_ui_has_serial_queue(self):
        worker = Mock()
        worker.synthesize.return_value = ({"seed": 0, "stop": "complete"}, 1.0)
        voice = self.root / "reference.voice.gguf"
        with (
            patch.object(app, "WORKER", worker),
            patch.object(app, "prepare_voice", return_value=("ref_hash", voice)) as encode,
        ):
            events = list(app.generate(**self.values(), progress=Mock()))
            encode.assert_called_once_with(str(self.ref), "こんにちは。")
            worker.add_voice.assert_called_once_with("ref_hash", voice)
            self.assertEqual(worker.synthesize.call_args.args[0]["voice"], "ref_hash")
            self.assertEqual(events[-1][0], events[-1][1])
        ui = app.build_ui()
        self.assertTrue(ui.enable_queue)
        self.assertEqual(ui.title, "Drama 日本語音声生成")
        queued = [fn for fn in ui.fns.values() if fn.fn in (app.generate, app.unload)]
        self.assertEqual(len(queued), 2)
        self.assertTrue(all(fn.concurrency_id == "drama" for fn in queued))

    def test_reference_encoding_runs_codec_on_cpu_and_caches_transcript(self):
        calls = []

        def run(command, **kwargs):
            if command[0] == "ffmpeg":
                Path(command[-1]).write_bytes(self.ref.read_bytes())
            else:
                calls.append(command)
                Path(command[4]).write_bytes(b"fixture voice")
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            patch.object(runtime, "STATE", self.root),
            patch.object(runtime.subprocess, "run", side_effect=run),
        ):
            first = runtime.prepare_voice(self.ref, "こんにちは。")
            second = runtime.prepare_voice(self.ref, "こんにちは。")
            changed = runtime.prepare_voice(self.ref, "こんばんは。")
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][calls[0].index("--device") + 1], "cpu")
        self.assertEqual(calls[0][calls[0].index("--transcript") + 1], "こんにちは。")

    def test_release_calls_real_upstream_cache_clear_and_closes_drama(self):
        module = types.ModuleType("irodori_tts.inference_runtime")
        module.clear_cached_runtime = Mock()
        with patch.dict(sys.modules, {module.__name__: module}):
            app_module = sys.modules["voice_clone_app"]
            app_module.unload_model()
        module.clear_cached_runtime.assert_called_once()
        worker = Mock()
        with patch.object(app, "WORKER", worker):
            app.unload()
            self.assertIsNone(app.WORKER)
        worker.close.assert_called_once()


class WorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.requests = self.root / "requests.jsonl"
        self.worker = runtime.Worker(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("fake_drama_worker.py")),
                str(self.requests),
            ],
            self.root / "worker.log",
            timeout=5,
        )
        self.addCleanup(self.worker.close)

    def test_persistent_worker_voice_cache_and_complete_wav(self):
        self.worker.add_voice("me", "reference.voice.gguf")
        pid = self.worker.process.pid
        self.worker.add_voice("me", "reference.voice.gguf")
        for index in range(2):
            target = self.root / f"out{index}.wav"
            terminal, seconds = self.worker.synthesize(
                {"text": "こんにちは。", "voice": "me", "seed": 0}, target
            )
            self.assertEqual(terminal["seed"], 0)
            self.assertEqual(seconds, 0.5)
            with wave.open(str(target)) as wav:
                self.assertEqual(
                    (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()),
                    (44100, 1, 2, 22050),
                )
            self.assertEqual(self.worker.process.pid, pid)
        requests = [
            json.loads(line) for line in self.requests.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([r["type"] for r in requests], ["add_voice", "synthesize", "synthesize"])

    def test_protocol_failures_remove_partial_audio_and_reset_worker(self):
        for behavior in ("bad_sequence", "bad_samples", "error", "eof"):
            with self.subTest(behavior=behavior):
                target = self.root / f"{behavior}.wav"
                with self.assertRaises((RuntimeError, EOFError)):
                    self.worker.synthesize({"text": behavior}, target)
                self.assertFalse(target.exists())
                self.assertFalse(target.with_suffix(".partial.wav").exists())
                self.assertIsNone(self.worker.process)

    def test_timeout_restarts_cleanly_without_stale_audio(self):
        self.worker.start()
        self.worker.timeout = 0.2
        with self.assertRaises(TimeoutError):
            self.worker.synthesize({"text": "timeout"}, self.root / "bad.wav")
        self.assertIsNone(self.worker.process)
        self.worker.timeout = 5
        result, seconds = self.worker.synthesize({"text": "正常"}, self.root / "good.wav")
        self.assertEqual(result["stop"], "complete")
        self.assertEqual(seconds, 0.5)


class ServiceStopTests(unittest.TestCase):
    def test_only_this_checkouts_selected_python_ui_matches(self):
        root = stop_ui.ROOT
        self.assertTrue(
            stop_ui.matches(root, ["/venv/bin/python", "voice_clone_app.py"], "irodori")
        )
        self.assertTrue(stop_ui.matches(root, ["python3", "-m", "drama.app"], "drama"))
        self.assertFalse(stop_ui.matches(root.parent, ["python", "voice_clone_app.py"], "irodori"))
        self.assertFalse(stop_ui.matches(root, ["python", "-m", "drama.app"], "irodori"))
        self.assertFalse(
            stop_ui.matches(root, ["uv", "run", "python", "voice_clone_app.py"], "irodori")
        )
        self.assertFalse(
            stop_ui.matches(root, ["python", "another.py", "voice_clone_app.py"], "irodori")
        )


if __name__ == "__main__":
    unittest.main()

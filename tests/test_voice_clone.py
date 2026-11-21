from __future__ import annotations

import ast
import dataclasses
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import voice_clone_app as app


def upstream_request_type():
    """Use the vendored request schema, without importing the GPU stack."""
    tree = ast.parse((app.ROOT / "irodori_tts/inference_runtime.py").read_text(encoding="utf-8"))
    request = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SamplingRequest"
    )
    source = "from __future__ import annotations\n" + ast.unparse(request)
    namespace = {"dataclass": dataclasses.dataclass}
    exec(compile(source, "upstream_sampling_request", "exec"), namespace)
    return namespace["SamplingRequest"]


class VoiceCloneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ref = Path(self.temp.name) / "reference.wav"
        self.ref.write_bytes(b"fixture")

    def values(self, **overrides):
        data = {
            "ref_files": [str(self.ref)],
            "text": "こんにちは。",
            "caption": "",
            "steps": 40,
            "seed": "0",
            "duration_scale": 1.0,
            "precision": "fp32",
        }
        data.update(overrides)
        return app.validate_inputs(**data)

    def test_reference_is_required_and_missing_file_is_rejected(self):
        for paths in ([], None, [str(self.ref.with_name("missing.wav"))]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                self.values(ref_files=paths)

    def test_empty_and_long_text_are_rejected_before_model_load(self):
        for text in (None, "  ", "あ" * 201):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.values(text=text)

    def test_invalid_numbers_do_not_reach_sampling(self):
        for change in (
            {"seed": "-1"},
            {"seed": "1.5"},
            {"seed": str(2**32)},
            {"steps": 0},
            {"steps": 10.5},
            {"steps": float("nan")},
            {"duration_scale": float("inf")},
            {"precision": "fp16"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.values(**change)

    def test_cloning_request_matches_real_upstream_schema(self):
        request = upstream_request_type()(**self.values())
        self.assertFalse(request.no_ref)
        self.assertEqual(request.ref_wavs, [str(self.ref)])
        self.assertEqual(request.seed, 0)
        self.assertIsNone(request.caption)
        self.assertEqual(request.num_candidates, 1)

    def test_random_seed_and_reference_order(self):
        second = self.ref.with_name("second.wav")
        second.write_bytes(b"fixture")
        values = self.values(ref_files=[str(second), str(self.ref)], seed="", caption=" やさしく ")
        self.assertIsNone(values["seed"])
        self.assertEqual(values["ref_wavs"], [str(second), str(self.ref)])
        self.assertEqual(values["caption"], "やさしく")

    def test_generation_passes_reference_and_saves_unique_wav(self):
        audio = Mock()
        audio.float.return_value = audio
        runtime = Mock()
        runtime.synthesize.return_value = types.SimpleNamespace(
            audio=audio, sample_rate=48000, used_seed=0, messages=[]
        )
        saved = []

        def save(path, value, rate):
            self.assertIs(value, audio)
            self.assertEqual(rate, 48000)
            saved.append(path)
            return path

        module = types.ModuleType("irodori_tts.inference_runtime")
        module.SamplingRequest = upstream_request_type()
        module.save_wav = save
        with (
            patch.dict(sys.modules, {module.__name__: module}),
            patch.object(app, "load_runtime", return_value=runtime),
            patch.object(app, "OUTPUT_DIR", Path(self.temp.name)),
        ):
            for _ in range(2):
                events = list(
                    app.generate(
                        [str(self.ref)], "こんにちは。", "", 40, "0", 1.0, "fp32", progress=Mock()
                    )
                )
                self.assertEqual(events[0][:2], (None, None))
                self.assertEqual(events[-1][0], events[-1][1])
                self.assertIn("48000 Hz", events[-1][2])
                self.assertEqual(runtime.synthesize.call_args.args[0].ref_wavs, [str(self.ref)])
                self.assertFalse(runtime.synthesize.call_args.args[0].no_ref)
        self.assertNotEqual(saved[0], saved[1])

    def test_japanese_ui_builds_with_serial_queue(self):
        ui = app.build_ui()
        self.assertEqual(ui.title, "Irodori 音声クローン")
        labels = [c.get("props", {}).get("label") for c in ui.config["components"]]
        self.assertIn("2. 読ませたい日本語", labels)
        self.assertTrue(ui.enable_queue)


if __name__ == "__main__":
    unittest.main()

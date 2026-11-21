#!/usr/bin/env python3
"""Japanese reference-audio cloning UI using the unmodified Irodori runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import threading
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Aratako/Irodori-TTS-v4.1-Small"
OUTPUT_DIR = ROOT / "outputs" / "audio"
GENERATION_LOCK = threading.Lock()


def validate_inputs(ref_files, text, caption, steps, seed, duration_scale, precision):
    paths = [str(p) for p in (ref_files or [])]
    if not paths:
        raise ValueError("参考音声をアップロードしてください。")
    if len(paths) > 16:
        raise ValueError("参考音声は一度に16ファイルまでです。")
    if any(not Path(p).is_file() for p in paths):
        raise ValueError("参考音声が見つかりません。もう一度アップロードしてください。")
    text = (text or "").strip()
    if not text:
        raise ValueError("読ませたい日本語の文章を入力してください。")
    if len(text) > 200:
        raise ValueError("文章は200文字以内に分けてください。最初は1〜2文がおすすめです。")
    caption = (caption or "").strip()
    if len(caption) > 200:
        raise ValueError("話し方の指定は200文字以内にしてください。")
    if not math.isfinite(float(steps)) or int(steps) != float(steps) or not 4 <= int(steps) <= 80:
        raise ValueError("生成ステップ数は4〜80の整数にしてください。")
    seed_text = str(seed if seed is not None else "").strip()
    try:
        seed_value = int(seed_text) if seed_text else None
    except ValueError as exc:
        raise ValueError("Seedは空欄か0以上の整数にしてください。") from exc
    if seed_value is not None and not 0 <= seed_value < 2**32:
        raise ValueError("Seedは0〜4294967295にしてください。")
    scale = float(duration_scale)
    if not math.isfinite(scale) or not 0.7 <= scale <= 1.4:
        raise ValueError("長さの倍率は0.7〜1.4にしてください。")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("演算精度はfp32またはbf16を選んでください。")
    return {
        "text": text,
        "caption": caption or None,
        "ref_wavs": paths,
        "no_ref": False,
        "num_steps": int(steps),
        "seed": seed_value,
        "duration_scale": scale,
        "num_candidates": 1,
        "decode_mode": "sequential",
        "trim_tail": True,
    }


def load_runtime(precision):
    # Lazy imports let the UI and input checks be tested without downloading weights.
    import torch

    from irodori_tts.inference_runtime import RuntimeKey, download_hf_checkpoint, get_cached_runtime

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUが見つかりません。GPU付きRunPodで起動してください。")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("このGPUはbf16に対応していません。fp32を選んでください。")
    key = RuntimeKey(
        checkpoint=download_hf_checkpoint(MODEL_ID),
        model_device="cuda",
        model_precision=precision,
        codec_device="cuda",
        codec_precision="fp32",
        compile_model=False,
        compile_dynamic=False,
    )
    runtime, _ = get_cached_runtime(key)
    if not runtime.model_cfg.use_speaker_condition_resolved:
        raise RuntimeError("音声クローンに対応したモデルを読み込めませんでした。")
    return runtime


def generate(
    ref_files, text, caption, steps, seed, duration_scale, precision, progress=gr.Progress()
):
    # Clear the preceding result before a new attempt, including validation failures.
    yield None, None, "入力を確認しています。"
    try:
        values = validate_inputs(ref_files, text, caption, steps, seed, duration_scale, precision)
        with GENERATION_LOCK:
            progress(0.05, desc="モデルを準備中（初回はダウンロードが必要です）")
            runtime = load_runtime(precision)
            from irodori_tts.inference_runtime import SamplingRequest, save_wav

            progress(0.25, desc="参考音声の声で生成しています")
            started = time.perf_counter()
            result = runtime.synthesize(
                SamplingRequest(**values), log_fn=lambda msg: print(msg, flush=True)
            )
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output = save_wav(
                OUTPUT_DIR / f"voice_{uuid4().hex}.wav", result.audio.float(), result.sample_rate
            )
            elapsed = time.perf_counter() - started
            progress(1, desc="生成完了")
            details = (
                f"生成完了 · {elapsed:.1f}秒 · Seed: {result.used_seed} · {result.sample_rate} Hz"
            )
            if result.messages:
                details += "\n" + "\n".join(result.messages)
            yield str(output), str(output), details
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        import traceback

        traceback.print_exc()
        if "out of memory" in str(exc).lower():
            raise gr.Error(
                "GPUメモリ不足です。参考音声と文章を短くするか、詳細設定でbf16を選んでください。"
            ) from exc
        raise gr.Error(f"生成に失敗しました。起動ログを確認してください。\n{exc}") from exc


def build_ui():
    with gr.Blocks(title="Irodori 音声クローン") as demo:
        gr.Markdown("# Irodori 音声クローン\n参考音声の声で、好きな日本語の文章を読み上げます。")
        with gr.Row():
            with gr.Column():
                refs = gr.File(
                    label="1. 参考音声（同じ人の声・複数可）",
                    file_count="multiple",
                    file_types=["audio"],
                    type="filepath",
                )
                gr.Markdown(
                    "雑音やBGMの少ない音声を使ってください。短い音声を複数、合計30秒程度がおすすめです。最大120秒まで使われます。参考音声の文字起こしは不要です。"
                )
                text = gr.Textbox(
                    label="2. 読ませたい日本語",
                    lines=5,
                    max_lines=12,
                    placeholder="こんにちは。今日はとてもいい天気ですね。",
                    info="まずは1〜2文で試してください。1回の生成は最大30秒程度です。長文は分けて入力してください。",
                )
                with gr.Accordion("詳細設定", open=False):
                    caption = gr.Textbox(
                        label="話し方（任意）",
                        placeholder="落ち着いて、やさしく話す",
                        info="最初は空欄がおすすめです。",
                    )
                    steps = gr.Slider(4, 80, value=40, step=1, label="生成ステップ数")
                    duration = gr.Slider(
                        0.7,
                        1.4,
                        value=1.0,
                        step=0.05,
                        label="長さの倍率",
                        info="大きいほど長め、小さいほど短めになります。",
                    )
                    seed = gr.Textbox(label="Seed（空欄なら毎回変わります）", value="")
                    precision = gr.Radio(
                        ["fp32", "bf16"],
                        value="fp32",
                        label="演算精度",
                        info="標準はfp32。GPUメモリを節約したい場合はbf16。",
                    )
                button = gr.Button("3. この声で生成する", variant="primary")
            with gr.Column():
                audio = gr.Audio(label="生成した音声", type="filepath", interactive=False)
                download = gr.File(label="WAVをダウンロード", interactive=False)
                status = gr.Textbox(
                    label="状況",
                    value="参考音声と文章を入れて、生成ボタンを押してください。",
                    lines=4,
                    interactive=False,
                )
        gr.Markdown(
            "Irodori-TTS v4.1-Small · 初回生成時にモデルをダウンロードします。2回目からは読み込んだモデルを再利用します。"
        )
        button.click(
            generate,
            inputs=[refs, text, caption, steps, seed, duration, precision],
            outputs=[audio, download, status],
            concurrency_limit=1,
            api_name=False,
        )
    return demo.queue(default_concurrency_limit=1, max_size=8)


def load_login():
    """Keep generated credentials out of source control and out of the browser UI."""
    username = os.environ.get("IRODORI_USER", "irodori")
    password = os.environ.get("IRODORI_PASSWORD")
    if password:
        return username, password
    path = ROOT / ".runtime" / "login.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        credentials = json.loads(path.read_text(encoding="utf-8"))
    else:
        credentials = {"username": username, "password": secrets.token_urlsafe(18)}
        with path.open("x", encoding="utf-8") as handle:
            os.chmod(path, 0o600)
            json.dump(credentials, handle)
    print(
        f"Login user: {credentials['username']}\nLogin password: {credentials['password']}",
        flush=True,
    )
    return credentials["username"], credentials["password"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument(
        "--check-ui", action="store_true", help="Build UI only; no GPU/model download."
    )
    args = parser.parse_args()
    demo = build_ui()
    if args.check_ui:
        print("UI OK (no inference performed)")
        return
    auth = load_login()
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        print(f"Open: https://{pod_id}-{args.server_port}.proxy.runpod.net", flush=True)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=False,
        auth=auth,
        show_error=True,
        max_file_size="100mb",
    )


if __name__ == "__main__":
    main()

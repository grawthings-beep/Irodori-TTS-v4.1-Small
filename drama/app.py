"""Japanese Gradio UI for Drama 3 reference cloning and voice design."""

from __future__ import annotations

import argparse
import atexit
import math
import os
import threading
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr

from drama.runtime import OUTPUTS, make_worker, prepare_voice
from voice_clone_app import load_login

LOCK = threading.Lock()
WORKER = None
CLONE = "参考音声の声で読む"
DESIGN = "声の説明から作る"
TAGS = [
    "囁き",
    "吐息",
    "ため息",
    "くすくす笑い",
    "笑いながら",
    "驚いて",
    "怒りを込めて",
    "優しく",
    "泣きながら",
    "耳元で",
]


def validate(mode, reference, transcript, text, description, seed, temperature, top_p, top_k):
    if mode not in {CLONE, DESIGN}:
        raise ValueError("生成方法を選んでください。")
    text = (text or "").strip()
    transcript = (transcript or "").strip()
    description = (description or "").strip()
    if not text or len(text) > 1000:
        raise ValueError(
            "読ませたい日本語を1〜1000文字で入力してください。まずは1〜2文で試してください。"
        )
    if len(description) > 200 or any(c in description for c in "[]\n\r"):
        raise ValueError("声・演技の説明は200文字以内の1行にしてください。[ ]は入力不要です。")
    for value in (text, description, transcript):
        if "<|" in value or "\x00" in value:
            raise ValueError(
                "内部制御トークンやNULL文字は入力できません。演技は[囁き]などで指定してください。"
            )
    if mode == CLONE:
        if not reference or not Path(reference).is_file():
            raise ValueError("参考音声をアップロードしてください。")
        if not transcript or len(transcript) > 2000:
            raise ValueError("参考音声で話している内容の正確な文字起こしを入力してください。")
    seed_text = str(seed if seed is not None else "").strip()
    try:
        seed_value = int(seed_text) if seed_text else None
    except ValueError as exc:
        raise ValueError("Seedは空欄か0以上の整数にしてください。") from exc
    if seed_value is not None and not 0 <= seed_value <= 2**53 - 1:
        raise ValueError("Seedは0〜9007199254740991にしてください。")
    temperature, top_p, k = float(temperature), float(top_p), float(top_k)
    if not math.isfinite(temperature) or not 0.2 <= temperature <= 1.5:
        raise ValueError("Temperatureは0.2〜1.5にしてください。")
    if not math.isfinite(top_p) or not 0.1 <= top_p <= 1:
        raise ValueError("Top-pは0.1〜1にしてください。")
    if not math.isfinite(k) or not k.is_integer() or not 1 <= k <= 100:
        raise ValueError("Top-kは1〜100の整数にしてください。")
    request = {
        "text": f"[{description}] {text}" if description else text,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": int(k),
        "experts_on_cpu": False,
    }
    if seed_value is not None:
        request["seed"] = seed_value
    return request, transcript


def unload():
    global WORKER
    with LOCK:
        if WORKER:
            WORKER.close()
            WORKER = None
    return "Dramaのモデルを終了してGPUを解放しました。次回生成時に再び読み込みます。"


def generate(
    mode,
    reference,
    transcript,
    text,
    description,
    seed,
    temperature,
    top_p,
    top_k,
    progress=gr.Progress(),
):
    global WORKER
    yield None, None, "入力を確認しています。"
    try:
        request, transcript = validate(
            mode, reference, transcript, text, description, seed, temperature, top_p, top_k
        )
        with LOCK:
            if mode == CLONE:
                progress(0.05, desc="参考音声を登録中（同じ音声と文字起こしは再利用します）")
                voice_name, voice_path = prepare_voice(reference, transcript)
            else:
                voice_name, voice_path = "none", None
            progress(0.15, desc="Dramaを読み込んでいます（初回・解放後のみ）")
            if WORKER is None:
                WORKER = make_worker()
            WORKER.start()
            if voice_path:
                WORKER.add_voice(voice_name, voice_path)
            request["voice"] = voice_name
            progress(0.25, desc="音声を生成中")
            started = time.perf_counter()
            output = OUTPUTS / f"drama_{uuid4().hex}.wav"
            result, seconds = WORKER.synthesize(
                request,
                output,
                progress=lambda p: progress(
                    0.25 + 0.7 * min(1.0, max(0.0, p)), desc="音声を生成中"
                ),
            )
            elapsed = time.perf_counter() - started
            status = f"生成完了 · 音声 {seconds:.2f}秒 · 合成 {elapsed:.2f}秒（準備時間を除く） · Seed: {result.get('seed')}"
            if result.get("stop") != "complete":
                status += "\n生成上限に達しました。文の最後まで読めているか確認し、短く分けて再生成してください。"
            progress(1, desc="生成完了")
            yield str(output), str(output), status
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise gr.Error(str(exc)) from exc


def add_tag(text, tag):
    return (text or "") + f"[{tag}]" if tag in TAGS else (text or "")


def build_ui():
    with gr.Blocks(title="Drama 日本語音声生成") as demo:
        gr.Markdown("# Drama 日本語音声生成\n声を再現し、台詞の演技を日本語タグで指定できます。")
        mode = gr.Radio([CLONE, DESIGN], value=CLONE, label="生成方法")
        with gr.Row():
            with gr.Column():
                reference = gr.Audio(
                    label="参考音声（0.5〜30秒）",
                    sources=["upload", "microphone"],
                    type="filepath",
                    format="wav",
                )
                transcript = gr.Textbox(
                    label="参考音声の文字起こし",
                    lines=3,
                    info="参考音声で実際に話している内容を正確に入力します。新しく読ませる台詞とは別です。",
                )
                text = gr.Textbox(
                    label="読ませたい日本語・演技タグ",
                    lines=5,
                    placeholder="[囁き]こんにちは。[笑いながら]今日もいい天気ですね。",
                )
                with gr.Row():
                    tag = gr.Dropdown(TAGS, value="囁き", label="演技タグ")
                    insert = gr.Button("文章の末尾にタグを追加")
                description = gr.Textbox(
                    label="声・演技の説明（任意）",
                    placeholder="落ち着いた声で、やさしく話す",
                    info="クローンでは参考音声と矛盾しない説明を使います。",
                )
                with gr.Accordion("詳細設定", open=False):
                    seed = gr.Textbox(label="Seed（空欄はランダム）")
                    temperature = gr.Slider(0.2, 1.5, value=1.0, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                    top_k = gr.Slider(1, 100, value=30, step=1, label="Top-k")
                button = gr.Button("この内容で生成する", variant="primary")
            with gr.Column():
                audio = gr.Audio(label="生成音声", interactive=False)
                download = gr.File(label="WAVをダウンロード", interactive=False)
                status = gr.Textbox(
                    label="状況",
                    lines=5,
                    interactive=False,
                    value="生成完了後に再生できます。Irodoriと比べる場合は、先にIrodoriのGPUを解放してください。",
                )
                release = gr.Button("DramaのGPUを解放する")
        gr.Markdown(
            "Built with Fish Audio · Aratako/Drama-3-Japanese · StudioYebisu GGUF\n\n"
            "[公式タグ一覧](https://huggingface.co/Aratako/Drama-3-Japanese/blob/main/TAG_LIST.md) · "
            "[モデルのライセンス](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE)"
        )
        mode.change(
            lambda choice: (gr.update(visible=choice == CLONE), gr.update(visible=choice == CLONE)),
            inputs=mode,
            outputs=[reference, transcript],
            api_name=False,
            queue=False,
        )
        insert.click(add_tag, [text, tag], text, api_name=False, queue=False)
        button.click(
            generate,
            [mode, reference, transcript, text, description, seed, temperature, top_p, top_k],
            [audio, download, status],
            api_name=False,
            concurrency_id="drama",
            concurrency_limit=1,
        )
        release.click(
            unload, outputs=status, api_name=False, concurrency_id="drama", concurrency_limit=1
        )
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--check-ui", action="store_true")
    args = parser.parse_args()
    demo = build_ui()
    if args.check_ui:
        print("Drama UI OK; no model inference performed.")
        return
    atexit.register(unload)
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        print(f"Drama: https://{pod_id}-{args.server_port}.proxy.runpod.net", flush=True)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=False,
        auth=load_login(),
        max_file_size="50mb",
        show_error=True,
    )


if __name__ == "__main__":
    main()

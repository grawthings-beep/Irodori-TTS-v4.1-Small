# Irodori-TTS v4.1-Small 音声クローン / RunPod

**参考音声をアップロードし、入力した日本語をその声で読み上げる**ための日本語UIです。追加学習も参考音声の文字起こしも不要です。

公式 [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) の推論処理をそのまま利用しています。音声変換・リアルタイムボイスチェンジャーではなく、参考音声から話者の声を条件付けして文章を新たに読み上げます。

**Drama 3の演技タグ・音声クローンを使う場合は [Dramaの導入手順](README_DRAMA.md) を参照してください。** 同じPodの7861番ポートに追加でき、Irodoriとは専用環境を分けています。両方の画面にGPU解放ボタンがあり、モデルを切り替えて使えます。

## 1. RunPodを用意

- GPU付きPod。初めての構成案は **RTX 4090 24GBなど、VRAM 24GBのGPU**。これは余裕を持たせるための選定で、実測した最低要件ではありません。
- Ubuntu 22.04 / 24.04系の **RunPod PyTorchテンプレート**。CUDA 12.8対応の新しいNVIDIAドライバーが必要です。
- `/workspace` の保存領域は **50GB以上**を目安にしてください。モデル・専用Python環境・キャッシュ・生成音声を保存します。
- **Expose HTTP Portsに `7860` を追加**してください。Jupyter用の既存ポートは残します。

GPUのレンタルは利用者がRunPod上で行います。このリポジトリはPodを自動購入・作成しません。

## 2. 端末に貼り付けて起動

PodのWeb TerminalまたはJupyter Terminalに、以下をそのまま貼り付けます。

```bash
cd /workspace && (test -d Irodori-TTS-v4.1-Small || git clone https://github.com/grawthings-beep/Irodori-TTS-v4.1-Small.git) && cd Irodori-TTS-v4.1-Small && mkdir -p logs && (umask 077; nohup bash scripts/runpod_start.sh >> logs/runpod.log 2>&1 < /dev/null &) && tail -n 80 -F logs/runpod.log
```

初回は依存パッケージを取得するため時間がかかります。`Running on local URL` が出たら、RunPodの **Connect → HTTP Services → Port 7860** を開きます。

起動ログに `Login user` と `Login password` が表示されます。その情報でログインしてください。初回に作ったパスワードは `.runtime/login.json` に保存され、再起動後も同じものを使います。RunPodの環境変数 `IRODORI_USER` / `IRODORI_PASSWORD` で設定することもできます。その場合は環境変数のパスワードを使います。

ログ表示を終えるには **Ctrl+C**。サーバーはバックグラウンドで動き続けます。再起動は同じコマンドで行えます。起動済みなら二重にサーバーを立てません。既存チェックアウトは自動更新しません。

## 3. 音声を生成

1. 「参考音声」に、再現したい声の音声ファイルを入れます。WAV推奨、複数ファイルも使えます。
2. 「読ませたい日本語」に文章を入力します。最初は1〜2文で試してください。
3. 「この声で生成する」を押します。初回生成時にv4.1-Small・音声コーデックなどをダウンロードします。
4. 音声を再生し、「WAVをダウンロード」から保存します。

公式の推奨は、同じ話者の雑音やBGMの少ない短い音声を複数、合計約30秒にする方法です。参考音声は合計最大120秒まで使われます。入力内容をそのままコピーするのではなく、声を参考にして入力文章を新しく読み上げます。元の録音の抑揚・タイミングを厳密に保つものではありません。

1回の生成は最大30秒程度です。200文字以下でも読む速さによっては収まらないため、長文は文ごとに分けてください。声が似ないときは、まず参考音声の品質や長さを調整し、詳細設定の話し方は空欄で試してください。

詳細設定で話し方、生成ステップ、長さの倍率、Seedを調整できます。メモリが足りない場合は参考音声と文章を短くするか、演算精度を `bf16` にします。

## 保存先・ログ

| 内容 | Pod内の場所 |
| --- | --- |
| 生成WAV | `/workspace/Irodori-TTS-v4.1-Small/outputs/audio/` |
| 起動・推論ログ | `/workspace/Irodori-TTS-v4.1-Small/logs/runpod.log` |
| モデルキャッシュ | `/workspace/Irodori-TTS-v4.1-Small/.cache/huggingface/` |
| ログイン情報 | `/workspace/Irodori-TTS-v4.1-Small/.runtime/login.json` |

保存領域の保持はRunPodのVolume設定に依存します。必要なWAVはPod削除前にダウンロードしてください。課金を止めたいときは、Web画面を閉じるだけでなくRunPod側でPodを停止してください。

## 起動時に問題が出たら

- **Port 7860が開かない**: HTTPポートに7860があることと、ログに起動完了が出ていることを確認します。
- **CUDA is unavailable / driverエラー**: GPU付きPodとNVIDIAドライバーを確認します。この構成は専用の `.venv` に公式のCUDA 12.8版PyTorchを入れます。
- **TorchCodec / FFmpegエラー**: ログを確認します。起動スクリプトはFFmpegとlibsndfileを用意し、実際にWAVの読み込みを検証します。
- **GPUメモリ不足**: 他のGPUアプリを終了し、参考音声と文章を短くするかbf16に切り替えます。
- **途中で読み上げが切れる**: 文章を短く分けて再生成します。

## 検証範囲

ローカルでは7件のテストで入力検証、日本語UI構築、公式のリクエスト定義との互換性、参考音声の受け渡しと保存処理を検証済みです（音声合成部分はテスト用の代替処理）。Bash構文と、HTTP経由でのログイン必須・誤パスワード拒否・ログイン後のUI応答も確認しました。**RunPodでの依存環境全体のインストール、モデルダウンロード・GPUによる音声生成・声の再現品質は未検証です。** 起動時に `scripts/preflight.py` が実際のIrodori/DACVAEインポート、CUDA行列演算、音声デコーダーを検証し、失敗した場合はサーバーを起動しません。

```bash
uv run --no-sync python -m unittest discover -s tests -p test_voice_clone.py -v
uv run --no-sync python voice_clone_app.py --check-ui
```

## 出典・ライセンス

- 公式コードの収録元: `Aratako/Irodori-TTS` / `f01c08cd40c0afa62a862439a5de12f6dcf62054`
- モデル: [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)
- 公式の説明: [README.upstream.md](README.upstream.md)
- MIT License: [LICENSE](LICENSE)。元の著作権表示を保持しています。
- [RunPodのポート公開手順](https://docs.runpod.io/pods/configuration/expose-ports)

追加ファイルは `voice_clone_app.py`、`drama/`、`scripts/`、`tests/`、CI設定と日本語手順です。Irodori公式の推論エンジン、ルートの`pyproject.toml`、`uv.lock`は変更していません。SDXL/LoRA用の依存パッケージ固定は使用しません。Dramaのモデルと外部パッチのライセンスは[別途記載](README_DRAMA.md#出典とライセンス)しています。

# Drama 3 日本語音声生成 / RunPod

参考音声とその文字起こしから声を再現し、日本語の台詞を生成する画面です。参考音声なしで、声の説明から生成することもできます。`[囁き]`・`[吐息]`・`[笑いながら]`などの演技タグを入力できます。

**Built with Fish Audio.** [Aratako/Drama-3-Japanese](https://huggingface.co/Aratako/Drama-3-Japanese) の [StudioYebisu GGUFパック](https://huggingface.co/StudioYebisu-ship-it/Drama-3-Japanese-speech-cpp-gguf) を使います。

## RunPodの設定

既存のIrodori Podに追加できます。CUDAコンパイラーを含む次のイメージを想定しています。

**Container image**

```text
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
```

| 設定 | 値 |
| --- | --- |
| GPU | RTX 4090など24GB VRAMを推奨。実測の最低要件ではありません |
| Container disk | 20 GB |
| Persistent storage | Volume disk、50 GB以上。既存データで埋まっている場合は増量 |
| Persistent storage mount path | `/workspace` |
| Container start command | イメージ既定のまま |

**HTTP Ports** はラベルと番号を別々の欄に入れます。既存の7860・8888を残し、7861を追加します。

| Port label | Port number | 用途 |
| --- | --- | --- |
| Irodori | `7860` | Irodoriの音声クローン画面 |
| Drama | `7861` | 今回追加するDramaの画面 |
| Jupyter | `8888` | Jupyter Lab・ターミナル |

**Environment variables**

| Key | Value |
| --- | --- |
| `IRODORI_USER` | `irodori` |
| `IRODORI_PASSWORD` | 自分で決めたパスワード（Dramaも共通） |
| `DRAMA_PORT` | `7861` |

既に設定済みの認証情報はそのまま使えます。未設定なら起動ログに表示される認証情報を使います。Jupyterの認証は別です。モデル取得にHugging Faceトークンは不要です。

## 起動コマンド

PodのJupyter TerminalまたはWeb Terminalに貼り付けます。既存のチェックアウトは`git pull --ff-only`で更新します。ローカル編集で更新できない場合は停止するので、ログを確認してください。

```bash
cd /workspace && (test -d Irodori-TTS-v4.1-Small || git clone https://github.com/grawthings-beep/Irodori-TTS-v4.1-Small.git) && cd Irodori-TTS-v4.1-Small && git pull --ff-only && bash scripts/launch_drama.sh
```

初回は約9.76 GBのモデルを取得し、speech.cppをCUDA対応でビルドします。`Running on local URL`が出たら **Connect → HTTP Services → Drama / Port 7861** を開きます。UIの起動後、初回生成時にモデルをGPUに読み込みます。

**Ctrl+Cはログ表示だけを終了**します。サーバーは動き続けます。同じコマンドを再実行してもサーバーを二重起動しません。コード更新を既に起動しているDramaへ反映する場合は、生成終了後に次を実行します。

```bash
cd /workspace/Irodori-TTS-v4.1-Small
python3 scripts/stop_ui.py drama
bash scripts/launch_drama.sh
```

## 生成する

1. 「参考音声の声で読む」を選び、雑音・BGMの少ない同じ話者の音声を入れます。0.5〜30秒、最初は5〜15秒程度を目安にします。
2. 「参考音声の文字起こし」に、その録音で実際に話している言葉を正確に入力します。Irodoriと異なり、この文字起こしが必要です。
3. 「読ませたい日本語・演技タグ」に新しい台詞を入力します。例：`[囁き]こんにちは。[笑いながら]今日もいい天気ですね。`
4. 「この内容で生成する」を押します。完成したWAVを再生・ダウンロードできます。

声の説明から作る場合は「声の説明から作る」に切り替え、「声・演技の説明」に`落ち着いた声で、やさしく話す`などを入力します。同じ参考音声と文字起こしは登録結果を再利用します。

まず1〜2文で試してください。生成上限に達すると画面に表示します。末尾まで読まれていない場合は短く分けて生成してください。タグの効き方や声の再現度は音声・文章・Seedに依存します。

この画面は**完成した音声を返す方式**です。パック公開者が報告した最初の音まで約0.3秒という値は、この画面の待ち時間を保証するものではありません。

## Irodoriと切り替える

両方のUIを開けますが、モデルは片方ずつGPUに載せて使います。

- Irodori → Drama：Irodori画面の **「IrodoriのGPUを解放する」** を押してからDramaで生成。
- Drama → Irodori：Drama画面の **「DramaのGPUを解放する」** を押してからIrodoriで生成。

古いIrodori画面に解放ボタンがない場合は、生成終了後に一度だけIrodoriを再起動します。次の停止処理はこのフォルダーから起動したIrodori UIだけを対象にします。モデル・生成音声は消しません。

```bash
cd /workspace/Irodori-TTS-v4.1-Small
python3 scripts/stop_ui.py irodori
(umask 077; nohup bash scripts/runpod_start.sh >> logs/runpod.log 2>&1 < /dev/null &)
tail -n 80 -F logs/runpod.log
```

## 保存先と調整

| 内容 | リポジトリからの場所 |
| --- | --- |
| 生成WAV | `outputs/drama/` |
| 起動・ビルドログ | `logs/drama.log` |
| モデル読み込み・合成ログ | `logs/drama-worker.log` |
| モデル・ライセンス・パッチ | `.runtime/drama/assets/` |
| 登録した参考音声の特徴量と文字起こし | `.runtime/drama/voices/` |
| Drama専用Python環境 | `.runtime/drama/venv/` |

起動時にモデル・パッチのSHA-256を照合します。実行エンジンはspeech.cpp v0.11.2と対応するggmlに固定しています。固定値は[drama/pins.json](drama/pins.json)を参照してください。IrodoriのPyTorch環境は変更しません。

- **`CUDA compiler missing`**：CUDA開発ツールを含むイメージが必要です。上記イメージを確認してください。
- **GPUメモリ不足**：他方の画面でGPUを解放します。Drama起動前の空きVRAMが12 GiB未満なら明示的に止めます。12 GiBでの動作を保証する検査ではありません。
- **ビルド中にメモリ不足**：環境変数`DRAMA_BUILD_JOBS=1`で再起動します。標準は2です。
- **ダウンロード失敗**：同じコマンドを再実行すると途中から再開します。容量不足なら`/workspace`のVolumeを増やしてください。
- **演技の間を短くしたい**：任意で`DRAMA_MAX_PAUSE_FRAMES=18`を設定してDramaを再起動します。標準は`0`で間の強制短縮を無効にしています。パックの高速化処理は有効ですが、この追加の間短縮は演技にも影響するため別設定です。

## 検証範囲

モデル推論を代替したテストで、音声登録、文字起こし、演技タグ、WAV保存、継続利用、タイムアウト・破損データからの復帰を検査します。GitHub Actionsではロック済み環境で両UIをテストし、公開パッチ適用後のspeech.cppをLinuxのCPU構成でビルドします。

**RunPodのCUDAビルド・実モデルによる音声合成・声の再現品質は、Pod上での確認が必要です。** CPUビルドやUIテストはGPU推論の検証を代替しません。

## 出典とライセンス

UIと起動コードにはこのリポジトリのMIT Licenseを適用します。モデル・外部パッチの利用条件は別です。

- [StudioYebisuパックのモデルライセンス（固定版）](https://huggingface.co/StudioYebisu-ship-it/Drama-3-Japanese-speech-cpp-gguf/blob/cca4c0d8cfab67a2bccbb634d52e544199ac5108/LICENSE)
- [パックのNOTICE（固定版）](https://huggingface.co/StudioYebisu-ship-it/Drama-3-Japanese-speech-cpp-gguf/blob/cca4c0d8cfab67a2bccbb634d52e544199ac5108/NOTICE)
- [speech.cpp v0.11.2](https://github.com/nyosegawa/speech.cpp/tree/v0.11.2)
- [Aratako公式の演技タグ](https://huggingface.co/Aratako/Drama-3-Japanese/blob/main/TAG_LIST.md)

配布パックに含まれるLICENSE・NOTICE・LICENSE-SPEECH-CPPをモデルと一緒に保存します。紹介文とLICENSE本文では商用利用条件の説明が異なるため、利用条件はリンク先の本文を確認してください。

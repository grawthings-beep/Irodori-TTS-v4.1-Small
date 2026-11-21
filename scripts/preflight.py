"""Exercise the actual imports, audio decoder and CUDA before launching the UI."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import gradio
    import soundfile as sf
    import torch
    import torchaudio
    import torchcodec
    from dacvae import DACVAE

    from irodori_tts.inference_runtime import RuntimeKey, SamplingRequest

    assert DACVAE and RuntimeKey and SamplingRequest
    print(
        f"Python {sys.version.split()[0]} | torch {torch.__version__} | torchaudio {torchaudio.__version__}"
    )
    print(f"Gradio {gradio.__version__} | TorchCodec {torchcodec.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a GPU Pod and check the NVIDIA driver.")
    value = torch.ones((64, 64), device="cuda")
    assert (value @ value).mean().item() == 64.0
    torch.cuda.synchronize()
    print(f"GPU computation OK: {torch.cuda.get_device_name(0)}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "decoder_check.wav"
        sf.write(path, [0.0] * 4800, 48000)
        audio, rate = torchaudio.load(str(path))
        assert rate == 48000 and audio.shape[-1] == 4800
    print("Irodori/DACVAE imports and audio decoding OK. Model inference has not run yet.")


if __name__ == "__main__":
    main()

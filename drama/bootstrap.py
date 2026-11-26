"""Fetch checksum-verified assets and the pinned, patched speech.cpp source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".runtime" / "drama"
PINS = json.loads(Path(__file__).with_name("pins.json").read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*args, cwd=None):
    subprocess.run(list(args), cwd=cwd, check=True)


def download(asset):
    target = STATE / "assets" / asset["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != asset["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {target}. Move this file aside and rerun.")
        return target
    partial = target.with_name(target.name + ".part")
    remaining = max(0, asset["bytes"] - (partial.stat().st_size if partial.exists() else 0))
    if shutil.disk_usage(target.parent).free < remaining + 2 * 1024**3:
        raise RuntimeError(
            "Not enough free storage for the download. Increase the /workspace volume."
        )
    url = f"https://huggingface.co/{PINS['pack_repo']}/resolve/{PINS['pack_revision']}/{asset['name']}"
    print(f"Downloading {asset['name']}", flush=True)
    run("curl", "-fL", "--retry", "3", "--retry-delay", "2", "-C", "-", "-o", str(partial), url)
    if sha256(partial) != asset["sha256"]:
        partial.unlink()
        raise RuntimeError(
            f"Downloaded checksum mismatch: {asset['name']}. Rerun to download again."
        )
    partial.replace(target)
    return target


def prepare(source_only=False):
    for asset in PINS["assets"]:
        if not (source_only and asset["name"].endswith(".gguf")):
            download(asset)
    source = STATE / "speech.cpp"
    if not source.exists():
        source.mkdir(parents=True)
        run("git", "init", str(source))
        run("git", "-C", str(source), "remote", "add", "origin", PINS["speech_repo"])
    if not (source / ".git").is_dir():
        raise RuntimeError(f"Expected an owned git checkout at {source}")
    actual = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if actual.returncode:
        run("git", "-C", str(source), "fetch", "--depth", "1", "origin", PINS["speech_revision"])
        run("git", "-C", str(source), "checkout", "--detach", PINS["speech_revision"])
    elif actual.stdout.strip() != PINS["speech_revision"]:
        raise RuntimeError(
            "speech.cpp is at a different revision. Preserve it elsewhere before reinstalling."
        )
    run("git", "-C", str(source), "submodule", "update", "--init", "--recursive", "--depth", "1")
    ggml = subprocess.check_output(
        ["git", "-C", str(source / "ggml"), "rev-parse", "HEAD"], text=True
    ).strip()
    if ggml != PINS["ggml_revision"]:
        raise RuntimeError("Unexpected ggml revision.")
    patch = STATE / "assets" / "drama3-speechcpp-fast.patch"
    already = subprocess.run(
        ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)], capture_output=True
    )
    if already.returncode:
        run("git", "-C", str(source), "apply", "--check", str(patch))
        run("git", "-C", str(source), "apply", str(patch))
    print("Pinned source and patch are ready.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-only", action="store_true", help="Skip model weights for build checks."
    )
    prepare(parser.parse_args().source_only)

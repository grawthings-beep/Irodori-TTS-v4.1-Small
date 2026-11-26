"""Stop only this checkout's selected Python UI on Linux, preserving all files."""

import argparse
import os
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def matches(cwd, args, service):
    if Path(cwd) != ROOT or not args or not Path(args[0]).name.startswith("python"):
        return False
    if service == "irodori":
        return len(args) > 1 and args[1] in ("voice_clone_app.py", str(ROOT / "voice_clone_app.py"))
    return len(args) > 2 and args[1:3] == ["-m", "drama.app"]


def stop(service):
    if not Path("/proc/self").exists():
        raise RuntimeError("Run this command in the Linux Pod terminal.")
    stopped = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            cwd = (item / "cwd").resolve(strict=True)
            args = (item / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if matches(cwd, args, service):
                os.kill(int(item.name), signal.SIGTERM)
                stopped.append(item)
        except (OSError, UnicodeError):
            continue
    deadline = time.monotonic() + 15
    while stopped and time.monotonic() < deadline:
        stopped = [item for item in stopped if item.exists()]
        if stopped:
            time.sleep(0.2)
    if stopped:
        raise RuntimeError("UI did not stop. Check its log before restarting.")
    print(f"{service}: stopped, or already stopped. Saved files were preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=["irodori", "drama"])
    stop(parser.parse_args().service)

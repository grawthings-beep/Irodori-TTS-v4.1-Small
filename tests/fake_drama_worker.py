"""Protocol fixture run in a real subprocess; never downloads or loads a model."""

import base64
import json
import sys
import time
from pathlib import Path


def emit(**event):
    print(json.dumps(event), flush=True)


emit(
    type="ready",
    protocol=4,
    version="0.11.2",
    model={"architecture": "drama3", "sample_rate": 44100},
)
for line in sys.stdin:
    request = json.loads(line)
    with Path(sys.argv[1]).open("a", encoding="utf-8") as log:
        log.write(json.dumps(request, ensure_ascii=False) + "\n")
    req_id = request["id"]
    if request["type"] == "add_voice":
        emit(type="end", id=req_id)
        continue
    text = request["text"]
    if text == "timeout":
        time.sleep(30)
    if text == "error":
        emit(type="error", id=req_id, error={"message": "fixture error"})
        continue
    if text == "eof":
        break
    pcm = b"\x00\x00\x01\x00" * 11025
    emit(type="progress", id=req_id, done=0.5)
    emit(
        type="chunk",
        id=req_id,
        seq=2 if text == "bad_sequence" else 0,
        pcm=base64.b64encode(pcm).decode("ascii"),
    )
    emit(
        type="end",
        id=req_id,
        samples=1 if text == "bad_samples" else 22050,
        seed=request.get("seed", 42),
        stop="complete",
    )

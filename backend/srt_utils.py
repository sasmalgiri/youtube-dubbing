from pathlib import Path
from typing import List, Dict


def _fmt_time(t: float) -> str:
    if t < 0:
        t = 0
    ms = int(round(t * 1000.0))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segments: List[Dict], out_path: Path, text_key: str = "text"):
    lines = []
    for i, seg in enumerate(sorted(segments, key=lambda s: s["start"]), start=1):
        start = _fmt_time(seg["start"])
        end = _fmt_time(seg["end"])
        text = seg.get(text_key, "").strip()
        if text:
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

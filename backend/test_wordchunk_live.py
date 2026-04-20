"""Live integration test — fetch English VTT → split sentences → Google Translate.

Skips TTS and video stretch (slow/expensive). Proves the new pipeline flow:
  English subs → sentences → Hindi translation
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dubbing.wordchunk import (
    _resolve_executable,
    _download_youtube_english_vtt,
    _parse_vtt_words,
    _split_into_sentences,
    _translate_sentences,
    _split_long_sentences,
)


TEST_URL = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
CHUNK_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def _progress(phase, pct, msg):
    print(f"  [progress] {phase} {pct*100:.0f}%  {msg}")


def main():
    ytdlp = _resolve_executable("yt-dlp")
    print(f"[resolver] yt-dlp -> {ytdlp}")

    tmp = Path(tempfile.mkdtemp())
    sub_dir = tmp / "yt_subs"

    print(f"\n[1/4] Fetching ENGLISH subs for: {TEST_URL}")
    vtt = _download_youtube_english_vtt(TEST_URL, sub_dir, "en", ytdlp)
    if not vtt:
        print("[FAIL] No English subs returned")
        return 1
    print(f"[ok] got: {vtt.name} ({vtt.stat().st_size} bytes)")

    print(f"\n[2/4] Parsing VTT words + timings...")
    words = _parse_vtt_words(vtt)
    if not words:
        print("[FAIL] No words parsed")
        return 1
    print(f"[ok] {len(words)} words")
    print(f"    first 8: {[(w, f'{s:.2f}') for w, s, _ in words[:8]]}")

    print(f"\n[3/4] Splitting into sentences...")
    sentences = _split_into_sentences(words)
    if not sentences:
        print("[FAIL] No sentences produced")
        return 1
    print(f"[ok] {len(sentences)} sentences")
    for i, s in enumerate(sentences[:3]):
        print(f"    sentence {i}: [{s['start']:.2f}-{s['end']:.2f}s]  \"{s['text'][:100]}\"")

    print(f"\n[4/4] Google Translate sentences → hi...")
    _translate_sentences(sentences, "hi", _progress, lambda: False, max_workers=10)
    for i, s in enumerate(sentences[:3]):
        print(f"    [{i}] en: \"{s['text'][:80]}\"")
        print(f"        hi: \"{s['text_translated'][:80]}\"")

    pieces = _split_long_sentences(sentences, max_words=CHUNK_SIZE)
    print(f"\n[summary] {len(sentences)} sentences → {len(pieces)} TTS pieces "
          f"(chunk cap = {CHUNK_SIZE} words)")
    print(f"[summary] sample pieces:")
    for i, p in enumerate(pieces[:5]):
        n_words = len(p["text"].split())
        print(f"    piece {i}: {n_words}w  [{p['start']:.2f}-{p['end']:.2f}s]  \"{p['text'][:80]}\"")

    print(f"\n[PASS] Full English-subs → sentence-split → Google Translate pipeline works")
    return 0


if __name__ == "__main__":
    sys.exit(main())

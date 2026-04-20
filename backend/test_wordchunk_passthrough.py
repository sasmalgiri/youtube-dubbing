"""Simulate the WordChunk VTT → chunk text → TTS input flow.

Verifies that YouTube's Hindi text arrives at Edge-TTS byte-identical to
what was in the VTT file — no translation engine, no modification, no
re-tokenisation.
"""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

from dubbing.wordchunk import _parse_vtt_words

# Sample YouTube auto-translate VTT (hi-en pseudo-lang) with inline word timings.
# This mimics the exact format YouTube returns for rolling two-line captions.
SAMPLE_VTT = """\
WEBVTT
Kind: captions
Language: hi

00:00:00.000 --> 00:00:03.000
<00:00:00.000><c>मेरे</c><00:00:00.500><c> पिता</c><00:00:01.000><c> राजा</c><00:00:01.500><c> थे</c><00:00:02.000><c> और</c>

00:00:02.500 --> 00:00:05.500
<00:00:02.000><c>और</c><00:00:02.500><c> मेरी</c><00:00:03.000><c> माँ</c><00:00:03.500><c> रानी</c><00:00:04.000><c> थीं</c>

00:00:05.000 --> 00:00:08.000
<00:00:04.500><c>वे</c><00:00:05.000><c> दोनों</c><00:00:05.500><c> सफेद</c><00:00:06.000><c> लोमड़ी</c><00:00:06.500><c> थे</c>

00:00:07.500 --> 00:00:10.000
<00:00:07.000><c>[music]</c><00:00:07.500><c> फिर</c><00:00:08.000><c> भी</c><00:00:08.500><c> जब</c>
"""

EXPECTED_WORDS_IN_ORDER = [
    "मेरे", "पिता", "राजा", "थे", "और",
    "मेरी", "माँ", "रानी", "थीं",
    "वे", "दोनों", "सफेद", "लोमड़ी", "थे",
    "फिर", "भी", "जब",
]


def simulate():
    tmp = Path(tempfile.mkdtemp())
    vtt = tmp / "sample.vtt"
    vtt.write_text(SAMPLE_VTT, encoding="utf-8")

    words = _parse_vtt_words(vtt)
    extracted = [w for w, _, _ in words]

    print("── Stage 1: VTT → word list ──")
    print(f"Extracted {len(words)} words (dedup should drop rolling repeats)")
    for w, s, e in words:
        print(f"  [{s:.2f}s–{e:.2f}s]  {w}")

    print("\n── Verification ──")
    if extracted == EXPECTED_WORDS_IN_ORDER:
        print("[PASS] Word order matches expected (duplicates from rolling cues removed)")
    else:
        print("[FAIL] Word order mismatch!")
        print(f"  Expected: {EXPECTED_WORDS_IN_ORDER}")
        print(f"  Got:      {extracted}")
        return False

    # ── Stage 2: Group into 4-word chunks (same logic as run_wordchunk) ──
    chunk_size = 4
    chunks = []
    for i in range(0, len(words), chunk_size):
        group = words[i:i + chunk_size]
        text = " ".join(w for w, _, _ in group).strip()
        if not text:
            continue
        chunks.append({
            "start": group[0][1],
            "end": group[-1][2],
            "text": text,
        })

    print(f"\n── Stage 2: {len(words)} words → {len(chunks)} chunks of {chunk_size} words ──")
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: [{c['start']:.2f}–{c['end']:.2f}s]  '{c['text']}'")

    # ── Stage 3: Verify the TEXT passed to Edge-TTS is pure Devanagari ──
    print("\n── Stage 3: What Edge-TTS receives ──")
    import re
    latin_leak = False
    for c in chunks:
        # This is the exact text _edge_tts_async() passes to edge_tts.Communicate()
        text = c["text"].strip()
        print(f"  TTS input: '{text}'")
        # Check for any Latin letters — would indicate English leaked through
        if re.search(r"[A-Za-z]", text):
            print(f"    [LEAK] Latin characters present!")
            latin_leak = True
        # Check for noise tags
        if re.search(r"\[(music|applause|laughter)\]", text, re.IGNORECASE):
            print(f"    [LEAK] Noise tag present!")
            latin_leak = True

    if latin_leak:
        print("\n[FAIL] Non-Hindi content leaked into TTS input")
        return False

    print("\n[PASS] All chunks contain only Devanagari + spaces — ready for Hindi TTS")

    # ── Stage 4: Byte-for-byte passthrough check ──
    print("\n── Stage 4: Byte-identical passthrough ──")
    original_hindi_words = set(EXPECTED_WORDS_IN_ORDER)
    tts_words = set()
    for c in chunks:
        for w in c["text"].split():
            tts_words.add(w)

    missing = original_hindi_words - tts_words
    extra = tts_words - original_hindi_words

    if missing or extra:
        print(f"[FAIL] Word set differs:")
        if missing: print(f"  missing from TTS input: {missing}")
        if extra:   print(f"  extra in TTS input:     {extra}")
        return False

    print("[PASS] Every VTT word reaches TTS byte-identical — no translation, no modification")
    return True


if __name__ == "__main__":
    ok = simulate()
    sys.exit(0 if ok else 1)

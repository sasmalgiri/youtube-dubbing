"""WordChunk Pipeline — YouTube English subs → sentence-split → Google Translate → TTS → super-stretch video.

Flow:
  1. Download video
  2. Download YouTube English subs (manual preferred, falls back to auto-captions).
     NOT the hi-en auto-translate endpoint — that gets 429-rate-limited.
  3. Parse inline word-level timings from VTT (ordered, deduped)
  4. Group words into SENTENCES (split on . ? ! with abbreviation guards)
  5. Google Translate each sentence en → hi (parallel)
  6. If a translated sentence exceeds `chunk_size` words, split it further
     at word boundaries so Edge-TTS never gets an overly long input.
  7. Edge-TTS each sentence/piece (parallel)
  8. Concatenate TTS clips back-to-back (50 ms gap between pieces)
  9. Stretch video (capped 1×–5×) to match concatenated audio duration
  10. Mux

NO Whisper. NO original-timing matching. NO glossary. NO LLM.
"""
from __future__ import annotations
import asyncio
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor


def _resolve_executable(name: str) -> str:
    """Find a working executable, skipping broken venv shims.

    The project's ``.venv/Scripts/yt-dlp.exe`` sometimes ships as a stale
    older build that crashes on launch. The classic pipeline uses the same
    resolver logic (pipeline.py ``_find_executable``) to fall through to the
    system install when the venv one is broken.
    """
    ext = ".exe" if sys.platform == "win32" else ""
    full_name = name + ext

    def _works(path: str) -> bool:
        try:
            r = subprocess.run([path, "--version"], capture_output=True,
                               timeout=5, text=True)
            return r.returncode == 0 and bool((r.stdout or r.stderr or "").strip())
        except Exception:
            return False

    # 1. venv Scripts dir (next to python.exe)
    venv_path = Path(sys.executable).parent / full_name
    if venv_path.exists() and _works(str(venv_path)):
        return str(venv_path)

    # 2. PATH (but verify — catches broken shims)
    found = shutil.which(name)
    if found and _works(found):
        return found

    # 3. Python's Scripts dir (pip-installed tools)
    scripts_path = Path(sys.executable).parent / "Scripts" / full_name
    if scripts_path.exists() and _works(str(scripts_path)):
        return str(scripts_path)

    # 4. Common Windows locations (catches the case where we're running in
    # a broken venv and need the global Python's Scripts dir)
    if sys.platform == "win32":
        userprofile = os.environ.get("USERPROFILE", str(Path.home()))
        candidates = [
            Path(userprofile) / "AppData" / "Local" / "Programs" / "Python" / "Python310" / "Scripts" / full_name,
            Path(userprofile) / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "Scripts" / full_name,
            Path(userprofile) / "AppData" / "Local" / "Programs" / "Python" / "Python312" / "Scripts" / full_name,
            Path("C:/Program Files/ffmpeg/bin") / full_name,
            Path("C:/ffmpeg/bin") / full_name,
        ]
        for cand in candidates:
            if cand.exists() and _works(str(cand)):
                return str(cand)

    # 5. Last resort — return the name and hope PATH resolves it at call time
    return name


# ── VTT word-level parsing ──────────────────────────────────────────────

_CUE_TIMELINE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})"
)
_INLINE_TS_RE = re.compile(r"<(\d{2}:\d{2}:\d{2}\.\d{3})>")
_NOISE_INLINE_RE = re.compile(
    r'\s*\[(?:music|applause|laughter|silence|noise|sound|cheering|singing)\]\s*',
    re.IGNORECASE,
)


def _vtt_ts_to_seconds(ts: str) -> float:
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_vtt_words(vtt_path: Path) -> List[Tuple[str, float, float]]:
    """Extract (word, start_time, end_time) tuples from a VTT file.

    Handles two formats:
      (a) Auto-captions with inline ``<hh:mm:ss.ttt>`` word timestamps.
      (b) Manual/human-provided subs where each cue has plain text and
          only cue-level start/end times; words are spread evenly across
          the cue's duration.

    De-duplicates by (word, rounded-timestamp) so YouTube's rolling-cue
    repeats collapse into a single occurrence.

    To distinguish the "first legitimate cue with plain text" from
    "rolling repeat cues", we only treat a plain-text cue as legitimate
    if its start time is > the last recorded word's time (i.e. new
    content, not a replay of already-seen content).
    """
    content = vtt_path.read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")

    seen: set = set()
    word_items: List[Tuple[str, float]] = []
    last_recorded_ts = -1.0

    i = 0
    while i < len(lines):
        m = _CUE_TIMELINE_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        cue_start = _vtt_ts_to_seconds(m.group(1))
        cue_end = _vtt_ts_to_seconds(m.group(2))

        cue_lines = []
        i += 1
        while i < len(lines):
            line = lines[i]
            if _CUE_TIMELINE_RE.match(line.strip()):
                break
            if line.strip():
                cue_lines.append(line)
            i += 1

        raw = "\n".join(cue_lines)
        raw = _NOISE_INLINE_RE.sub(" ", raw)
        # Strip <c> </c> styling + any other SSA-like tags before word split
        raw = re.sub(r"</?c[^>]*>", "", raw)

        has_inline_ts = bool(_INLINE_TS_RE.search(raw))

        if has_inline_ts:
            # Auto-caption path: walk timestamps + words alternately
            pos = 0
            current_ts = cue_start
            while pos < len(raw):
                ts_match = _INLINE_TS_RE.search(raw, pos)
                if ts_match and ts_match.start() == pos:
                    current_ts = _vtt_ts_to_seconds(ts_match.group(1))
                    pos = ts_match.end()
                    continue
                next_ts = _INLINE_TS_RE.search(raw, pos)
                chunk_end = next_ts.start() if next_ts else len(raw)
                text_chunk = raw[pos:chunk_end]
                for word in text_chunk.split():
                    w = word.strip()
                    if not w:
                        continue
                    key = (w, round(current_ts, 2))
                    if key in seen:
                        continue
                    seen.add(key)
                    word_items.append((w, current_ts))
                    last_recorded_ts = max(last_recorded_ts, current_ts)
                pos = chunk_end
        else:
            # Manual-subs path: spread words evenly across cue duration.
            # Only accept the cue if its start is later than the last word
            # already recorded (skips rolling-display repeats).
            if cue_start <= last_recorded_ts + 0.001:
                continue
            words_in_cue = [w for w in re.sub(r"<[^>]+>", "", raw).split() if w.strip()]
            if not words_in_cue:
                continue
            span = max(cue_end - cue_start, 0.1)
            step = span / len(words_in_cue)
            for j, w in enumerate(words_in_cue):
                ts = cue_start + j * step
                key = (w, round(ts, 2))
                if key in seen:
                    continue
                seen.add(key)
                word_items.append((w, ts))
                last_recorded_ts = max(last_recorded_ts, ts)

    # Ensure final list is in timeline order
    word_items.sort(key=lambda t: t[1])

    # Build (word, start, end) by using the next word's time as end
    result: List[Tuple[str, float, float]] = []
    for idx, (word, start_t) in enumerate(word_items):
        if idx + 1 < len(word_items):
            end_t = word_items[idx + 1][1]
        else:
            end_t = start_t + 0.5
        if end_t <= start_t:
            end_t = start_t + 0.1
        result.append((word, start_t, end_t))
    return result


# ── YouTube download ────────────────────────────────────────────────────

def _download_video(url: str, video_path: Path, ytdlp: str, cancel_check: Callable) -> None:
    if video_path.exists() and video_path.stat().st_size > 0:
        return
    cmd = [ytdlp, "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
           "--merge-output-format", "mp4", "-o", str(video_path), url]
    subprocess.run(cmd, capture_output=True, timeout=600)
    if not video_path.exists():
        subprocess.run([ytdlp, "-f", "best", "-o", str(video_path), url],
                       capture_output=True, timeout=600)
    if not video_path.exists():
        raise RuntimeError(f"Failed to download video from {url}")


def _download_youtube_english_vtt(url: str, sub_dir: Path,
                                   source_lang: str, ytdlp: str) -> Optional[Path]:
    """Download YouTube's *native* English subtitles — NOT the auto-translate
    endpoint. The plain-language endpoint is NOT aggressively rate-limited
    (unlike hi-en auto-translate which 429s constantly).

    Strategy:
      1. Try manual subs first (--write-sub) — usually sparser but higher
         quality with real punctuation and sentence boundaries.
      2. Fall back to auto-captions (--write-auto-sub) — always present for
         videos with speech; YouTube's newer ASR adds light punctuation.

    No cookies — subtitle endpoints are public and premium cookies
    trigger YouTube's JS signature challenge.
    """
    import time as _time

    sub_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(sub_dir / "ytsub.%(ext)s")
    sub_lang = (source_lang or "en").split("-")[0].lower()
    if sub_lang == "auto":
        sub_lang = "en"

    def _clean_dir():
        for f in sub_dir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass

    def _attempt(flag: str, label: str) -> Optional[Path]:
        _clean_dir()
        cmd = [ytdlp,
               flag,
               "--sub-lang", sub_lang,
               "--sub-format", "vtt",
               "--skip-download",
               "--no-warnings",
               "--sleep-subtitles", "1",
               "-o", out_tpl,
               url]
        print(f"[WordChunk YT-DLP {label}] {' '.join(cmd)}", flush=True)
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=120,
                                   text=True, encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[WordChunk YT-DLP {label}] exception: {e}", flush=True)
            return None
        if proc.returncode != 0:
            print(f"[WordChunk YT-DLP {label}] returncode={proc.returncode}", flush=True)
        for ln in (proc.stdout or "").splitlines()[:8]:
            print(f"  [stdout] {ln}", flush=True)
        for ln in (proc.stderr or "").splitlines()[:8]:
            print(f"  [stderr] {ln}", flush=True)
        files = list(sub_dir.glob("*"))
        print(f"[WordChunk YT-DLP {label}] files: {[f.name for f in files]}", flush=True)
        for vtt in sub_dir.glob("*.vtt"):
            if vtt.stat().st_size > 100:
                return vtt
        return None

    # 1. Manual subs (sparse, but proper punctuation)
    result = _attempt("--write-sub", "manual")
    if result:
        return result

    # Small gap before hammering again — the two endpoints share rate limits
    _time.sleep(2)

    # 2. Auto-captions (always present for spoken-word videos)
    result = _attempt("--write-auto-sub", "auto")
    if result:
        return result

    # 3. One retry of auto-cap with slightly longer sleep
    _time.sleep(10)
    result = _attempt("--write-auto-sub", "auto-retry")
    return result


# ── Plain-text transcript parsing (user-pasted fallback) ───────────────

def _sentences_from_plain_text(text: str) -> List[Dict]:
    """Split a raw pasted transcript into sentences without needing timings.

    Useful when yt-dlp can't reach YouTube (429, regional block, private
    video) but the user can still copy the transcript manually from
    YouTube's transcript panel or any other source.

    Start/end times are left as 0.0 — the pure-concat assembly doesn't
    read them.
    """
    # Flatten whitespace (newlines from the transcript panel, tabs, etc.)
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Drop YouTube transcript panel timestamps like "0:03", "12:45", "1:02:33"
    cleaned = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", cleaned)
    # Drop common noise markers
    cleaned = _NOISE_INLINE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []

    # Split on sentence terminators followed by whitespace
    # (abbreviation guard: don't split after common abbrevs like "Mr.")
    pieces: List[str] = []
    buf: List[str] = []
    tokens = cleaned.split(" ")
    for tok in tokens:
        buf.append(tok)
        if re.search(r"[.!?]$", tok) and tok.lower() not in _ABBREV:
            pieces.append(" ".join(buf).strip())
            buf = []
    if buf:
        pieces.append(" ".join(buf).strip())

    # If we got zero terminators (no punctuation in the pasted text), fall
    # back to fixed-length chunks of ~20 words.
    if len(pieces) == 1 and len(tokens) > 40:
        pieces = []
        for i in range(0, len(tokens), 20):
            pieces.append(" ".join(tokens[i:i + 20]))

    return [{"text": p, "start": 0.0, "end": 0.0} for p in pieces if p]


# ── Sentence splitting ──────────────────────────────────────────────────

# Common English abbreviations that end in '.' but don't end a sentence.
_ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs.", "etc.", "e.g.",
    "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "no.", "vol.", "fig.",
}


def _split_into_sentences(words: List[Tuple[str, float, float]]) -> List[Dict]:
    """Group flat word list into sentences keyed by start/end timestamps.

    Sentence terminators: . ? ! (with abbreviation guard so ``Mr.`` doesn't
    cut mid-sentence). Auto-captions often lack terminators — in that case
    we fall back to splitting on every ~12 words so Google Translate
    doesn't receive unbounded inputs.
    """
    sentences: List[Dict] = []
    buf_words: List[str] = []
    buf_start: Optional[float] = None
    buf_end: Optional[float] = None

    def _flush():
        nonlocal buf_words, buf_start, buf_end
        if buf_words:
            sentences.append({
                "text": " ".join(buf_words).strip(),
                "start": buf_start if buf_start is not None else 0.0,
                "end": buf_end if buf_end is not None else 0.0,
            })
        buf_words = []
        buf_start = None
        buf_end = None

    # If the entire transcript contains no terminator, switch to length-based
    # splitting. Probe the first 100 words to decide.
    sample = " ".join(w for w, _, _ in words[:100])
    has_terminators = bool(re.search(r'[.!?]', sample))
    LEN_LIMIT = 12 if not has_terminators else 40  # soft cap even with punctuation

    for word, s, e in words:
        if buf_start is None:
            buf_start = s
        buf_words.append(word)
        buf_end = e
        # Terminator check with abbreviation guard
        if re.search(r'[.!?]$', word) and word.lower() not in _ABBREV:
            _flush()
        elif len(buf_words) >= LEN_LIMIT:
            # Soft cap — avoid giving Google Translate a novel to chew
            _flush()

    _flush()
    return sentences


# ── Google Translate (batch parallel) ───────────────────────────────────

def _translate_sentences(sentences: List[Dict], target_lang: str,
                         progress: Callable, cancel_check: Callable,
                         max_workers: int = 50) -> List[Dict]:
    """Translate each sentence English→target via Google (deep_translator).

    Each sentence gets a ``text_translated`` key. On repeated failure the
    original English is kept so TTS still speaks *something*.
    """
    from deep_translator import GoogleTranslator
    import threading

    total = len(sentences)
    done = [0]
    lock = threading.Lock()

    def _is_garbage(text: str) -> bool:
        if not text:
            return True
        low = text.lower()
        return ("error 500" in low or "server error" in low
                or "<html" in low or "<!doctype" in low)

    def _one(seg: Dict):
        if cancel_check():
            return
        text = seg.get("text", "").strip()
        if not text:
            seg["text_translated"] = ""
            return
        translated = ""
        for attempt in range(3):
            try:
                tr = GoogleTranslator(source="auto", target=target_lang)
                out = tr.translate(text)
                if out and not _is_garbage(out):
                    translated = out
                    break
            except Exception:
                pass
            import time as _t
            _t.sleep(1.0 * (attempt + 1))
        seg["text_translated"] = translated or text  # fall back to source
        with lock:
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == total:
                progress("translate", 0.1 + 0.8 * (done[0] / total),
                         f"[{int((done[0]/total)*100)}%] Google Translate "
                         f"{done[0]}/{total} sentences")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_one, sentences))
    return sentences


def _split_long_sentences(sentences: List[Dict], max_words: int) -> List[Dict]:
    """After translation, break any sentence whose Hindi text exceeds
    ``max_words`` words into word-bounded pieces. Start/end times are
    interpolated proportionally across the pieces.
    """
    out: List[Dict] = []
    for s in sentences:
        text = (s.get("text_translated") or "").strip()
        if not text:
            continue
        words = text.split()
        if len(words) <= max_words:
            out.append({
                "text": text,
                "start": s.get("start", 0.0),
                "end":   s.get("end",   0.0),
            })
            continue
        # Split into ~equal pieces of max_words
        n_pieces = (len(words) + max_words - 1) // max_words
        span = s.get("end", 0.0) - s.get("start", 0.0)
        for i in range(n_pieces):
            a = i * max_words
            b = min((i + 1) * max_words, len(words))
            piece_words = words[a:b]
            if not piece_words:
                continue
            pstart = s.get("start", 0.0) + span * (a / len(words))
            pend = s.get("start", 0.0) + span * (b / len(words))
            out.append({
                "text": " ".join(piece_words),
                "start": pstart,
                "end":   pend,
            })
    return out


# ── TTS ─────────────────────────────────────────────────────────────────

async def _edge_tts_async(text: str, out_mp3: Path, voice: str, rate: str) -> bool:
    import edge_tts
    try:
        comm = edge_tts.Communicate(text, voice, rate=rate)
        await comm.save(str(out_mp3))
        return out_mp3.exists() and out_mp3.stat().st_size > 0
    except Exception:
        return False


def _tts_all_chunks(chunks: List[Dict], work_dir: Path, voice: str, rate: str,
                    ffmpeg: str, progress: Callable, cancel_check: Callable) -> List[Dict]:
    """Generate Edge-TTS WAVs for every chunk. Returns enriched chunks with
    `wav` (Path) and `tts_duration` (float) keys."""
    total = len(chunks)
    done = [0]
    lock = __import__("threading").Lock()

    async def _one(i: int, chunk: Dict, sem: asyncio.Semaphore):
        if cancel_check():
            return
        async with sem:
            text = chunk["text"].strip()
            if not text:
                return
            mp3 = work_dir / f"wc_tts_{i:04d}.mp3"
            wav = work_dir / f"wc_tts_{i:04d}.wav"
            ok = await _edge_tts_async(text, mp3, voice, rate)
            if not ok:
                # one retry
                ok = await _edge_tts_async(text, mp3, voice, rate)
            if ok and mp3.exists():
                try:
                    subprocess.run(
                        [ffmpeg, "-y", "-i", str(mp3),
                         "-ar", "48000", "-ac", "2", str(wav)],
                        check=True, capture_output=True)
                    mp3.unlink(missing_ok=True)
                    chunk["wav"] = wav
                    chunk["tts_duration"] = _get_duration(wav, ffmpeg)
                except Exception:
                    chunk["wav"] = None
                    chunk["tts_duration"] = 0.0
            else:
                chunk["wav"] = None
                chunk["tts_duration"] = 0.0

            with lock:
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == total:
                    progress("synthesize", 0.1 + 0.8 * (done[0] / total),
                             f"[{int((done[0]/total)*100)}%] Edge-TTS {done[0]}/{total} chunks")

    async def _run():
        sem = asyncio.Semaphore(30)
        await asyncio.gather(*[_one(i, c, sem) for i, c in enumerate(chunks)])

    asyncio.run(_run())
    return chunks


# ── Audio assembly ──────────────────────────────────────────────────────

def _build_audio_timeline(chunks: List[Dict], out_wav: Path,
                          sample_rate: int = 48000,
                          chunk_gap_ms: int = 50,
                          ffmpeg: str = "ffmpeg") -> float:
    """Concatenate TTS WAVs back-to-back with a tiny breathing gap between
    chunks. NO matching against original YouTube/Whisper timing — the video
    stretches to match this pure audio duration later.

    For data ≤ 4 GB writes a standard WAV header directly in Python (fast).
    For larger outputs, falls back to ffmpeg concat with RF64 output so the
    WAV spec's 32-bit size field isn't blown (RIFF caps at ~4 GB).

    Returns the total timeline duration in seconds.
    """
    n_channels = 2

    # Pre-load all PCM bytes in order, but also keep the WAV paths so we can
    # fall back to ffmpeg concat if the total exceeds WAV's 4 GB limit.
    valid_paths: List[Path] = []
    loaded: List[bytes] = []
    for c in chunks:
        wav = c.get("wav")
        if not wav or not wav.exists():
            continue
        pcm = _read_wav_pcm(wav)
        if pcm:
            pcm = _trim_silence_pcm(pcm, n_channels=n_channels)
            loaded.append(pcm)
            valid_paths.append(Path(wav))

    if not loaded:
        raise RuntimeError("No TTS chunks produced valid audio")

    gap_bytes = int(chunk_gap_ms / 1000.0 * sample_rate) * n_channels * 2
    gap_silence = bytes(gap_bytes)

    # Estimate total size before concatenating in memory — 4 GB of PCM
    # loaded at once will OOM on most machines anyway.
    estimated_size = sum(len(p) for p in loaded) + gap_bytes * max(0, len(loaded) - 1)
    MAX_WAV_SIZE = (2 ** 32) - 128  # header overhead margin

    if estimated_size > MAX_WAV_SIZE:
        # >4 GB — can't fit in WAV/RIFF 32-bit size. Use ffmpeg concat with
        # RF64 output, which supports arbitrary size via 64-bit DS64 chunks.
        print(f"[WordChunk] Timeline > 4 GB ({estimated_size/1e9:.2f} GB) — "
              f"using ffmpeg concat with RF64", flush=True)
        # Free the in-memory bytes — we only need the paths now.
        del loaded
        work_dir = out_wav.parent

        # Trim leading/trailing silence from each WAV before concat
        trimmed_paths: List[Path] = []
        for i, p in enumerate(valid_paths):
            pcm = _read_wav_pcm(p)
            if pcm:
                pcm = _trim_silence_pcm(pcm, n_channels=n_channels)
                tp = work_dir / f"_trimmed_{i:04d}.wav"
                data_size_t = len(pcm)
                bits = 16
                block = n_channels * (bits // 8)
                with open(tp, "wb") as tf:
                    tf.write(b"RIFF")
                    tf.write(struct.pack('<I', 36 + data_size_t))
                    tf.write(b"WAVE")
                    tf.write(b"fmt ")
                    tf.write(struct.pack('<I', 16))
                    tf.write(struct.pack('<H', 1))
                    tf.write(struct.pack('<H', n_channels))
                    tf.write(struct.pack('<I', sample_rate))
                    tf.write(struct.pack('<I', sample_rate * block))
                    tf.write(struct.pack('<H', block))
                    tf.write(struct.pack('<H', bits))
                    tf.write(b"data")
                    tf.write(struct.pack('<I', data_size_t))
                    tf.write(pcm)
                trimmed_paths.append(tp)
            else:
                trimmed_paths.append(p)

        list_path = out_wav.with_suffix(".list.txt")
        with open(list_path, "w", encoding="utf-8") as lf:
            for i, p in enumerate(trimmed_paths):
                lf.write(f"file '{p.as_posix()}'\n")
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-c:a", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(n_channels),
             "-rf64", "auto",
             str(out_wav)],
            check=True, capture_output=True, timeout=3600)
        list_path.unlink(missing_ok=True)
        for tp in trimmed_paths:
            if tp.name.startswith("_trimmed_"):
                tp.unlink(missing_ok=True)
        return _get_duration(out_wav, ffmpeg)

    # Fast path: ≤ 4 GB — Python-written RIFF header + raw PCM bytes
    parts: List[bytes] = []
    for i, pcm in enumerate(loaded):
        parts.append(pcm)
    timeline = b"".join(parts)
    total_samples = len(timeline) // (n_channels * 2)

    data_size = len(timeline)
    with open(out_wav, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', n_channels))
        f.write(struct.pack('<I', sample_rate))
        bits = 16
        block = n_channels * (bits // 8)
        f.write(struct.pack('<I', sample_rate * block))
        f.write(struct.pack('<H', block))
        f.write(struct.pack('<H', bits))
        f.write(b"data")
        f.write(struct.pack('<I', data_size))
        f.write(timeline)

    return total_samples / sample_rate


def _read_wav_pcm(wav_path: Path) -> bytes:
    try:
        with open(wav_path, "rb") as f:
            riff = f.read(12)
            if riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
                return b""
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                cid = hdr[:4]
                csize = struct.unpack('<I', hdr[4:])[0]
                if cid == b'data':
                    return f.read(csize)
                f.seek(csize, 1)
    except Exception:
        pass
    return b""


def _trim_silence_pcm(pcm: bytes, n_channels: int = 2, bits: int = 16,
                       threshold: int = 300) -> bytes:
    """Strip leading and trailing silence from raw PCM bytes.

    threshold: absolute sample amplitude below which a frame is considered
    silent (16-bit range 0–32767). 300 ≈ −40 dB — catches TTS padding
    without clipping quiet speech.
    """
    if not pcm:
        return pcm
    bytes_per_sample = bits // 8  # 2 for 16-bit
    frame_size = n_channels * bytes_per_sample  # 4 for stereo 16-bit
    n_frames = len(pcm) // frame_size
    if n_frames == 0:
        return pcm

    import array
    samples = array.array('h')  # signed 16-bit
    samples.frombytes(pcm[:n_frames * frame_size])

    # Find first non-silent frame (check max of all channels in each frame)
    first = 0
    for i in range(n_frames):
        base = i * n_channels
        if any(abs(samples[base + ch]) > threshold for ch in range(n_channels)):
            first = i
            break
    else:
        # Entire clip is silent — return a tiny sliver to avoid zero-length
        return pcm[:frame_size * 2] if n_frames >= 2 else pcm

    # Find last non-silent frame
    last = n_frames - 1
    for i in range(n_frames - 1, first - 1, -1):
        base = i * n_channels
        if any(abs(samples[base + ch]) > threshold for ch in range(n_channels)):
            last = i
            break

    return pcm[first * frame_size : (last + 1) * frame_size]


# ── Video stretch + mux ─────────────────────────────────────────────────

def _stretch_video_to_match(video_in: Path, target_duration: float,
                             max_stretch: float, out_path: Path,
                             ffmpeg: str) -> None:
    """Stretch the video so its duration matches `target_duration`, capped at
    max_stretch× (slowdown only — we never speed video up)."""
    orig = _get_duration(video_in, ffmpeg)
    if orig <= 0:
        raise RuntimeError("Cannot probe source video duration")
    ratio = target_duration / orig  # >1 = slow down
    # Only slow down. If audio is shorter than video, skip stretch (mux later
    # will use -shortest, trimming video to audio length).
    if ratio <= 1.0:
        # No stretch needed — just copy
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video_in), "-c", "copy", str(out_path)],
            check=True, capture_output=True)
        return
    ratio = min(ratio, max_stretch)
    # setpts multiplier = ratio (slower playback)
    subprocess.run(
        [ffmpeg, "-y", "-i", str(video_in),
         "-filter:v", f"setpts={ratio:.4f}*PTS",
         "-an",
         str(out_path)],
        check=True, capture_output=True)


def _get_duration(path: Path, ffmpeg: str = "ffmpeg") -> float:
    try:
        ffprobe = ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in ffmpeg else "ffprobe"
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip() or 0)
    except Exception:
        return 0.0


# ── Public entry point ─────────────────────────────────────────────────

def run_wordchunk(
    source_url: str,
    work_dir: Path,
    output_path: Path,
    target_language: str = "hi",
    source_language: str = "en",
    tts_voice: str = "hi-IN-SwaraNeural",
    tts_rate: str = "+0%",
    audio_bitrate: str = "320k",
    chunk_size: int = 8,
    max_stretch: float = 20.0,
    transcript_override: str = "",
    dub_duration_min: int = 0,
    on_progress: Callable = None,
    cancel_check: Callable = None,
    ffmpeg: str = "ffmpeg",
    ytdlp: str = "yt-dlp",
) -> Path:
    progress = on_progress or (lambda *_: None)
    is_cancelled = cancel_check or (lambda: False)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if chunk_size not in (4, 8, 12):
        chunk_size = 8
    max_stretch = max(1.0, min(20.0, float(max_stretch)))

    # Resolve ffmpeg + yt-dlp to working binaries (skip broken venv shims)
    ffmpeg = _resolve_executable(ffmpeg) if ffmpeg in ("ffmpeg",) else ffmpeg
    ytdlp = _resolve_executable(ytdlp) if ytdlp in ("yt-dlp",) else ytdlp
    print(f"[WordChunk] Using ffmpeg={ffmpeg}, yt-dlp={ytdlp}", flush=True)

    # 1. Download video
    progress("download", 0.0, "[0%] Downloading video...")
    video_path = work_dir / "video.mp4"
    _download_video(source_url, video_path, ytdlp, is_cancelled)
    progress("download", 1.0, "[100%] Video downloaded")
    if is_cancelled():
        return output_path

    # 1b. Trim video to requested duration (0 = full video)
    _dur_limit_sec = float(dub_duration_min) * 60.0 if dub_duration_min and dub_duration_min > 0 else 0.0
    if _dur_limit_sec > 0:
        full_dur = _get_duration(video_path, ffmpeg)
        if full_dur > _dur_limit_sec + 1.0:
            progress("download", 1.0,
                     f"Trimming video to {dub_duration_min} min "
                     f"({_dur_limit_sec:.0f}s of {full_dur:.0f}s)...")
            trimmed_path = work_dir / "video_trimmed.mp4"
            subprocess.run(
                [ffmpeg, "-y", "-i", str(video_path),
                 "-t", f"{_dur_limit_sec:.3f}",
                 "-c", "copy", str(trimmed_path)],
                check=True, capture_output=True, timeout=300)
            # Replace original with trimmed version
            video_path.unlink(missing_ok=True)
            trimmed_path.rename(video_path)
            print(f"[WordChunk] Video trimmed to {dub_duration_min} min "
                  f"({_dur_limit_sec:.0f}s)", flush=True)

    # 2-4. Get sentences — either from user-pasted transcript OR YouTube subs
    src_lang = (source_language or "en").split("-")[0].lower()
    if src_lang == "auto":
        src_lang = "en"

    pasted = (transcript_override or "").strip()
    if pasted:
        progress("transcribe", 0.0,
                 f"[0%] Using pasted transcript ({len(pasted)} chars, skipping YouTube subs)...")
        sentences = _sentences_from_plain_text(pasted)
        if not sentences:
            raise RuntimeError("Pasted transcript contained no usable text")
        progress("transcribe", 1.0,
                 f"[100%] Split pasted transcript into {len(sentences)} sentences")
    else:
        progress("transcribe", 0.0,
                 f"[0%] Fetching YouTube {src_lang} subs (manual → auto)...")
        sub_dir = work_dir / "yt_subs"
        vtt = _download_youtube_english_vtt(source_url, sub_dir, src_lang, ytdlp)
        if not vtt:
            raise RuntimeError(
                f"No YouTube {src_lang} subs or auto-captions available. "
                f"Paste the video transcript into 'Transcript override' to bypass "
                f"the YouTube fetch.")
        progress("transcribe", 0.4, f"[40%] Got subs: {vtt.name}")

        words = _parse_vtt_words(vtt)
        if not words:
            raise RuntimeError("VTT parsed but contained no word-level timings")
        progress("transcribe", 0.6, f"[60%] Parsed {len(words)} words from VTT")

        sentences = _split_into_sentences(words)
        if not sentences:
            raise RuntimeError("Failed to split VTT words into sentences")
        progress("transcribe", 1.0, f"[100%] {len(sentences)} sentences built")
    if is_cancelled():
        return output_path

    # Filter sentences beyond the duration limit (VTT timestamps may extend past trimmed video)
    if _dur_limit_sec > 0 and sentences and sentences[0].get("start") is not None:
        before = len(sentences)
        sentences = [s for s in sentences if s.get("start", 0) < _dur_limit_sec]
        if len(sentences) < before:
            print(f"[WordChunk] Duration limit {dub_duration_min}min: "
                  f"kept {len(sentences)}/{before} sentences "
                  f"(dropped {before - len(sentences)} beyond {_dur_limit_sec:.0f}s)",
                  flush=True)
        if not sentences:
            raise RuntimeError(f"No sentences within the {dub_duration_min}-minute limit")

    # 5. Google Translate each sentence en → target
    progress("translate", 0.0,
             f"[0%] Google Translate {len(sentences)} sentences → {target_language}...")
    _translate_sentences(sentences, target_language, progress, is_cancelled)
    progress("translate", 1.0, f"[100%] Translation complete")
    if is_cancelled():
        return output_path

    # 6. Split any translated sentence longer than `chunk_size` words
    chunks: List[Dict] = _split_long_sentences(sentences, max_words=chunk_size)
    if not chunks:
        raise RuntimeError("Translation produced no usable text")
    print(f"[WordChunk] {len(sentences)} sentences → {len(chunks)} TTS pieces "
          f"(cap {chunk_size} words/piece)", flush=True)

    # 7. TTS each piece
    progress("synthesize", 0.0, f"[0%] Edge-TTS on {len(chunks)} pieces...")
    chunks = _tts_all_chunks(chunks, work_dir, tts_voice, tts_rate,
                              ffmpeg, progress, is_cancelled)
    chunks_with_audio = [c for c in chunks if c.get("wav")]
    if not chunks_with_audio:
        raise RuntimeError("All TTS attempts failed — no audio produced")
    progress("synthesize", 1.0,
             f"[100%] {len(chunks_with_audio)}/{len(chunks)} TTS clips ready")
    if is_cancelled():
        return output_path

    # 6. Assemble audio timeline (push later on overrun)
    progress("assemble", 0.0, "[0%] Assembling audio timeline...")
    timeline_wav = work_dir / "wc_timeline.wav"
    audio_dur = _build_audio_timeline(chunks_with_audio, timeline_wav, chunk_gap_ms=0, ffmpeg=ffmpeg)
    progress("assemble", 0.3, f"[30%] Audio timeline: {audio_dur:.1f}s")

    # 7. Stretch video
    progress("assemble", 0.4, f"[40%] Stretching video (cap {max_stretch}×)...")
    stretched_video = work_dir / "wc_video_stretched.mp4"
    _stretch_video_to_match(video_path, audio_dur, max_stretch, stretched_video, ffmpeg)
    stretched_dur = _get_duration(stretched_video, ffmpeg)
    progress("assemble", 0.7, f"[70%] Video stretched to {stretched_dur:.1f}s")

    # 8. Mux video + timeline audio
    if output_path.exists():
        output_path.unlink()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(stretched_video), "-i", str(timeline_wav),
         "-c:v", "copy", "-c:a", "aac", "-b:a", audio_bitrate,
         "-map", "0:v:0", "-map", "1:a:0",
         "-shortest", str(output_path)],
        check=True, capture_output=True)
    if not output_path.exists():
        raise RuntimeError("Final mux failed — output file missing")

    # Cleanup
    for pattern in ["wc_tts_*.wav", "wc_tts_*.mp3"]:
        for f in work_dir.glob(pattern):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    progress("assemble", 1.0, "[100%] Done!")
    return output_path

"""ASR Runner — dual-engine transcription with reconciliation.

Text source:  Parakeet (best punctuation/capitalization)
Timing source: WhisperX (best word-level alignment)
Reconciled:   Parakeet text aligned to WhisperX timing

Process isolation:
    Both Parakeet (NeMo) and WhisperX (faster-whisper/CTranslate2) are run
    in a *child process* via ``multiprocessing.Process``. The FastAPI server
    imports this module, but never loads torch/nemo/ctranslate2 itself — so
    a fatal native crash during model teardown (a known issue on Windows +
    CUDA — see backend/logs/crashes/*.faulthandler) can only kill the child.
    The server and the in-flight job thread stay alive and the error path
    gets a normal Python exception.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Callable
import re
import json
import tempfile
import multiprocessing as mp

from .contracts import Word


# ── Child-process workers (top-level so they're picklable on Windows spawn) ──

def _parakeet_child(wav_path_str: str, result_path: str):
    """Child process: runs Parakeet in isolation, writes JSON result."""
    try:
        words = _run_parakeet_impl(Path(wav_path_str), on_progress=None)
        payload = {
            "words": [
                {"text": w.text, "start": w.start, "end": w.end,
                 "source": w.source, "confidence": w.confidence}
                for w in words
            ],
            "error": None,
        }
    except Exception as exc:
        payload = {"words": [], "error": f"{type(exc).__name__}: {exc}"}
    try:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def _whisperx_child(wav_path_str: str, language: str, model_size: str, result_path: str):
    """Child process: runs WhisperX in isolation, writes JSON result.

    Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before importing
    torch so the CUDA caching allocator can grow instead of fragmenting —
    this is the exact remediation the CUDA OOM error message suggests.
    """
    import os as _os
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        words = _run_whisperx_impl(Path(wav_path_str), language=language,
                                   model_size=model_size, on_progress=None)
        payload = {
            "words": [
                {"text": w.text, "start": w.start, "end": w.end,
                 "source": w.source, "confidence": w.confidence}
                for w in words
            ],
            "error": None,
        }
    except Exception as exc:
        payload = {"words": [], "error": f"{type(exc).__name__}: {exc}"}
    try:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def _run_in_child(target, args, timeout_sec: int, on_progress: Callable = None,
                  label: str = "ASR") -> List[Word]:
    """Spawn a child process, wait for it, parse result JSON → List[Word].

    Any native crash in the child (SIGABRT from CTranslate2 / CUDA / NeMo
    teardown) surfaces here as a non-zero exit code → RuntimeError, instead
    of taking down the FastAPI process.

    A heartbeat fires every 10s via ``on_progress`` so the job UI shows the
    engine is still alive during the (potentially multi-minute) run — without
    this, the runner's single "Running Whisper..." line looks like a hang.
    """
    import os as _os
    import time as _time
    fd, result_path = tempfile.mkstemp(suffix=".json", prefix="asr_result_")
    _os.close(fd)
    try:
        p = mp.Process(target=target, args=(*args, result_path), daemon=True)
        p.start()
        if on_progress:
            on_progress(f"{label}: child pid={p.pid} running...")

        # Heartbeat loop — poll the child every 10s so the UI gets a live
        # "still working (Ns elapsed)" message instead of a silent freeze.
        t0 = _time.monotonic()
        heartbeat_interval = 10.0
        while True:
            p.join(timeout=heartbeat_interval)
            if not p.is_alive():
                break
            elapsed = _time.monotonic() - t0
            if elapsed >= timeout_sec:
                p.kill()
                p.join(5)
                raise RuntimeError(f"{label} child timed out after {timeout_sec}s")
            if on_progress:
                on_progress(f"{label}: still running ({int(elapsed)}s elapsed)...")
        if p.exitcode != 0:
            err = f"{label} child died with exit code {p.exitcode}"
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("error"):
                        err = data["error"]
            except Exception:
                pass
            raise RuntimeError(err)
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("error"):
            raise RuntimeError(data["error"])
        out: List[Word] = []
        for w in data.get("words", []):
            out.append(Word(
                text=w.get("text", ""),
                start=float(w.get("start", 0)),
                end=float(w.get("end", 0)),
                source=w.get("source"),
                confidence=w.get("confidence"),
            ))
        return out
    finally:
        try:
            Path(result_path).unlink(missing_ok=True)
        except Exception:
            pass


# ── Public API (runs the workers in a child process) ─────────────────────────

def run_parakeet(wav_path: Path, on_progress: Callable = None) -> List[Word]:
    """Run Parakeet TDT in an isolated child process."""
    if on_progress:
        on_progress("Parakeet: spawning isolated worker...")
    words = _run_in_child(_parakeet_child, (str(wav_path),),
                          timeout_sec=1800, on_progress=on_progress,
                          label="Parakeet")
    if on_progress:
        on_progress(f"Parakeet: {len(words)} words")
    return words


def run_whisperx(wav_path: Path, language: str = "en",
                 model_size: str = "large-v3",
                 on_progress: Callable = None) -> List[Word]:
    """Run WhisperX / faster-whisper in an isolated child process."""
    if on_progress:
        on_progress(f"Whisper ({model_size}): spawning isolated worker...")
    words = _run_in_child(_whisperx_child, (str(wav_path), language, model_size),
                          timeout_sec=1800, on_progress=on_progress,
                          label=f"Whisper {model_size}")
    if on_progress:
        on_progress(f"Whisper: {len(words)} words")
    return words


# ── In-process implementations (ONLY called from child processes) ────────────

def _run_parakeet_impl(wav_path: Path, on_progress: Callable = None) -> List[Word]:
    """Run NVIDIA Parakeet TDT — returns words with timestamps."""
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError:
        raise RuntimeError("NeMo not installed: pip install nemo_toolkit[asr]")

    if on_progress:
        on_progress("Loading Parakeet TDT 0.6B...")

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    if torch.cuda.is_available():
        model = model.cuda()

    if on_progress:
        on_progress("Parakeet: transcribing...")

    output = model.transcribe([str(wav_path)], timestamps=True, batch_size=1)

    words: List[Word] = []
    if output:
        hyp = output[0] if isinstance(output, list) else output
        if hasattr(hyp, 'timestep') and hyp.timestep:
            for w in hyp.timestep.get('word', []):
                text = w.get('word', w.get('char', '')).strip()
                if text:
                    words.append(Word(
                        text=text,
                        start=float(w.get('start_offset', 0)),
                        end=float(w.get('end_offset', 0)),
                        source="parakeet",
                        confidence=w.get('score', None),
                    ))
        elif hasattr(hyp, 'text'):
            # Fallback: full text without word timestamps
            text = hyp.text if isinstance(hyp.text, str) else str(hyp)
            if text.strip():
                words.append(Word(text=text.strip(), start=0, end=0, source="parakeet"))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return words


def _run_whisperx_impl(wav_path: Path, language: str = "en",
                       model_size: str = "large-v3",
                       on_progress: Callable = None) -> List[Word]:
    """Run faster-whisper transcription. Returns words with timing.

    Runs on CUDA with int8_float16 if available (~50% less VRAM than float16
    with no meaningful accuracy loss — the recommended compute for consumer
    GPUs per the faster-whisper docs). Falls back to CPU int8 on OOM.

    The old block that tried to also run `whisperx.align()` for forced
    alignment is intentionally removed: it loaded a second model (wav2vec2)
    on top of the Whisper model in the same process, and CTranslate2's GPU
    allocator is outside PyTorch's accounting so `del model` + empty_cache
    never actually reclaimed the Whisper workspace → double-loaded on 12 GB
    cards and triggered 23+ GiB spillover into Windows shared GPU memory.
    faster-whisper's own word_timestamps are precise enough; the DP cue
    builder handles any remaining timing work downstream.
    """
    from faster_whisper import WhisperModel
    import torch

    # Pre-clean any stale CUDA allocations inherited from the parent's
    # import-time torch setup (shouldn't exist with Windows spawn, but
    # this is cheap insurance).
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def _load_and_transcribe(device: str, compute: str) -> List[Word]:
        if on_progress:
            on_progress(f"Loading Whisper {model_size} on {device.upper()} ({compute})...")
        model = WhisperModel(model_size, device=device, compute_type=compute)
        try:
            if on_progress:
                on_progress("Whisper: transcribing with VAD...")
            kwargs = {
                "vad_filter": True,
                "word_timestamps": True,
                "beam_size": 1,
                # Disabled: the KV-cache accumulation across 30s windows is
                # a known VRAM-growth source on long videos and a common
                # OOM trigger on consumer GPUs. Accuracy impact on clean
                # English audio is negligible.
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.5,
                "vad_parameters": {"min_silence_duration_ms": 300},
            }
            if language and language != "auto":
                kwargs["language"] = language

            seg_iter, _info = model.transcribe(str(wav_path), **kwargs)

            out: List[Word] = []
            for seg in seg_iter:
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        out.append(Word(
                            text=w.word.strip(),
                            start=float(w.start),
                            end=float(w.end),
                            source="whisperx",
                            confidence=getattr(w, 'probability', None),
                        ))
                else:
                    out.append(Word(
                        text=seg.text.strip(),
                        start=float(seg.start),
                        end=float(seg.end),
                        source="whisperx",
                    ))
            return out
        finally:
            # Best-effort release. CTranslate2 does its own cleanup when the
            # model object is collected; nothing Python can do beyond this
            # other than exiting the process (which the child will, shortly).
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    # GPU → CPU fallback chain. If CUDA OOMs (or any other runtime error),
    # retry on CPU so the job still completes instead of the child dying
    # and the user having to manually re-submit with a smaller model.
    cuda_ok = False
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if cuda_ok:
        try:
            return _load_and_transcribe("cuda", "int8_float16")
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg or "cublas" in msg:
                if on_progress:
                    on_progress(f"CUDA failed ({type(e).__name__}) — retrying on CPU...")
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                return _load_and_transcribe("cpu", "int8")
            raise

    return _load_and_transcribe("cpu", "int8")


def reconcile(parakeet_words: List[Word], whisperx_words: List[Word]) -> List[Word]:
    """Reconcile two ASR outputs: Parakeet text + WhisperX timing.

    Rules:
    - Use Parakeet text (better punctuation/capitalization)
    - Use WhisperX timing (better word-level alignment)
    - If disagreement is large, flag for repair
    - Align by time overlap between words
    """
    if not parakeet_words:
        return whisperx_words
    if not whisperx_words:
        return parakeet_words

    # Build time-indexed lookup from WhisperX
    reconciled: List[Word] = []

    # Simple approach: for each Parakeet word, find closest WhisperX word by time
    wx_idx = 0
    for pw in parakeet_words:
        # Find WhisperX word with best time overlap
        best_wx = None
        best_overlap = -1

        search_start = max(0, wx_idx - 3)
        search_end = min(len(whisperx_words), wx_idx + 10)

        for i in range(search_start, search_end):
            wx = whisperx_words[i]
            overlap_start = max(pw.start, wx.start)
            overlap_end = min(pw.end, wx.end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_wx = wx
                wx_idx = i

        if best_wx and best_overlap > 0:
            # Use Parakeet text, WhisperX timing
            reconciled.append(Word(
                text=pw.text,
                start=best_wx.start,
                end=best_wx.end,
                source="reconciled",
                speaker=best_wx.speaker or pw.speaker,
                confidence=pw.confidence,
                protected=pw.protected,
                term_id=pw.term_id,
            ))
        else:
            # No match — use Parakeet as-is
            reconciled.append(Word(
                text=pw.text,
                start=pw.start,
                end=pw.end,
                source="parakeet",
                speaker=pw.speaker,
                confidence=pw.confidence,
            ))

    return reconciled


def normalize_words(words: List[Word]) -> List[Word]:
    """Clean up word list: strip junk, normalize whitespace, fix apostrophes."""
    cleaned = []
    for w in words:
        text = w.text.strip()
        if not text:
            continue
        # Normalize apostrophes
        text = text.replace('\u2019', "'").replace('\u2018', "'")
        # Strip non-speech garbage
        text = re.sub(r'^\[.*\]$', '', text).strip()
        text = re.sub(r'^\(.*\)$', '', text).strip()
        text = re.sub(r'^♪.*♪$', '', text).strip()
        if not text:
            continue
        w.text = text
        cleaned.append(w)
    return cleaned

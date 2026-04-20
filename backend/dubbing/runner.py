"""Pipeline Runner — the NEW modular pipeline orchestrator.

Replaces monolith logic with clean module calls.
pipeline.py becomes a thin wrapper that calls this.

Flow:
  Audio → Parakeet + WhisperX → normalize → glossary tag → reconcile
  → DP cue build → English QC → glossary lock → surgical repair
  → Hindi translate → Hindi fit → glossary validate → format
  → pre-TTS QC → TTS segments export
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Callable
import json

from . import asr_runner, glossary, glossary_builder, cue_builder, qc
from . import translation, fit_hi, format_hi, tts_bridge
from . import slot_recompute, slot_verify, global_stretch, sentence_segmenter
from .contracts import Word, Cue, GlossaryTerm
from .defaults import PRODUCTION


ProgressFn = Callable[[str, float, str], None]


class DubbingRunner:
    """New modular dubbing pipeline runner.

    Single source of truth at each layer:
    - Text source: Parakeet
    - Timing source: WhisperX
    - Cue source: DP cue builder
    - Speech source: fitted Hindi cues
    """

    def __init__(self, work_dir: Path, source_lang: str = "en",
                 target_lang: str = "hi", on_progress: ProgressFn = None):
        self.work_dir = work_dir
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._progress = on_progress or (lambda *_: None)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.words: List[Word] = []
        self.cues: List[Cue] = []
        self.glossary_terms: List[GlossaryTerm] = []

    def report(self, step: str, progress: float, msg: str):
        self._progress(step, progress, msg)

    # ════════════════════════════════════════════════════════════════
    # STAGE A: ASR
    # ════════════════════════════════════════════════════════════════

    def run_asr(self, wav_path: Path, use_parakeet: bool = True,
                whisper_model_size: str = "large-v3") -> List[Word]:
        """Run dual ASR: Parakeet text + WhisperX timing → reconciled words.

        When use_parakeet=False, only Whisper runs and its words are used
        directly (no reconciliation — Whisper provides both text and timing).
        """

        parakeet_words = []
        whisperx_words = []

        # 1. Parakeet (text source)
        if use_parakeet:
            try:
                self.report("transcribe", 0.05, "Running Parakeet TDT (text source)...")
                parakeet_words = asr_runner.run_parakeet(
                    wav_path, on_progress=lambda msg: self.report("transcribe", 0.1, msg))
                self.report("transcribe", 0.3, f"Parakeet: {len(parakeet_words)} words")
            except Exception as e:
                self.report("transcribe", 0.1, f"Parakeet failed: {e}, using Whisper only")

        # 2. Whisper (timing source; also text source when Parakeet is off)
        try:
            self.report("transcribe", 0.35,
                        f"Running Whisper {whisper_model_size} (isolated process)...")
            whisperx_words = asr_runner.run_whisperx(
                wav_path, language=self.source_lang,
                model_size=whisper_model_size,
                on_progress=lambda msg: self.report("transcribe", 0.5, msg))
            self.report("transcribe", 0.6, f"Whisper: {len(whisperx_words)} words")
        except Exception as e:
            self.report("transcribe", 0.4, f"Whisper failed: {e}")
            if not parakeet_words:
                # Nothing to fall back on — either Parakeet was off, or it
                # also failed above. Either way the ASR step is done.
                raise RuntimeError(f"ASR failed: {e}")

        # 3. Normalize
        if parakeet_words:
            parakeet_words = asr_runner.normalize_words(parakeet_words)
        if whisperx_words:
            whisperx_words = asr_runner.normalize_words(whisperx_words)

        # 4. Reconcile: Parakeet text + WhisperX timing
        if parakeet_words and whisperx_words:
            self.report("transcribe", 0.65, "Reconciling ASR outputs...")
            self.words = asr_runner.reconcile(parakeet_words, whisperx_words)
        elif parakeet_words:
            self.words = parakeet_words
        else:
            self.words = whisperx_words

        self.report("transcribe", 0.7, f"ASR complete: {len(self.words)} reconciled words")
        return self.words

    def load_words_from_segments(self, segments: List[dict]) -> List[Word]:
        """Load words from existing pipeline segments (for gradual migration)."""
        self.words = []
        for seg in segments:
            # If segment has word-level data
            if seg.get("words"):
                for w in seg["words"]:
                    self.words.append(Word(
                        text=w.get("word", w.get("text", "")),
                        start=w.get("start", 0),
                        end=w.get("end", 0),
                        source="imported",
                    ))
            else:
                # Segment-level: treat whole segment as one "word"
                text = seg.get("text", "").strip()
                if text:
                    for word_text in text.split():
                        self.words.append(Word(
                            text=word_text,
                            start=seg.get("start", 0),
                            end=seg.get("end", 0),
                            source="imported",
                        ))
        self.words = asr_runner.normalize_words(self.words)
        return self.words

    # ════════════════════════════════════════════════════════════════
    # STAGE B: GLOSSARY + CUE BUILDING
    # ════════════════════════════════════════════════════════════════

    def load_glossary(self, glossary_path: Path = None):
        """Load glossary from file, or auto-extract if no file exists."""
        if glossary_path and glossary_path.exists():
            self.glossary_terms = glossary.load_glossary(glossary_path)
            self.report("transcribe", 0.72, f"Loaded {len(self.glossary_terms)} glossary terms")
        else:
            # Auto-extract as fallback (offline ideally, but acceptable for first run)
            self.glossary_terms = glossary_builder.extract_terms_from_words(self.words)
            self.report("transcribe", 0.72, f"Auto-extracted {len(self.glossary_terms)} glossary terms")
            # Save for next run
            if glossary_path:
                glossary_builder.save_glossary(self.glossary_terms, glossary_path)

    def build_cues(self, segmenter: str = "dp",
                   buffer_pct: float = 0.20,
                   max_sentences_per_cue: int = 2) -> List[Cue]:
        """Tag words → build cues → English QC.

        Args:
            segmenter: "dp" (default — DP optimal) or "sentence" (sentence-first packing)
            buffer_pct: Hindi expansion buffer for sentence segmenter (default 20%)
            max_sentences_per_cue: max sentences per cue for sentence segmenter (default 2)
        """
        # 1. Tag words (protect glossary terms from being split)
        self.report("transcribe", 0.75, "Tagging glossary terms on word stream...")
        self.words = glossary.tag_words(self.words, self.glossary_terms)

        # 2. Build cues — choose segmenter
        if segmenter == "sentence":
            self.report("transcribe", 0.8,
                        f"Building sentence-based cues (buffer={buffer_pct:.0%}, "
                        f"max_sent={max_sentences_per_cue})...")
            self.cues = sentence_segmenter.build_cues_by_sentence(
                self.words,
                buffer_pct=buffer_pct,
                max_sentences_per_cue=max_sentences_per_cue,
            )
            stats = sentence_segmenter.summary(self.words, self.cues, buffer_pct)
            self.report("transcribe", 0.85,
                        f"Sentence segmenter: {stats['total_sentences']} sentences → "
                        f"{stats['total_cues']} cues "
                        f"(avg {stats['avg_words_per_cue']} words/cue, "
                        f"budget {stats['word_budget_per_cue']})")
        else:
            self.report("transcribe", 0.8, "Building optimal cue boundaries (DP)...")
            self.cues = cue_builder.build_cues(self.words)
            self.report("transcribe", 0.85, f"Built {len(self.cues)} cues")

        # 3. Tag cues with glossary (for translation lock)
        self.cues = glossary.tag_cues(self.cues, self.glossary_terms)

        # 4. English QC
        self.report("transcribe", 0.9, "Running English QC...")
        self.cues = qc.english_qc(self.cues)
        issues = qc.count_issues(self.cues)
        self.report("transcribe", 0.95,
                     f"English QC: {issues['pass_rate']:.0%} pass "
                     f"({issues['flagged_cues']}/{issues['total_cues']} flagged)")

        return self.cues

    # ════════════════════════════════════════════════════════════════
    # STAGE C: TRANSLATION + HINDI FITTING
    # ════════════════════════════════════════════════════════════════

    def translate(self, translate_fn: Callable = None) -> List[Cue]:
        """Translate cues to Hindi.

        translate_fn: external function that takes (cue.text_clean_en, hints) → Hindi text
        If not provided, translate_cues() must be called with an actual translator.
        """
        self.report("translate", 0.0, f"Translating {len(self.cues)} cues to Hindi...")

        # Build translation hints for ALL cues (duration, word targets, etc.)
        for cue in self.cues:
            if not cue.text_clean_en.strip():
                continue
            target_words = max(3, int((cue.duration / 60.0) * translation.HINDI_WPM))
            cue._translation_hints = {
                "duration_ms": int(cue.duration * 1000),
                "target_words": target_words,
                "protected_terms": cue.protected_terms,
                "speaker": cue.speaker,
                "emotion": cue.emotion,
            }

        if translate_fn:
            # Use external translator (e.g., monolith's _translate_segments)
            for i, cue in enumerate(self.cues):
                if cue.text_clean_en.strip():
                    hints = getattr(cue, '_translation_hints', {})
                    cue.text_hi_raw = translate_fn(cue.text_clean_en, hints)
                if (i + 1) % 20 == 0:
                    self.report("translate", 0.1 + 0.7 * ((i + 1) / len(self.cues)),
                                f"Translated {i + 1}/{len(self.cues)}")
        else:
            # No translator provided — cues stay with empty text_hi_raw
            self.report("translate", 0.5,
                        "WARNING: No translate_fn provided — Hindi text will be empty")

        self.report("translate", 0.8, "Translation complete")
        return self.cues

    def fit_hindi(self) -> List[Cue]:
        """Hindi fitting → glossary validation → formatting → pre-TTS QC."""

        # 1. Dub-fit rewrite (formal→spoken, compression)
        self.report("translate", 0.85, "Hindi fitting...")
        self.cues = fit_hi.fit_cues(self.cues, self.glossary_terms)

        # 2. Glossary validation (check terms survived)
        self.report("translate", 0.9, "Validating glossary terms in Hindi...")
        self.cues = glossary.validate_hindi(self.cues, self.glossary_terms)

        # 3. Subtitle formatting (AFTER validation)
        self.report("translate", 0.92, "Formatting Hindi subtitles...")
        self.cues = format_hi.format_cues(self.cues)

        # 4. Pre-TTS QC gate
        self.report("translate", 0.95, "Pre-TTS QC gate...")
        self.cues = qc.pre_tts_qc(self.cues)
        issues = qc.count_issues(self.cues)
        self.report("translate", 0.98,
                     f"Pre-TTS QC: {issues['pass_rate']:.0%} pass "
                     f"({issues['flagged_cues']}/{issues['total_cues']} flagged)")

        return self.cues

    # ════════════════════════════════════════════════════════════════
    # STAGE E: SLOT RECOMPUTE + VERIFY (optional — off by default)
    # ════════════════════════════════════════════════════════════════

    def recompute_slots(
        self,
        tts_wav_dir: Path,
        wav_pattern: str = "seg_{id:04d}.wav",
        av_sync_mode: str = None,
        max_audio_speedup: float = None,
        min_video_speed: float = None,
        slot_verify_mode: str = None,
        ffprobe: str = "ffprobe",
    ) -> List[Cue]:
        """Measure TTS durations → recompute slot timeline → verify.

        Call this AFTER TTS renders wavs but BEFORE final video assembly.
        Only runs if av_sync_mode != "original".

        Args:
            tts_wav_dir: directory containing per-segment TTS wav files
            wav_pattern: filename pattern with {id} placeholder
            av_sync_mode: "original" | "capped" | "audio_first" (default from config)
            max_audio_speedup: cap for capped mode (default 1.30)
            min_video_speed: floor before flagging (default 0.70)
            slot_verify_mode: "off" | "dry_run" | "auto_fix" | "post_verify"
            ffprobe: path to ffprobe binary
        """
        mode = av_sync_mode if av_sync_mode is not None else PRODUCTION.av_sync_mode
        max_speed = max_audio_speedup if max_audio_speedup is not None else PRODUCTION.max_audio_speedup
        min_vspeed = min_video_speed if min_video_speed is not None else PRODUCTION.min_video_speed
        verify = slot_verify_mode if slot_verify_mode is not None else PRODUCTION.slot_verify

        if mode == "original":
            self.report("slot_recompute", 0.0,
                        "AV sync mode = original — skipping slot recompute")
            return self.cues

        if not tts_wav_dir.is_dir():
            raise FileNotFoundError(f"tts_wav_dir does not exist: {tts_wav_dir}")

        # 1. Measure TTS durations via ffprobe
        self.report("slot_recompute", 0.1,
                     f"Measuring TTS durations for {len(self.cues)} segments...")
        self.cues = slot_recompute.measure_tts_durations(
            self.cues, tts_wav_dir, wav_pattern, ffprobe)

        measured = sum(1 for c in self.cues if c.tts_duration is not None)
        self.report("slot_recompute", 0.3,
                     f"Measured {measured}/{len(self.cues)} TTS durations")

        # 2. Recompute slots
        self.report("slot_recompute", 0.4,
                     f"Recomputing slots (mode={mode}, max_speedup={max_speed}x)...")
        self.cues = slot_recompute.recompute_slots(
            self.cues,
            mode=mode,
            max_speedup=max_speed,
            min_video_speed=min_vspeed,
            drift_warn_ms=PRODUCTION.slot_drift_warn_ms,
            drift_fail_ms=PRODUCTION.slot_drift_fail_ms,
        )

        stats = slot_recompute.summary(self.cues)
        self.report("slot_recompute", 0.7,
                     f"Slots recomputed: {stats.get('expansion_pct', 0):+.1f}% expansion, "
                     f"{stats.get('ok', 0)} OK / {stats.get('warn', 0)} warn / "
                     f"{stats.get('fail', 0)} fail")

        # 3. Verify (if enabled)
        if verify == "dry_run":
            self.report("slot_recompute", 0.8, "Running slot verify (dry run)...")
            report = slot_verify.dry_run(
                self.cues, self.work_dir,
                PRODUCTION.slot_drift_warn_ms, PRODUCTION.slot_drift_fail_ms)
            self.report("slot_recompute", 0.95, "Slot verify report saved")
        elif verify == "auto_fix":
            self.report("slot_recompute", 0.8, "Running slot verify (auto fix)...")
            report = slot_verify.auto_fix(
                self.cues, self.work_dir,
                PRODUCTION.slot_drift_warn_ms, PRODUCTION.slot_drift_fail_ms)
            self.report("slot_recompute", 0.95, "Slot verify auto-fix complete")

        self.report("slot_recompute", 1.0, "Slot recompute complete")
        return self.cues

    def post_verify(
        self,
        assembled_video: Path,
        ffprobe: str = "ffprobe",
    ) -> str:
        """Run post-assembly verification — compare final video duration vs expected.

        Call this AFTER the final video has been assembled.
        """
        self.report("slot_recompute", 0.0, "Running post-assembly verification...")
        report = slot_verify.post_verify(
            self.cues, assembled_video, self.work_dir,
            ffprobe=ffprobe,
            drift_warn_ms=PRODUCTION.slot_drift_warn_ms,
            drift_fail_ms=PRODUCTION.slot_drift_fail_ms,
        )
        self.report("slot_recompute", 1.0, "Post-verify complete")
        return report

    # ════════════════════════════════════════════════════════════════
    # STAGE F: GLOBAL STRETCH (alternative to per-segment recompute)
    # ════════════════════════════════════════════════════════════════

    def compute_global_stretch(
        self,
        tts_wav_dir: Path,
        original_video_duration: float,
        audio_speedup: float = 1.25,
        ffprobe: str = "ffprobe",
    ) -> "global_stretch.GlobalStretchResult":
        """Measure total TTS audio → compute one uniform video speed.

        No per-segment logic. Just: total audio @1.25x vs original video → one speed.

        Args:
            tts_wav_dir: directory containing all TTS wav files
            original_video_duration: original video length in seconds
            audio_speedup: speedup applied to all TTS audio (default 1.25x)
            ffprobe: path to ffprobe binary

        Returns:
            GlobalStretchResult with video_speed, overflow, report, etc.
        """
        self.report("global_stretch", 0.1,
                     f"Computing global stretch ({len(self.cues)} cues, "
                     f"speedup={audio_speedup}x)...")

        segments = tts_bridge.export_tts_segments(self.cues)

        result = global_stretch.compute_global_stretch(
            tts_wav_dir=tts_wav_dir,
            original_video_duration=original_video_duration,
            audio_speedup=audio_speedup,
            segments=segments,
            ffprobe=ffprobe,
        )

        # Save report
        global_stretch.save_report(result, self.work_dir)

        self.report("global_stretch", 1.0,
                     f"Global stretch: {result.total_audio:.1f}s audio vs "
                     f"{result.original_video_duration:.1f}s video → "
                     f"{result.video_speed:.2f}x video speed "
                     f"(overflow {result.overflow:+.1f}s)")

        return result

    # ════════════════════════════════════════════════════════════════
    # STAGE D: EXPORT
    # ════════════════════════════════════════════════════════════════

    def export_for_tts(self) -> List[dict]:
        """Export cues as pipeline-compatible TTS segments."""
        return tts_bridge.export_tts_segments(self.cues)

    def export_srt(self, output_path: Path, text_key: str = "text_hi_display",
                   use_revised_timeline: bool = False):
        """Export Hindi SRT subtitle file."""
        tts_bridge.export_srt(self.cues, output_path, text_key,
                              use_revised_timeline=use_revised_timeline)

    def export_csv(self, output_path: Path):
        """Export for ElevenLabs manual dub format."""
        tts_bridge.export_csv(self.cues, output_path)

    def export_json(self, output_path: Path):
        """Export full cue data for debugging."""
        tts_bridge.export_json(self.cues, output_path)

    def export_source_srt(self, output_path: Path):
        """Export English source SRT."""
        tts_bridge.export_srt(self.cues, output_path, text_key="text_clean_en")

    # ════════════════════════════════════════════════════════════════
    # CONVENIENCE: Full run
    # ════════════════════════════════════════════════════════════════

    def run_full(self, wav_path: Path, translate_fn: Callable = None,
                 glossary_path: Path = None, use_parakeet: bool = True,
                 whisper_model_size: str = "large-v3",
                 # ── Segmenter options ──
                 segmenter: str = "dp",
                 buffer_pct: float = 0.20,
                 max_sentences_per_cue: int = 2,
                 # ── Slot recompute options (all optional) ──
                 tts_wav_dir: Path = None,
                 av_sync_mode: str = None,
                 max_audio_speedup: float = None,
                 min_video_speed: float = None,
                 slot_verify_mode: str = None,
                 ) -> List[dict]:
        """Run the complete pipeline: ASR → cues → translate → fit → [slot recompute] → export.

        Returns TTS-ready segments for pipeline.py to synthesize.

        Slot recompute only runs if:
          1. tts_wav_dir is provided (TTS wavs must already exist), AND
          2. av_sync_mode is not "original"
        """
        # ASR
        self.run_asr(wav_path, use_parakeet=use_parakeet,
                     whisper_model_size=whisper_model_size)

        # Glossary
        self.load_glossary(glossary_path)

        # Cue building
        self.build_cues(
            segmenter=segmenter,
            buffer_pct=buffer_pct,
            max_sentences_per_cue=max_sentences_per_cue,
        )

        # Translation
        self.translate(translate_fn)

        # Hindi fitting + QC
        self.fit_hindi()

        # Slot recompute (optional — only if TTS wavs dir provided)
        mode = av_sync_mode if av_sync_mode is not None else PRODUCTION.av_sync_mode
        if tts_wav_dir and mode != "original":
            self.recompute_slots(
                tts_wav_dir=tts_wav_dir,
                av_sync_mode=mode,
                max_audio_speedup=max_audio_speedup,
                min_video_speed=min_video_speed,
                slot_verify_mode=slot_verify_mode,
            )

        # Export
        segments = self.export_for_tts()

        # Also save debug artifacts
        self.export_json(self.work_dir / "cues_debug.json")
        self.export_csv(self.work_dir / "cues_elevenlabs.csv")

        self.report("translate", 1.0, f"Pipeline complete: {len(segments)} segments ready for TTS")

        return segments

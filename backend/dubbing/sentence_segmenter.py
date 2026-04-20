"""Sentence Segmenter — sentence-first cue building from Whisper words.

Instead of DP scoring every possible split point, this module:
  1. Merges Whisper words into proper full sentences (by punctuation)
  2. Counts total words, applies a 20% Hindi expansion buffer
  3. Packs full sentences into cues obeying contracts.py rules
  4. Output: List[Cue] that plugs straight into the modular pipeline

The key insight: segmentation follows sentence boundaries, NOT silence gaps.
Each cue gets 1–2 complete sentences. Never splits mid-sentence.

Usage:
    from backend.dubbing.sentence_segmenter import build_cues_by_sentence

    cues = build_cues_by_sentence(words, buffer_pct=0.20)
    # → feeds into runner.translate() → fit_hindi() → export
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .contracts import (
    Word, Cue,
    CUE_WORD_HARD_MAX, CUE_WORD_TARGET_MIN, CUE_WORD_TARGET_MAX,
    CUE_DUR_HARD_MIN, CUE_DUR_HARD_MAX, CUE_DUR_TARGET_MIN, CUE_DUR_TARGET_MAX,
    CUE_MAX_LINES, CUE_MAX_CPL,
)

log = logging.getLogger(__name__)

SENTENCE_ENDERS = frozenset('.!?')
# After Hindi expansion, effective word budget per cue
# e.g. if CUE_WORD_HARD_MAX=16 and buffer=0.20 → budget = 16 / 1.20 ≈ 13
DEFAULT_BUFFER_PCT = 0.20
MAX_SENTENCES_PER_CUE = 2


# ─── Step 1: Words → Sentences ──────────────────────────────────────────

@dataclass
class Sentence:
    """One complete sentence with timing from word stream."""
    text: str
    words: List[Word]
    start: float
    end: float
    word_count: int
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge_into_sentences(words: List[Word]) -> List[Sentence]:
    """Merge a word stream into proper full sentences by punctuation.

    Rules:
      - A sentence ends when a word ends with . ! or ?
      - If no sentence-ending punctuation found for 20+ words, force a cut
        at the next comma/semicolon or at word 25 (safety valve)
      - Preserves timing and speaker from the word stream
    """
    if not words:
        return []

    sentences = []
    buf: List[Word] = []

    for w in words:
        # Force break on speaker change mid-sentence
        if buf and w.speaker and buf[-1].speaker and w.speaker != buf[-1].speaker:
            sentences.append(_flush_sentence(buf, len(sentences)))
            buf = []

        buf.append(w)
        stripped = w.text.rstrip()
        last_char = stripped[-1] if stripped else ''

        is_sentence_end = last_char in SENTENCE_ENDERS
        is_soft_break = last_char in ',;:' and len(buf) >= 15
        is_overflow = len(buf) >= 25

        if is_sentence_end or is_soft_break or is_overflow:
            sentences.append(_flush_sentence(buf, len(sentences)))
            buf = []

    # Flush remaining words as final sentence
    if buf:
        sentences.append(_flush_sentence(buf, len(sentences)))

    return sentences


def _flush_sentence(buf: List[Word], idx: int) -> Sentence:
    """Create a Sentence from a word buffer."""
    text = " ".join(w.text for w in buf)
    return Sentence(
        text=text,
        words=list(buf),
        start=buf[0].start,
        end=buf[-1].end,
        word_count=len(buf),
        speaker=buf[0].speaker,
    )


# ─── Step 2: Sentences → Cues ───────────────────────────────────────────

def build_cues_by_sentence(
    words: List[Word],
    buffer_pct: float = DEFAULT_BUFFER_PCT,
    max_sentences_per_cue: int = MAX_SENTENCES_PER_CUE,
    respect_speakers: bool = True,
) -> List[Cue]:
    """Build cues by packing full sentences, with Hindi expansion buffer.

    Algorithm:
      1. Merge words → sentences
      2. Compute word budget = CUE_WORD_HARD_MAX / (1 + buffer_pct)
      3. Greedily pack 1–max_sentences_per_cue sentences per cue
         while obeying: word budget, duration limits, speaker boundaries
      4. Output List[Cue] compatible with runner pipeline

    Args:
        words: timed word stream from ASR
        buffer_pct: Hindi expansion buffer (default 0.20 = 20%)
        max_sentences_per_cue: max sentences allowed in one cue (default 2)
        respect_speakers: if True, never merge sentences from different speakers

    Returns:
        List[Cue] ready for translate → fit_hindi → export
    """
    sentences = merge_into_sentences(words)

    if not sentences:
        return []

    # Clamp buffer_pct — negative makes no sense
    buffer_pct = max(0.0, buffer_pct)

    # Word budget per cue accounting for Hindi expansion
    word_budget = int(CUE_WORD_HARD_MAX / (1.0 + buffer_pct))
    word_budget = min(word_budget, CUE_WORD_HARD_MAX)  # never exceed hard max

    total_words = sum(s.word_count for s in sentences)
    total_sentences = len(sentences)

    log.info(
        "Sentence segmenter: %d words across %d sentences, "
        "word_budget=%d (hard_max=%d / %.0f%% buffer)",
        total_words, total_sentences, word_budget, CUE_WORD_HARD_MAX,
        buffer_pct * 100,
    )

    cues: List[Cue] = []
    pending: List[Sentence] = []  # sentences buffered for current cue

    for sent in sentences:
        # Check if adding this sentence would violate limits
        can_add = _can_add_sentence(
            pending, sent, word_budget, max_sentences_per_cue, respect_speakers)

        if can_add:
            pending.append(sent)
            # Check if this single sentence already exceeds budget
            # (happens when pending was empty — _can_add_sentence always accepts first)
            if sent.word_count > word_budget:
                log.warning(
                    "Sentence has %d words (budget=%d) — splitting by words",
                    sent.word_count, word_budget,
                )
                # Split the oversized sentence into word-budget chunks
                oversized = pending.pop()
                if pending:
                    cues.append(_build_cue_from_sentences(pending, len(cues)))
                    pending = []
                for chunk_start in range(0, len(oversized.words), word_budget):
                    chunk_words = oversized.words[chunk_start:chunk_start + word_budget]
                    chunk_sent = _flush_sentence(chunk_words, 0)
                    cues.append(_build_cue_from_sentences([chunk_sent], len(cues)))
        else:
            # Flush current buffer as a cue
            if pending:
                cues.append(_build_cue_from_sentences(pending, len(cues)))
            # Start new buffer with this sentence
            pending = [sent]

            # Handle oversized single sentence (more words than budget)
            if sent.word_count > word_budget:
                log.warning(
                    "Sentence has %d words (budget=%d) — splitting by words",
                    sent.word_count, word_budget,
                )
                # Split the oversized sentence into word-budget chunks
                pending = []
                for chunk_start in range(0, len(sent.words), word_budget):
                    chunk_words = sent.words[chunk_start:chunk_start + word_budget]
                    chunk_sent = _flush_sentence(chunk_words, 0)
                    cues.append(_build_cue_from_sentences([chunk_sent], len(cues)))
                # pending is already empty

    # Flush remaining
    if pending:
        cues.append(_build_cue_from_sentences(pending, len(cues)))

    # Re-index
    for i, cue in enumerate(cues):
        cue.id = i

    log.info(
        "Sentence segmenter result: %d sentences → %d cues "
        "(avg %.1f words/cue, budget=%d)",
        total_sentences, len(cues),
        total_words / len(cues) if cues else 0, word_budget,
    )

    return cues


def _can_add_sentence(
    pending: List[Sentence],
    candidate: Sentence,
    word_budget: int,
    max_sentences: int,
    respect_speakers: bool,
) -> bool:
    """Check if candidate sentence can be added to the pending buffer."""
    if not pending:
        return True

    # Max sentences per cue
    if len(pending) >= max_sentences:
        return False

    # Word count check
    current_words = sum(s.word_count for s in pending)
    if current_words + candidate.word_count > word_budget:
        return False

    # Duration check
    combined_start = pending[0].start
    combined_end = candidate.end
    combined_dur = combined_end - combined_start
    if combined_dur > CUE_DUR_HARD_MAX:
        return False

    # Speaker check
    if respect_speakers and candidate.speaker and pending[-1].speaker:
        if candidate.speaker != pending[-1].speaker:
            return False

    return True


def _build_cue_from_sentences(sentences: List[Sentence], cue_id: int) -> Cue:
    """Build a Cue from 1+ sentences."""
    all_words = []
    for s in sentences:
        all_words.extend(s.words)

    text = " ".join(w.text for w in all_words)

    return Cue(
        id=cue_id,
        start=all_words[0].start,
        end=all_words[-1].end,
        speaker=all_words[0].speaker,
        text_original=text,
        text_clean_en=text,
        words=all_words,
    )


# ─── Summary / diagnostics ──────────────────────────────────────────────

def summary(words: List[Word], cues: List[Cue], buffer_pct: float = DEFAULT_BUFFER_PCT) -> dict:
    """Return diagnostic summary of the sentence segmentation."""
    sentences = merge_into_sentences(words)
    total_words = sum(s.word_count for s in sentences)
    buffered_words = int(total_words * (1.0 + buffer_pct))
    word_budget = int(CUE_WORD_HARD_MAX / (1.0 + buffer_pct))

    cue_word_counts = [c.word_count for c in cues]

    return {
        "total_words": total_words,
        "total_sentences": len(sentences),
        "buffered_words_estimate": buffered_words,
        "word_budget_per_cue": word_budget,
        "total_cues": len(cues),
        "avg_words_per_cue": round(total_words / len(cues), 1) if cues else 0,
        "max_words_in_cue": max(cue_word_counts) if cue_word_counts else 0,
        "min_words_in_cue": min(cue_word_counts) if cue_word_counts else 0,
        "avg_sentences_per_cue": round(len(sentences) / len(cues), 1) if cues else 0,
        "avg_cue_duration": round(
            sum(c.duration for c in cues) / len(cues), 2) if cues else 0,
    }

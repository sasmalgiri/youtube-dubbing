# Custom voice references for XTTS cloning

Drop a WAV or MP3 file here to override the automatic speaker reference the
pipeline extracts from the source video.

## When you'd use this

- Source video has **intro music or a logo** for the first 20-40 seconds — auto-extraction gets poisoned by music, clone sounds off. Manual reference = clean clone.
- Source has **multiple speakers** — auto-extraction picks whichever voice the first clean window happens to belong to. Manual reference lets you fix which one gets cloned.
- Source is **mostly captions / screen** with only brief voice — there may not be enough continuous speech to auto-pick. Drop your own 10-second speaker sample.
- You want a **consistent voice across many videos** (brand voice) — use the same reference WAV every time.

## What works

- **Format**: WAV (preferred) or MP3. Gets auto-converted to 22050 Hz mono inside.
- **Length**: 6–15 seconds is the sweet spot. XTTS truncates past ~30s anyway.
- **Content**: clean spoken speech by a single speaker. No music, minimal noise. Quiet room + a decent mic is better than a studio that has background music.
- **Language**: doesn't have to be the target dub language. XTTS is cross-lingual — a 10-second English reference produces a Hindi clone of the same voice.

## What the pipeline does with it

1. If this directory has any `*.wav` or `*.mp3`, the first one (alphabetical) becomes the reference — auto-extraction from the source is skipped entirely.
2. Your file is converted to 22050 Hz mono, trimmed to the first 10 seconds, saved as `work/voice_ref.wav` for the job.
3. Pre-flight validation checks the clip isn't silent / too short before model load. If it fails validation you'll see an actionable error in the job log.

## Safety

- Files here are in `.gitignore` (`voices/my_voice_refs/*.wav` etc.) — they never get committed, so personal voice samples stay local.
- This README file is committed so the directory exists in a fresh clone.

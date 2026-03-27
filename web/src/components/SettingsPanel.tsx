'use client';

import { useState } from 'react';

export interface DubbingSettings {
    asr_model: string;
    translation_engine: string;
    tts_rate: string;
    mix_original: boolean;
    original_volume: number;
    use_cosyvoice: boolean;
    use_chatterbox: boolean;
    use_elevenlabs: boolean;
    use_google_tts: boolean;
    use_coqui_xtts: boolean;
    use_edge_tts: boolean;
    prefer_youtube_subs: boolean;
    use_yt_translate: boolean;
    multi_speaker: boolean;
    transcribe_only: boolean;
    audio_priority: boolean;
    audio_bitrate: string;
    encode_preset: string;
    split_duration: number;
    fast_assemble: boolean;
    dub_chain: string[];
    enable_manual_review: boolean;
}

interface SettingsPanelProps {
    settings: DubbingSettings;
    onChange: (settings: DubbingSettings) => void;
}

export default function SettingsPanel({ settings, onChange }: SettingsPanelProps) {
    const [open, setOpen] = useState(false);

    const update = (partial: Partial<DubbingSettings>) => {
        onChange({ ...settings, ...partial });
    };

    return (
        <div className="glass-card overflow-hidden">
            <button
                onClick={() => setOpen(!open)}
                className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-white/[0.02] transition-colors"
            >
                <div className="flex items-center gap-2">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-text-muted">
                        <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                        <circle cx="12" cy="12" r="3" />
                    </svg>
                    <span className="text-sm font-medium text-text-secondary">Advanced Settings</span>
                </div>
                <svg
                    width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                    className={`text-text-muted transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
                >
                    <path d="m6 9 6 6 6-6" />
                </svg>
            </button>

            {open && (
                <div className="px-5 pb-5 space-y-5 animate-slide-up border-t border-border pt-4">
                    {/* ── Transcription Section ── */}
                    <div>
                        <p className="text-sm font-medium text-text-primary mb-1">Transcription (Whisper)</p>
                        <p className="text-[10px] text-text-muted mb-3">Converts speech in the video to text. Larger models are slower but more accurate.</p>
                        <div className="space-y-3">
                            <div>
                                <p className="text-xs text-text-muted mb-1.5">Whisper Model</p>
                                <div className="grid grid-cols-5 gap-1.5">
                                    {[
                                        { value: 'base',          label: 'Base',    desc: 'Fastest' },
                                        { value: 'small',         label: 'Small',   desc: 'Fast' },
                                        { value: 'medium',        label: 'Medium',  desc: 'Balanced' },
                                        { value: 'large-v3-turbo',label: 'Turbo',   desc: 'Fast+accurate' },
                                        { value: 'large-v3',      label: 'Large-v3',desc: 'Best' },
                                    ].map((m) => (
                                        <button
                                            key={m.value}
                                            onClick={() => update({ asr_model: m.value })}
                                            className={`
                                                px-3 py-2 rounded-lg text-xs text-center transition-all border
                                                ${settings.asr_model === m.value
                                                    ? 'bg-primary/20 border-primary text-primary-light'
                                                    : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                            `}
                                        >
                                            <div className="font-medium">{m.label}</div>
                                            <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* YouTube Subtitles */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Use YouTube Subtitles</p>
                                    <p className="text-xs text-text-muted">Skip Whisper, use existing subs (faster)</p>
                                </div>
                                <button
                                    type="button" title="Toggle YouTube Subtitles" onClick={() => update({ prefer_youtube_subs: !settings.prefer_youtube_subs })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.prefer_youtube_subs ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.prefer_youtube_subs ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>

                            {/* YouTube Auto-Translate */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">YT Auto-Translate</p>
                                    <p className="text-xs text-text-muted">Use YouTube&apos;s translated subs (skips Whisper + translation)</p>
                                </div>
                                <button
                                    type="button" title="Toggle YouTube Translate" onClick={() => update({ use_yt_translate: !settings.use_yt_translate })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_yt_translate ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_yt_translate ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>

                            {/* Chain Dub */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Chain Dub (English → Hindi)</p>
                                    <p className="text-xs text-text-muted">Dub to English first using subs, then English to Hindi (best for non-English videos)</p>
                                </div>
                                <button
                                    onClick={() => update({
                                        dub_chain: settings.dub_chain.length > 0 ? [] : ['en', 'hi'],
                                    })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.dub_chain.length > 0 ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.dub_chain.length > 0 ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* ── Translation Section ── */}
                    <div>
                        <p className="text-sm font-medium text-text-primary mb-1">Translation Engine</p>
                        <p className="text-[10px] text-text-muted mb-3">How the transcribed text gets translated. Auto picks the best for Hindi. Chain Dub gives highest quality: IndicTrans2+ first, then Turbo engines refine the output (best for long videos).</p>
                        <div className="grid grid-cols-5 gap-2 mb-3">
                            {[
                                { value: 'auto', label: 'Auto', desc: 'Best available' },
                                { value: 'turbo', label: 'Turbo', desc: 'Groq+SambaNova parallel' },
                                { value: 'groq', label: 'Groq', desc: 'Llama 3.3 70B (free)' },
                                { value: 'sambanova', label: 'SambaNova', desc: 'Llama 3.3 70B (free)' },
                                { value: 'gemini', label: 'Gemini', desc: 'Google AI (free)' },
                            ].map((m) => (
                                <button
                                    key={m.value}
                                    onClick={() => update({ translation_engine: m.value })}
                                    className={`
                                        px-3 py-2 rounded-lg text-xs text-center transition-all border
                                        ${settings.translation_engine === m.value
                                            ? 'bg-primary/20 border-primary text-primary-light'
                                            : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                    `}
                                >
                                    <div className="font-medium">{m.label}</div>
                                    <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                </button>
                            ))}
                        </div>
                        <div className="grid grid-cols-4 gap-2">
                            {[
                                { value: 'chain_dub', label: 'Chain Dub', desc: 'IndicTrans2+ → Turbo refine' },
                                { value: 'nllb_polish', label: 'IndicTrans2+', desc: 'IndicTrans2 → LLM → Rules' },
                                { value: 'google_polish', label: 'Google+Polish', desc: 'Fast Google → LLM polish' },
                                { value: 'nllb', label: 'IndicTrans2', desc: 'Local meaning model' },
                                { value: 'ollama', label: 'Ollama', desc: 'Local LLM (GPU)' },
                                { value: 'hinglish', label: 'Hinglish AI', desc: 'Custom Hindi model' },
                                { value: 'google', label: 'Google', desc: 'Free, basic' },
                            ].map((m) => (
                                <button
                                    key={m.value}
                                    onClick={() => update({ translation_engine: m.value })}
                                    className={`
                                        px-3 py-2 rounded-lg text-xs text-center transition-all border
                                        ${settings.translation_engine === m.value
                                            ? 'bg-primary/20 border-primary text-primary-light'
                                            : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                    `}
                                >
                                    <div className="font-medium">{m.label}</div>
                                    <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Transcribe Only (manual translation) */}
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-text-primary">Transcribe Only</p>
                            <p className="text-xs text-text-muted">Get SRT to translate yourself (e.g. with Claude), then upload back</p>
                        </div>
                        <button
                            type="button" title="Toggle Transcribe Only" onClick={() => update({ transcribe_only: !settings.transcribe_only })}
                            className={`
                                w-11 h-6 rounded-full transition-colors relative
                                ${settings.transcribe_only ? 'bg-primary' : 'bg-white/10'}
                            `}
                        >
                            <div className={`
                                w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                ${settings.transcribe_only ? 'translate-x-6' : 'translate-x-1'}
                            `} />
                        </button>
                    </div>

                    {/* Multi-Speaker Voices */}
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-text-primary">Multi-Speaker Voices</p>
                            <p className="text-xs text-text-muted">Detect speakers & assign distinct voices (needs HF_TOKEN, adds ~30s)</p>
                        </div>
                        <button
                            type="button" title="Toggle Multi-speaker" onClick={() => update({ multi_speaker: !settings.multi_speaker })}
                            className={`
                                w-11 h-6 rounded-full transition-colors relative
                                ${settings.multi_speaker ? 'bg-primary' : 'bg-white/10'}
                            `}
                        >
                            <div className={`
                                w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                ${settings.multi_speaker ? 'translate-x-6' : 'translate-x-1'}
                            `} />
                        </button>
                    </div>

                    {/* TTS Engines */}
                    <div>
                        <p className="text-sm font-medium text-text-primary mb-1">TTS Engines</p>
                        <p className="text-[10px] text-text-muted mb-3">English dub auto-uses: Chatterbox-Turbo → Chatterbox Multilingual → XTTS v2 → Edge-TTS (all free, local). Non-English uses toggles below. CosyVoice 2 = best for Hindi.</p>
                        <div className="space-y-3">
                            {/* CosyVoice 2 */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">CosyVoice 2 ⭐</p>
                                    <p className="text-xs text-text-muted">Free, GPU, near-ElevenLabs quality, voice clones original speaker in Hindi</p>
                                </div>
                                <button
                                    type="button"
                                    title="Toggle CosyVoice 2"
                                    onClick={() => update({ use_cosyvoice: !settings.use_cosyvoice })}
                                    className={`w-11 h-6 rounded-full transition-colors relative ${settings.use_cosyvoice ? 'bg-primary' : 'bg-white/10'}`}
                                >
                                    <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.use_cosyvoice ? 'translate-x-6' : 'translate-x-1'}`} />
                                </button>
                            </div>
                            {/* Chatterbox */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Chatterbox AI</p>
                                    <p className="text-xs text-text-muted">Free, GPU required, most human-like</p>
                                </div>
                                <button
                                    type="button" title="Toggle Chatterbox" onClick={() => update({ use_chatterbox: !settings.use_chatterbox })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_chatterbox ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_chatterbox ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>

                            {/* ElevenLabs */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">ElevenLabs</p>
                                    <p className="text-xs text-text-muted">Paid API, needs ELEVENLABS_API_KEY in .env</p>
                                </div>
                                <button
                                    type="button" title="Toggle ElevenLabs" onClick={() => update({ use_elevenlabs: !settings.use_elevenlabs })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_elevenlabs ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_elevenlabs ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>

                            {/* Coqui XTTS v2 */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Coqui XTTS v2</p>
                                    <p className="text-xs text-text-muted">Free, GPU required, voice cloning from original speaker</p>
                                </div>
                                <button
                                    type="button" title="Toggle Coqui XTTS v2" onClick={() => update({ use_coqui_xtts: !settings.use_coqui_xtts })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_coqui_xtts ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_coqui_xtts ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>

                            {/* Google Cloud TTS */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Google Cloud TTS</p>
                                    <p className="text-xs text-text-muted">Free 1M chars/mo, WaveNet/Neural2 voices, needs GCP credentials</p>
                                </div>
                                <button
                                    type="button" title="Toggle Google TTS" onClick={() => update({ use_google_tts: !settings.use_google_tts })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_google_tts ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_google_tts ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>

                            {/* Edge-TTS */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Edge-TTS</p>
                                    <p className="text-xs text-text-muted">Free, no GPU needed, decent quality</p>
                                </div>
                                <button
                                    type="button" title="Toggle Edge TTS" onClick={() => update({ use_edge_tts: !settings.use_edge_tts })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_edge_tts ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_edge_tts ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>
                        </div>
                        {settings.use_coqui_xtts && settings.use_edge_tts ? (
                            <p className="text-[10px] text-primary mt-2 font-medium">
                                Hybrid Mode: Coqui XTTS + Edge-TTS will run in parallel (~2x faster)
                            </p>
                        ) : (
                            <p className="text-[10px] text-text-muted mt-2">First enabled engine from top to bottom will be used. Enable both Coqui + Edge for hybrid parallel mode.</p>
                        )}
                    </div>

                    {/* TTS Speech Rate */}
                    <div>
                        <label className="label mb-2 block">
                            Speech Rate: <span className="text-primary-light">{settings.tts_rate}</span>
                        </label>
                        <input
                            type="range"
                            min={-50}
                            max={50}
                            value={parseInt(settings.tts_rate) || 0}
                            onChange={(e) => {
                                const v = parseInt(e.target.value);
                                update({ tts_rate: `${v >= 0 ? '+' : ''}${v}%` });
                            }}
                            className="w-full accent-primary"
                        />
                        <div className="flex justify-between text-[10px] text-text-muted">
                            <span>Slower</span>
                            <span>Normal</span>
                            <span>Faster</span>
                        </div>
                    </div>

                    {/* Mix Background Music */}
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-text-primary">Mix Background Music</p>
                            <p className="text-xs text-text-muted">Keep original background music (vocals removed) behind dubbed voice</p>
                        </div>
                        <button
                            type="button" title="Toggle Mix Original Audio" onClick={() => update({ mix_original: !settings.mix_original })}
                            className={`
                                w-11 h-6 rounded-full transition-colors relative
                                ${settings.mix_original ? 'bg-primary' : 'bg-white/10'}
                            `}
                        >
                            <div className={`
                                w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                ${settings.mix_original ? 'translate-x-6' : 'translate-x-1'}
                            `} />
                        </button>
                    </div>

                    {/* Music Volume */}
                    {settings.mix_original && (
                        <div className="animate-slide-up">
                            <label className="label mb-2 block">
                                Music Volume: <span className="text-primary-light">{Math.round(settings.original_volume * 100)}%</span>
                            </label>
                            <input
                                type="range"
                                min={0}
                                max={50}
                                value={settings.original_volume * 100}
                                onChange={(e) => update({ original_volume: parseInt(e.target.value) / 100 })}
                                className="w-full accent-primary"
                                title="Original volume"
                            />
                        </div>
                    )}

                    {/* ── Audio & Performance Section ── */}
                    <div>
                        <p className="text-sm font-medium text-text-primary mb-1">Audio & Performance</p>
                        <p className="text-[10px] text-text-muted mb-3">Controls how audio and video are assembled. Audio Priority lets TTS speak naturally without speed changes.</p>
                        <div className="space-y-3">
                            {/* Audio Priority */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Audio Priority</p>
                                    <p className="text-xs text-text-muted">TTS speaks naturally, video adjusts to match (best for listening)</p>
                                </div>
                                <button
                                    type="button" title="Toggle Audio Priority" onClick={() => update({ audio_priority: !settings.audio_priority })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.audio_priority ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.audio_priority ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>

                            {/* Audio Bitrate */}
                            <div>
                                <p className="text-xs text-text-muted mb-1.5">Audio Quality</p>
                                <div className="grid grid-cols-4 gap-2">
                                    {[
                                        { value: '128k', label: '128k', desc: 'Small file' },
                                        { value: '192k', label: '192k', desc: 'Standard' },
                                        { value: '256k', label: '256k', desc: 'High' },
                                        { value: '320k', label: '320k', desc: 'Best' },
                                    ].map((m) => (
                                        <button
                                            type="button"
                                            key={m.value}
                                            onClick={() => update({ audio_bitrate: m.value })}
                                            className={`
                                                px-2 py-2 rounded-lg text-xs text-center transition-all border
                                                ${settings.audio_bitrate === m.value
                                                    ? 'bg-primary/20 border-primary text-primary-light'
                                                    : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                            `}
                                        >
                                            <div className="font-medium">{m.label}</div>
                                            <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Encode Speed */}
                            <div>
                                <p className="text-xs text-text-muted mb-1.5">Video Encode Speed</p>
                                <div className="grid grid-cols-4 gap-2">
                                    {[
                                        { value: 'ultrafast', label: 'Ultra Fast', desc: 'Fastest' },
                                        { value: 'veryfast', label: 'Very Fast', desc: 'Default' },
                                        { value: 'fast', label: 'Fast', desc: 'Better' },
                                        { value: 'medium', label: 'Medium', desc: 'Best video' },
                                    ].map((m) => (
                                        <button
                                            type="button"
                                            key={m.value}
                                            onClick={() => update({ encode_preset: m.value })}
                                            className={`
                                                px-2 py-2 rounded-lg text-xs text-center transition-all border
                                                ${settings.encode_preset === m.value
                                                    ? 'bg-primary/20 border-primary text-primary-light'
                                                    : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                            `}
                                        >
                                            <div className="font-medium">{m.label}</div>
                                            <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Fast Assemble */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Fast Assemble</p>
                                    <p className="text-xs text-text-muted">In-memory audio (instant). Off = ffmpeg mixing (slower, preserves overlapping audio)</p>
                                </div>
                                <button
                                    type="button" title="Toggle Fast Assemble" onClick={() => update({ fast_assemble: !settings.fast_assemble })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.fast_assemble ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.fast_assemble ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>

                            {/* Manual Review Queue */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Manual Review Queue</p>
                                    <p className="text-xs text-text-muted">Save segments that failed QC after all retries to manual_review_queue.json for inspection</p>
                                </div>
                                <button
                                    type="button" title="Toggle Manual Review Queue" onClick={() => update({ enable_manual_review: !settings.enable_manual_review })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.enable_manual_review ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.enable_manual_review ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>
                        </div>

                        {/* Split Long Videos */}
                        <div>
                            <p className="text-xs text-text-muted mb-1.5">Split Long Videos — breaks video into parts, dubs each separately. Avoids timeout/crashes on 1h+ videos.</p>
                            <div className="grid grid-cols-4 gap-2">
                                {[
                                    { value: 0, label: 'Off', desc: 'No splitting' },
                                    { value: 30, label: '30 min', desc: 'Split every 30m' },
                                    { value: 40, label: '40 min', desc: 'Split every 40m' },
                                    { value: 60, label: '60 min', desc: 'Split every 1h' },
                                ].map((m) => (
                                    <button
                                        key={m.value}
                                        onClick={() => update({ split_duration: m.value })}
                                        className={`
                                            px-3 py-2 rounded-lg text-xs text-center transition-all border
                                            ${settings.split_duration === m.value
                                                ? 'bg-primary/20 border-primary text-primary-light'
                                                : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                        `}
                                    >
                                        <div className="font-medium">{m.label}</div>
                                        <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

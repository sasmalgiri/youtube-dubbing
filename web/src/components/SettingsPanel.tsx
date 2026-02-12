'use client';

import { useState } from 'react';

export interface DubbingSettings {
    tts_rate: string;
    mix_original: boolean;
    original_volume: number;
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
                    {/* TTS Speech Rate */}
                    <div>
                        <label className="label mb-2 block">
                            Speech Rate: <span className="text-primary-light">{settings.tts_rate}</span>
                        </label>
                        <input
                            type="range"
                            min={-50}
                            max={50}
                            value={parseInt(settings.tts_rate)}
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

                    {/* Mix Original Audio */}
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-text-primary">Mix Original Audio</p>
                            <p className="text-xs text-text-muted">Blend original audio softly behind the dubbed voice</p>
                        </div>
                        <button
                            onClick={() => update({ mix_original: !settings.mix_original })}
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

                    {/* Original Volume */}
                    {settings.mix_original && (
                        <div className="animate-slide-up">
                            <label className="label mb-2 block">
                                Original Volume: <span className="text-primary-light">{Math.round(settings.original_volume * 100)}%</span>
                            </label>
                            <input
                                type="range"
                                min={0}
                                max={50}
                                value={settings.original_volume * 100}
                                onChange={(e) => update({ original_volume: parseInt(e.target.value) / 100 })}
                                className="w-full accent-primary"
                            />
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

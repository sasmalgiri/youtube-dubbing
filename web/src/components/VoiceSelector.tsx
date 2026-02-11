'use client';

import { useVoices } from '@/hooks/useVoices';

interface VoiceSelectorProps {
    value: string;
    onChange: (voice: string) => void;
}

export default function VoiceSelector({ value, onChange }: VoiceSelectorProps) {
    const { voices, loading } = useVoices('hi');

    return (
        <div>
            <label className="label mb-2 block">Hindi Voice</label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {loading ? (
                    <div className="col-span-2 text-sm text-text-muted py-3 text-center">
                        Loading voices...
                    </div>
                ) : (
                    voices.map((voice) => {
                        const isSelected = value === voice.ShortName;
                        const isFemale = voice.Gender === 'Female';
                        return (
                            <button
                                key={voice.ShortName}
                                onClick={() => onChange(voice.ShortName)}
                                className={`
                                    flex items-center gap-3 p-3 rounded-xl border transition-all text-left
                                    ${isSelected
                                        ? 'border-primary bg-primary/10 ring-1 ring-primary/30'
                                        : 'border-border bg-white/[0.02] hover:bg-white/5'
                                    }
                                `}
                            >
                                <div className={`
                                    w-9 h-9 rounded-full flex items-center justify-center text-sm
                                    ${isFemale ? 'bg-pink-500/20 text-pink-400' : 'bg-blue-500/20 text-blue-400'}
                                `}>
                                    {isFemale ? (
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <circle cx="12" cy="8" r="5" />
                                            <path d="M20 21a8 8 0 0 0-16 0" />
                                        </svg>
                                    ) : (
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <circle cx="12" cy="8" r="5" />
                                            <path d="M20 21a8 8 0 0 0-16 0" />
                                        </svg>
                                    )}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className={`text-sm font-medium ${isSelected ? 'text-primary-light' : 'text-text-primary'}`}>
                                        {voice.ShortName.split('-').slice(2).join(' ').replace('Neural', '')}
                                    </p>
                                    <p className="text-xs text-text-muted">
                                        {voice.Gender} &middot; {voice.Locale}
                                    </p>
                                </div>
                                {isSelected && (
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M20 6 9 17l-5-5" />
                                    </svg>
                                )}
                            </button>
                        );
                    })
                )}
            </div>
        </div>
    );
}

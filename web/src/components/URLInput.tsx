'use client';

import { useState, useCallback } from 'react';
import { extractYouTubeId, isValidYouTubeUrl, getThumbnailUrl } from '@/lib/utils';

interface URLInputProps {
    onSubmit: (url: string) => void;
    disabled?: boolean;
}

export default function URLInput({ onSubmit, disabled }: URLInputProps) {
    const [url, setUrl] = useState('');
    const videoId = extractYouTubeId(url);
    const isValid = isValidYouTubeUrl(url);

    const handleSubmit = useCallback(() => {
        if (isValid && !disabled) {
            onSubmit(url.trim());
        }
    }, [url, isValid, disabled, onSubmit]);

    const handlePaste = useCallback((e: React.ClipboardEvent) => {
        const pasted = e.clipboardData.getData('text');
        if (isValidYouTubeUrl(pasted)) {
            setUrl(pasted);
        }
    }, []);

    return (
        <div className="space-y-4">
            <div className="relative">
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-text-muted">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17" />
                        <path d="m10 15 5-3-5-3z" />
                    </svg>
                </div>
                <input
                    type="text"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    onPaste={handlePaste}
                    onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                    placeholder="Paste YouTube URL here..."
                    className="input-field pl-12 pr-4 py-4 text-base"
                    disabled={disabled}
                />
                {url && !isValid && (
                    <div className="absolute right-4 top-1/2 -translate-y-1/2">
                        <span className="text-xs text-error">Invalid URL</span>
                    </div>
                )}
                {isValid && (
                    <div className="absolute right-4 top-1/2 -translate-y-1/2">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M20 6 9 17l-5-5" />
                        </svg>
                    </div>
                )}
            </div>

            {/* Thumbnail Preview */}
            {videoId && (
                <div className="animate-slide-up glass-card p-3 flex items-center gap-4">
                    <img
                        src={getThumbnailUrl(videoId)}
                        alt="Video thumbnail"
                        className="w-32 h-20 object-cover rounded-lg"
                    />
                    <div className="flex-1 min-w-0">
                        <p className="text-sm text-text-secondary mb-1">Ready to dub</p>
                        <p className="text-xs text-text-muted truncate">{url}</p>
                    </div>
                </div>
            )}

            <button
                onClick={handleSubmit}
                disabled={!isValid || disabled}
                className="btn-primary w-full py-4 text-base font-semibold flex items-center justify-center gap-2"
            >
                {disabled ? (
                    <>
                        <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                        </svg>
                        Processing...
                    </>
                ) : (
                    <>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="m5 8 6 4-6 4V8Z" />
                            <path d="m13 8 6 4-6 4V8Z" />
                        </svg>
                        Start Dubbing
                    </>
                )}
            </button>
        </div>
    );
}

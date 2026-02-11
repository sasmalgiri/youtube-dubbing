'use client';

import { useState } from 'react';
import { cn } from '@/lib/utils';

interface VideoPlayerProps {
    originalUrl: string;
    dubbedUrl: string;
}

export default function VideoPlayer({ originalUrl, dubbedUrl }: VideoPlayerProps) {
    const [tab, setTab] = useState<'dubbed' | 'original'>('dubbed');

    return (
        <div className="glass-card overflow-hidden">
            {/* Tab header */}
            <div className="flex border-b border-border">
                <button
                    onClick={() => setTab('dubbed')}
                    className={cn(
                        'flex-1 py-3 text-sm font-medium transition-colors relative',
                        tab === 'dubbed' ? 'text-primary-light' : 'text-text-muted hover:text-text-secondary',
                    )}
                >
                    Dubbed (Hindi)
                    {tab === 'dubbed' && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                    )}
                </button>
                <button
                    onClick={() => setTab('original')}
                    className={cn(
                        'flex-1 py-3 text-sm font-medium transition-colors relative',
                        tab === 'original' ? 'text-primary-light' : 'text-text-muted hover:text-text-secondary',
                    )}
                >
                    Original
                    {tab === 'original' && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                    )}
                </button>
            </div>

            {/* Video */}
            <div className="bg-black aspect-video">
                <video
                    key={tab}
                    controls
                    className="w-full h-full"
                    src={tab === 'dubbed' ? dubbedUrl : originalUrl}
                />
            </div>
        </div>
    );
}

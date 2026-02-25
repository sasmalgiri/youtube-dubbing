'use client';

import Link from 'next/link';
import { extractYouTubeId, getThumbnailUrl } from '@/lib/utils';
import { resultVideoUrl } from '@/lib/api';
import type { BatchItem } from '@/hooks/useBatchManager';

const STATE_BADGES: Record<string, { label: string; className: string }> = {
    pending: { label: 'Queued', className: 'bg-text-muted/20 text-text-muted' },
    creating: { label: 'Starting', className: 'bg-yellow-500/20 text-yellow-400' },
    running: { label: 'Running', className: 'bg-primary/20 text-primary' },
    done: { label: 'Done', className: 'bg-green-500/20 text-green-400' },
    error: { label: 'Failed', className: 'bg-error/20 text-error' },
};

interface BatchItemCardProps {
    item: BatchItem;
    index: number;
}

export default function BatchItemCard({ item, index }: BatchItemCardProps) {
    const videoId = extractYouTubeId(item.url);
    const badge = STATE_BADGES[item.state] || STATE_BADGES.pending;

    return (
        <div className="glass-card p-4 flex items-center gap-4">
            {/* Index + Thumbnail */}
            <div className="relative flex-shrink-0">
                {videoId ? (
                    <img
                        src={getThumbnailUrl(videoId)}
                        alt="Thumbnail"
                        className="w-28 h-16 object-cover rounded-lg"
                    />
                ) : (
                    <div className="w-28 h-16 rounded-lg bg-card flex items-center justify-center">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-text-muted">
                            <path d="M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17" />
                            <path d="m10 15 5-3-5-3z" />
                        </svg>
                    </div>
                )}
                <span className="absolute -top-2 -left-2 w-6 h-6 rounded-full bg-primary text-white text-xs font-bold flex items-center justify-center">
                    {index + 1}
                </span>
            </div>

            {/* Info */}
            <div className="flex-1 min-w-0 space-y-1.5">
                <div className="flex items-center gap-2">
                    <p className="text-sm text-text-primary truncate flex-1">
                        {item.videoTitle || item.url}
                    </p>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badge.className}`}>
                        {badge.label}
                    </span>
                </div>

                {/* Progress bar for running state */}
                {(item.state === 'running' || item.state === 'creating') && (
                    <div className="space-y-1">
                        <div className="w-full h-1.5 bg-border rounded-full overflow-hidden">
                            <div
                                className="h-full bg-primary rounded-full transition-all duration-500"
                                style={{ width: `${item.progress}%` }}
                            />
                        </div>
                        <p className="text-xs text-text-muted truncate">
                            {item.step && <span className="text-text-secondary">{item.step}</span>}
                            {item.step && item.message && ' — '}
                            {item.message}
                        </p>
                    </div>
                )}

                {/* Error message */}
                {item.state === 'error' && item.error && (
                    <p className="text-xs text-error truncate">{item.error}</p>
                )}

                {/* Done - downloaded indicator */}
                {item.state === 'done' && (
                    <p className="text-xs text-green-400">
                        {item.downloaded ? 'Downloaded' : 'Complete — ready to download'}
                    </p>
                )}
            </div>

            {/* Actions */}
            <div className="flex-shrink-0 flex items-center gap-2">
                {item.state === 'done' && item.jobId && (
                    <>
                        <a
                            href={resultVideoUrl(item.jobId)}
                            download
                            className="p-2 rounded-lg hover:bg-primary/10 text-primary transition-colors"
                            title="Download video"
                        >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="7 10 12 15 17 10" />
                                <line x1="12" x2="12" y1="15" y2="3" />
                            </svg>
                        </a>
                        <Link
                            href={`/jobs/${item.jobId}`}
                            className="p-2 rounded-lg hover:bg-card text-text-secondary transition-colors"
                            title="View details"
                        >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                                <polyline points="15 3 21 3 21 9" />
                                <line x1="10" x2="21" y1="14" y2="3" />
                            </svg>
                        </Link>
                    </>
                )}
                {item.state === 'running' && item.jobId && (
                    <Link
                        href={`/jobs/${item.jobId}`}
                        className="p-2 rounded-lg hover:bg-card text-text-secondary transition-colors"
                        title="View details"
                    >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                            <polyline points="15 3 21 3 21 9" />
                            <line x1="10" x2="21" y1="14" y2="3" />
                        </svg>
                    </Link>
                )}
            </div>
        </div>
    );
}

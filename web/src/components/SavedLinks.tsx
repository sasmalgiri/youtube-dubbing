'use client';

import { useState, useEffect, useCallback } from 'react';
import { getLinks, deleteLink, type SavedLink } from '@/lib/api';
import { extractYouTubeId, getThumbnailUrl } from '@/lib/utils';

interface SavedLinksProps {
    onSelect: (url: string) => void;
}

export default function SavedLinks({ onSelect }: SavedLinksProps) {
    const [links, setLinks] = useState<SavedLink[]>([]);
    const [open, setOpen] = useState(false);

    const loadLinks = useCallback(() => {
        getLinks().then(setLinks).catch(() => {});
    }, []);

    useEffect(() => { loadLinks(); }, [loadLinks]);

    // Refresh links periodically to pick up auto-saved ones
    useEffect(() => {
        const interval = setInterval(loadLinks, 5000);
        return () => clearInterval(interval);
    }, [loadLinks]);

    const handleDelete = useCallback(async (id: string) => {
        const updated = await deleteLink(id);
        setLinks(updated);
    }, []);

    return (
        <div className="glass-card overflow-hidden">
            {/* Header */}
            <button
                onClick={() => setOpen(!open)}
                className="w-full flex items-center justify-between p-4 hover:bg-white/[0.02] transition-colors"
            >
                <div className="flex items-center gap-2">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                        <path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z" />
                    </svg>
                    <span className="text-sm font-medium text-text-primary">
                        Saved Links
                    </span>
                    {links.length > 0 && (
                        <span className="text-xs bg-primary/20 text-primary px-1.5 py-0.5 rounded-full">
                            {links.length}
                        </span>
                    )}
                </div>
                <svg
                    width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                    className={`text-text-muted transition-transform ${open ? 'rotate-180' : ''}`}
                >
                    <path d="m6 9 6 6 6-6" />
                </svg>
            </button>

            {open && (
                <div className="border-t border-border">
                    {/* Links list */}
                    {links.length === 0 ? (
                        <div className="p-4 text-center text-sm text-text-muted">
                            No saved links yet. Paste a URL above and save it here.
                        </div>
                    ) : (
                        <div className="max-h-80 overflow-y-auto">
                            {links.map((link) => {
                                const videoId = extractYouTubeId(link.url);
                                return (
                                    <div
                                        key={link.id}
                                        className="flex items-center gap-3 p-3 hover:bg-white/[0.03] transition-colors border-b border-border/30 last:border-0 group"
                                    >
                                        {/* Thumbnail */}
                                        {videoId && (
                                            <img
                                                src={getThumbnailUrl(videoId)}
                                                alt=""
                                                className="w-20 h-12 object-cover rounded-md flex-shrink-0"
                                            />
                                        )}

                                        {/* URL + title */}
                                        <div
                                            className="flex-1 min-w-0 cursor-pointer"
                                            onClick={() => onSelect(link.url)}
                                        >
                                            {link.title && (
                                                <p className="text-sm text-text-primary truncate">{link.title}</p>
                                            )}
                                            <p className="text-xs text-text-muted truncate">{link.url}</p>
                                        </div>

                                        {/* Actions */}
                                        <div className="flex items-center gap-1 flex-shrink-0">
                                            {/* Use button */}
                                            <button
                                                onClick={() => onSelect(link.url)}
                                                className="p-1.5 rounded-md hover:bg-primary/20 text-text-muted hover:text-primary transition-colors"
                                                title="Use this URL"
                                            >
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                    <path d="m5 8 6 4-6 4V8Z" />
                                                    <path d="m13 8 6 4-6 4V8Z" />
                                                </svg>
                                            </button>
                                            {/* Delete button */}
                                            <button
                                                onClick={() => handleDelete(link.id)}
                                                className="p-1.5 rounded-md hover:bg-error/20 text-text-muted hover:text-error transition-colors opacity-0 group-hover:opacity-100"
                                                title="Remove"
                                            >
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                    <path d="M18 6 6 18M6 6l12 12" />
                                                </svg>
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

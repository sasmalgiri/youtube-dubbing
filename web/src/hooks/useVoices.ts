'use client';

import { useEffect, useState } from 'react';
import { fetchVoices, type Voice } from '@/lib/api';

export function useVoices(lang: string = 'hi') {
    const [voices, setVoices] = useState<Voice[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        fetchVoices(lang)
            .then((v) => {
                if (!cancelled) {
                    setVoices(v.sort((a, b) => a.ShortName.localeCompare(b.ShortName)));
                    setLoading(false);
                }
            })
            .catch(() => {
                if (!cancelled) setLoading(false);
            });
        return () => { cancelled = true; };
    }, [lang]);

    return { voices, loading };
}

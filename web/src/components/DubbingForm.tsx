'use client';

import { useEffect, useMemo, useState } from 'react';
import { createJob, fetchVoices, getJob, resultUrl, type Voice, type JobStatus } from '@/lib/api';

export function DubbingForm() {
    const [file, setFile] = useState<File | null>(null);
    const [voices, setVoices] = useState<Voice[]>([]);
    const [voice, setVoice] = useState('en-US-AriaNeural');
    const [mixOriginal, setMixOriginal] = useState(true);
    const [originalVolume, setOriginalVolume] = useState(0.2);
    const [timeAlign, setTimeAlign] = useState(true);

    const [loadingVoices, setLoadingVoices] = useState(false);
    const [err, setErr] = useState<string | null>(null);

    const [job, setJob] = useState<JobStatus | null>(null);
    const [creating, setCreating] = useState(false);

    const canSubmit = useMemo(() => !!file && !creating, [file, creating]);

    useEffect(() => {
        let cancelled = false;
        async function load() {
            setLoadingVoices(true);
            setErr(null);
            try {
                const v = await fetchVoices();
                if (cancelled) return;
                setVoices(v);
                if (v.length && !v.find((x) => x.ShortName === voice)) {
                    setVoice(v[0].ShortName);
                }
            } catch (e: any) {
                if (cancelled) return;
                setErr(e?.message ?? String(e));
            } finally {
                if (!cancelled) setLoadingVoices(false);
            }
        }
        load();
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        if (!job?.id) return;
        if (job.state === 'done' || job.state === 'error') return;

        let cancelled = false;
        const t = setInterval(async () => {
            try {
                const next = await getJob(job.id);
                if (!cancelled) setJob(next);
            } catch (e: any) {
                if (!cancelled) setErr(e?.message ?? String(e));
            }
        }, 1500);

        return () => {
            cancelled = true;
            clearInterval(t);
        };
    }, [job]);

    async function onSubmit(e: React.FormEvent) {
        e.preventDefault();
        setErr(null);
        if (!file) return;

        const fd = new FormData();
        fd.append('file', file);
        fd.append('voice', voice);
        fd.append('mix_original', String(mixOriginal));
        fd.append('original_volume', String(originalVolume));
        fd.append('time_align', String(timeAlign));

        setCreating(true);
        try {
            const { id } = await createJob(fd);
            setJob({ id, state: 'queued', progress: 0 });
        } catch (e: any) {
            setErr(e?.message ?? String(e));
        } finally {
            setCreating(false);
        }
    }

    return (
        <div className="space-y-4">
            {err && (
                <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">
                    {err}
                </div>
            )}

            <form onSubmit={onSubmit} className="space-y-4">
                <div className="space-y-1">
                    <div className="label">Video file (mp4)</div>
                    <input
                        className="input"
                        type="file"
                        accept="video/mp4,video/x-m4v,video/*"
                        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                    />
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1">
                        <div className="label">Voice</div>
                        <select
                            className="input"
                            value={voice}
                            disabled={loadingVoices}
                            onChange={(e) => setVoice(e.target.value)}
                        >
                            {voices.length ? (
                                voices.map((v) => (
                                    <option key={v.ShortName} value={v.ShortName}>
                                        {v.ShortName} {v.Locale ? `(${v.Locale})` : ''}
                                    </option>
                                ))
                            ) : (
                                <option value={voice}>{voice}</option>
                            )}
                        </select>
                        <div className="text-xs text-white/50">
                            {loadingVoices
                                ? 'Loading voices from backend...'
                                : 'Voices come from backend /voices'}
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="flex items-center gap-2 text-sm text-white/80">
                            <input
                                type="checkbox"
                                checked={timeAlign}
                                onChange={(e) => setTimeAlign(e.target.checked)}
                            />
                            Time-align TTS
                        </label>
                        <label className="flex items-center gap-2 text-sm text-white/80">
                            <input
                                type="checkbox"
                                checked={mixOriginal}
                                onChange={(e) => setMixOriginal(e.target.checked)}
                            />
                            Mix original audio
                        </label>
                        <div className="space-y-1">
                            <div className="label">Original volume ({Math.round(originalVolume * 100)}%)</div>
                            <input
                                className="input"
                                type="number"
                                step="0.05"
                                min="0"
                                max="1"
                                value={originalVolume}
                                onChange={(e) => setOriginalVolume(Number(e.target.value))}
                                disabled={!mixOriginal}
                            />
                        </div>
                    </div>
                </div>

                <button className="btn-primary" type="submit" disabled={!canSubmit}>
                    {creating ? 'Starting…' : 'Start dubbing'}
                </button>
            </form>

            {job && (
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                    <div className="flex items-center justify-between gap-3">
                        <div>
                            <div className="text-sm font-medium">Job: {job.id}</div>
                            <div className="text-xs text-white/60">State: {job.state}</div>
                        </div>
                        {typeof job.progress === 'number' && (
                            <div className="text-xs text-white/70">{Math.round(job.progress * 100)}%</div>
                        )}
                    </div>

                    {job.message && <div className="mt-2 text-sm text-white/70">{job.message}</div>}

                    {job.state === 'done' && (
                        <div className="mt-3">
                            <a className="btn-secondary inline-block" href={resultUrl(job.id)}>
                                Download dubbed video
                            </a>
                        </div>
                    )}
                </div>
            )}

            <div className="text-xs text-white/50">
                Free deployment suggestion: Vercel for UI + Render free tier for backend.
            </div>
        </div>
    );
}

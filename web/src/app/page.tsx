'use client';

import { useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import URLInput from '@/components/URLInput';
import VoiceSelector from '@/components/VoiceSelector';
import SettingsPanel, { type DubbingSettings } from '@/components/SettingsPanel';
import JobCard from '@/components/JobCard';
import { createJob, getJobs, type JobStatus } from '@/lib/api';

export default function HomePage() {
    const router = useRouter();
    const [voice, setVoice] = useState('hi-IN-SwaraNeural');
    const [settings, setSettings] = useState<DubbingSettings>({
        asr_model: 'small',
        tts_rate: '+0%',
        mix_original: true,
        original_volume: 0.15,
        time_aligned: true,
    });
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [recentJobs, setRecentJobs] = useState<JobStatus[]>([]);

    const loadJobs = useCallback(() => {
        getJobs().then(setRecentJobs).catch(() => {});
    }, []);

    useEffect(() => {
        loadJobs();
    }, [loadJobs]);

    const handleSubmit = useCallback(async (url: string) => {
        setSubmitting(true);
        setError(null);
        try {
            const { id } = await createJob({
                url,
                voice,
                ...settings,
            });
            router.push(`/jobs/${id}`);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to start dubbing');
            setSubmitting(false);
        }
    }, [voice, settings, router]);

    return (
        <div className="min-h-screen">
            {/* Hero Section */}
            <div className="border-b border-border bg-gradient-to-b from-primary/[0.03] to-transparent">
                <div className="max-w-3xl mx-auto px-8 py-16">
                    <div className="text-center mb-10">
                        <h1 className="text-4xl font-bold text-text-primary mb-3">
                            Dub YouTube Videos
                            <span className="text-primary"> into Hindi</span>
                        </h1>
                        <p className="text-text-secondary text-lg">
                            Paste a YouTube URL and get an AI-dubbed video in seconds.
                            Powered by Whisper ASR and Edge-TTS.
                        </p>
                    </div>

                    {/* URL Input */}
                    <URLInput onSubmit={handleSubmit} disabled={submitting} />

                    {error && (
                        <div className="mt-4 p-3 rounded-xl bg-error/10 border border-error/20 text-error text-sm">
                            {error}
                        </div>
                    )}
                </div>
            </div>

            {/* Settings Section */}
            <div className="max-w-3xl mx-auto px-8 py-8 space-y-6">
                {/* Voice Selector */}
                <div className="glass-card p-5">
                    <VoiceSelector value={voice} onChange={setVoice} />
                </div>

                {/* Advanced Settings */}
                <SettingsPanel settings={settings} onChange={setSettings} />

                {/* Recent Jobs */}
                {recentJobs.length > 0 && (
                    <div>
                        <h2 className="text-lg font-semibold text-text-primary mb-4">Recent Jobs</h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {recentJobs.slice(0, 6).map((job) => (
                                <JobCard key={job.id} job={job} onDelete={loadJobs} />
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

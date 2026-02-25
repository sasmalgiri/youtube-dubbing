'use client';

import { useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import URLInput from '@/components/URLInput';
import LanguageSelector, { LANGUAGES } from '@/components/LanguageSelector';
import VoiceSelector from '@/components/VoiceSelector';
import SettingsPanel, { type DubbingSettings } from '@/components/SettingsPanel';
import JobCard from '@/components/JobCard';
import { createJob, getJobs, type JobStatus } from '@/lib/api';

const DEFAULT_VOICES: Record<string, string> = {
    hi: 'hi-IN-SwaraNeural',
    en: 'en-US-JennyNeural',
    es: 'es-ES-ElviraNeural',
    fr: 'fr-FR-DeniseNeural',
    de: 'de-DE-KatjaNeural',
    ja: 'ja-JP-NanamiNeural',
    ko: 'ko-KR-SunHiNeural',
    zh: 'zh-CN-XiaoxiaoNeural',
    pt: 'pt-BR-FranciscaNeural',
    ru: 'ru-RU-SvetlanaNeural',
    ar: 'ar-SA-ZariyahNeural',
    it: 'it-IT-ElsaNeural',
    tr: 'tr-TR-EmelNeural',
    bn: 'bn-IN-TanishaaNeural',
    ta: 'ta-IN-PallaviNeural',
    te: 'te-IN-ShrutiNeural',
    mr: 'mr-IN-AarohiNeural',
    gu: 'gu-IN-DhwaniNeural',
    kn: 'kn-IN-SapnaNeural',
    ml: 'ml-IN-SobhanaNeural',
    pa: 'pa-IN-GurpreetNeural',
    ur: 'ur-PK-UzmaNeural',
};

export default function HomePage() {
    const router = useRouter();
    const [sourceLanguage, setSourceLanguage] = useState('auto');
    const [targetLanguage, setTargetLanguage] = useState('hi');
    const [voice, setVoice] = useState('hi-IN-SwaraNeural');
    const [settings, setSettings] = useState<DubbingSettings>({
        tts_rate: '+0%',
        mix_original: false,
        original_volume: 0.10,
    });
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [recentJobs, setRecentJobs] = useState<JobStatus[]>([]);

    // When target language changes, pick a default voice for that language
    useEffect(() => {
        const defaultVoice = DEFAULT_VOICES[targetLanguage];
        if (defaultVoice) {
            setVoice(defaultVoice);
        }
    }, [targetLanguage]);

    const loadJobs = useCallback(() => {
        getJobs().then(setRecentJobs).catch(() => {});
    }, []);

    useEffect(() => {
        loadJobs();
    }, [loadJobs]);

    const targetName = LANGUAGES.find((l) => l.code === targetLanguage)?.name || targetLanguage;

    const handleSubmit = useCallback(async (url: string) => {
        setSubmitting(true);
        setError(null);
        try {
            const { id } = await createJob({
                url,
                source_language: sourceLanguage,
                target_language: targetLanguage,
                voice,
                ...settings,
            });
            router.push(`/jobs/${id}`);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to start dubbing');
            setSubmitting(false);
        }
    }, [sourceLanguage, targetLanguage, voice, settings, router]);

    return (
        <div className="min-h-screen">
            {/* Hero Section */}
            <div className="border-b border-border bg-gradient-to-b from-primary/[0.03] to-transparent">
                <div className="max-w-3xl mx-auto px-8 py-16">
                    <div className="text-center mb-10">
                        <h1 className="text-4xl font-bold text-text-primary mb-3">
                            Dub YouTube Videos
                            <span className="text-primary"> into {targetName}</span>
                        </h1>
                        <p className="text-text-secondary text-lg">
                            Paste a YouTube URL and get a dubbed video with a single AI voice.
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
                {/* Language Selector */}
                <div className="glass-card p-5">
                    <LanguageSelector
                        sourceLanguage={sourceLanguage}
                        targetLanguage={targetLanguage}
                        onSourceChange={setSourceLanguage}
                        onTargetChange={setTargetLanguage}
                    />
                </div>

                {/* Voice Selector */}
                <div className="glass-card p-5">
                    <VoiceSelector value={voice} onChange={setVoice} language={targetLanguage} />
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

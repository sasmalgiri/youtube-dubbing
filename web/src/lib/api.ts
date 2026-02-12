const API_BASE = '';  // Uses Next.js rewrite proxy to localhost:8000

// ── Types ───────────────────────────────────────────────────────────────────

export interface Voice {
    ShortName: string;
    Gender: string;
    Locale: string;
    FriendlyName?: string;
}

export interface JobCreateRequest {
    url: string;
    voice?: string;
    tts_rate?: string;
    mix_original?: boolean;
    original_volume?: number;
}

export interface JobStatus {
    id: string;
    state: 'queued' | 'running' | 'done' | 'error';
    current_step: string;
    step_progress: number;
    overall_progress: number;
    message: string;
    error?: string | null;
    source_url: string;
    video_title: string;
    created_at: number;
}

export interface TranscriptSegment {
    start: number;
    end: number;
    text: string;
    text_translated: string;
}

export interface Transcript {
    segments: TranscriptSegment[];
}

export interface SSEEvent {
    step?: string;
    progress?: number;
    overall?: number;
    message?: string;
    type?: string;
    state?: string;
    error?: string;
}

// ── API Functions ───────────────────────────────────────────────────────────

export async function fetchVoices(lang: string = 'hi'): Promise<Voice[]> {
    const res = await fetch(`${API_BASE}/api/voices?lang=${lang}`);
    if (!res.ok) throw new Error('Failed to fetch voices');
    return res.json();
}

export async function createJob(req: JobCreateRequest): Promise<{ id: string }> {
    const res = await fetch(`${API_BASE}/api/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(err.detail || 'Failed to create job');
    }
    return res.json();
}

export async function getJob(id: string): Promise<JobStatus> {
    const res = await fetch(`${API_BASE}/api/jobs/${id}`);
    if (!res.ok) throw new Error('Failed to fetch job');
    return res.json();
}

export async function getJobs(): Promise<JobStatus[]> {
    const res = await fetch(`${API_BASE}/api/jobs`);
    if (!res.ok) throw new Error('Failed to fetch jobs');
    return res.json();
}

export async function getTranscript(id: string): Promise<Transcript> {
    const res = await fetch(`${API_BASE}/api/jobs/${id}/transcript`);
    if (!res.ok) throw new Error('Failed to fetch transcript');
    return res.json();
}

export async function deleteJob(id: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/jobs/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete job');
}

export function resultVideoUrl(id: string): string {
    return `${API_BASE}/api/jobs/${id}/result`;
}

export function originalVideoUrl(id: string): string {
    return `${API_BASE}/api/jobs/${id}/original`;
}

export function resultSrtUrl(id: string): string {
    return `${API_BASE}/api/jobs/${id}/srt`;
}

// ── SSE Helper ──────────────────────────────────────────────────────────────

export function subscribeToJobEvents(
    jobId: string,
    onEvent: (event: SSEEvent) => void,
    onError?: (error: Error) => void,
): () => void {
    const eventSource = new EventSource(`${API_BASE}/api/jobs/${jobId}/events`);

    eventSource.onmessage = (e) => {
        try {
            const data: SSEEvent = JSON.parse(e.data);
            onEvent(data);
            if (data.type === 'complete') {
                eventSource.close();
            }
        } catch {
            // ignore parse errors
        }
    };

    eventSource.onerror = () => {
        eventSource.close();
        onError?.(new Error('SSE connection lost'));
    };

    return () => eventSource.close();
}

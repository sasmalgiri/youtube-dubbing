'use client';

import { cn } from '@/lib/utils';

const STEPS = [
    { key: 'download', label: 'Download', icon: 'download' },
    { key: 'extract', label: 'Extract', icon: 'audio' },
    { key: 'subtitles', label: 'Subtitles', icon: 'text' },
    { key: 'translate', label: 'Translate', icon: 'translate' },
    { key: 'synthesize', label: 'Synthesize', icon: 'voice' },
    { key: 'assemble', label: 'Assemble', icon: 'video' },
];

interface ProgressPipelineProps {
    currentStep: string;
    stepProgress: number;
    overallProgress: number;
    message: string;
    isComplete: boolean;
    isError: boolean;
}

function StepIcon({ icon, size = 16 }: { icon: string; size?: number }) {
    const props = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };

    switch (icon) {
        case 'download':
            return <svg {...props}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" x2="12" y1="15" y2="3" /></svg>;
        case 'audio':
            return <svg {...props}><path d="M2 10v3" /><path d="M6 6v11" /><path d="M10 3v18" /><path d="M14 8v7" /><path d="M18 5v13" /><path d="M22 10v3" /></svg>;
        case 'text':
            return <svg {...props}><path d="M17 6.1H3" /><path d="M21 12.1H3" /><path d="M15.1 18H3" /></svg>;
        case 'translate':
            return <svg {...props}><path d="m5 8 6 6" /><path d="m4 14 6-6 2-3" /><path d="M2 5h12" /><path d="M7 2h1" /><path d="m22 22-5-10-5 10" /><path d="M14 18h6" /></svg>;
        case 'voice':
            return <svg {...props}><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" x2="12" y1="19" y2="22" /></svg>;
        case 'video':
            return <svg {...props}><path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5" /><rect x="2" y="6" width="14" height="12" rx="2" /></svg>;
        default:
            return null;
    }
}

export default function ProgressPipeline({ currentStep, stepProgress, overallProgress, message, isComplete, isError }: ProgressPipelineProps) {
    const currentIndex = STEPS.findIndex(s => s.key === currentStep);

    return (
        <div className="glass-card p-6">
            {/* Overall progress bar */}
            <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-text-primary">
                        {isComplete ? 'Dubbing Complete!' : isError ? 'Error Occurred' : 'Dubbing in Progress...'}
                    </span>
                    <span className="text-sm text-text-muted">
                        {Math.round(overallProgress * 100)}%
                    </span>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                    <div
                        className={cn(
                            'h-full rounded-full transition-all duration-700 ease-out',
                            isError ? 'bg-error' : isComplete ? 'bg-success' : 'bg-gradient-to-r from-primary to-accent',
                        )}
                        style={{ width: `${overallProgress * 100}%` }}
                    />
                </div>
            </div>

            {/* Steps */}
            <div className="flex items-center justify-between">
                {STEPS.map((step, i) => {
                    const isDone = isComplete || i < currentIndex;
                    const isActive = !isComplete && !isError && i === currentIndex;
                    const isPending = !isComplete && i > currentIndex;
                    const hasError = isError && i === currentIndex;

                    return (
                        <div key={step.key} className="flex items-center flex-1">
                            {/* Step circle */}
                            <div className="flex flex-col items-center">
                                <div
                                    className={cn(
                                        'w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300',
                                        isDone && 'bg-success/20 border-success text-success',
                                        isActive && 'bg-primary/20 border-primary text-primary-light animate-pulse-glow',
                                        isPending && 'bg-white/[0.02] border-border border-dashed text-text-muted',
                                        hasError && 'bg-error/20 border-error text-error',
                                    )}
                                >
                                    {isDone ? (
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M20 6 9 17l-5-5" />
                                        </svg>
                                    ) : hasError ? (
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M18 6 6 18" /><path d="m6 6 12 12" />
                                        </svg>
                                    ) : (
                                        <StepIcon icon={step.icon} />
                                    )}
                                </div>
                                <span className={cn(
                                    'text-[10px] mt-2 font-medium',
                                    isDone && 'text-success',
                                    isActive && 'text-primary-light',
                                    isPending && 'text-text-muted',
                                    hasError && 'text-error',
                                )}>
                                    {step.label}
                                </span>

                                {/* Step progress bar */}
                                {isActive && (
                                    <div className="w-10 h-0.5 bg-white/5 rounded-full mt-1 overflow-hidden">
                                        <div
                                            className="h-full bg-primary rounded-full transition-all duration-500"
                                            style={{ width: `${stepProgress * 100}%` }}
                                        />
                                    </div>
                                )}
                            </div>

                            {/* Connector line */}
                            {i < STEPS.length - 1 && (
                                <div className={cn(
                                    'flex-1 h-0.5 mx-2 rounded-full transition-colors duration-300',
                                    i < currentIndex || isComplete ? 'bg-success/50' : 'bg-border',
                                )} />
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Current message */}
            {message && (
                <div className={cn(
                    'mt-4 text-center text-sm',
                    isError ? 'text-error' : 'text-text-secondary',
                )}>
                    {message}
                </div>
            )}
        </div>
    );
}

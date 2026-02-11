"""
VoiceDub Backend API
====================
FastAPI server with SSE progress, YouTube URL input, and translation support.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from pipeline import Pipeline, PipelineConfig, list_voices

# ── Types ────────────────────────────────────────────────────────────────────

JobState = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    id: str
    state: JobState = "queued"
    current_step: str = ""
    step_progress: float = 0.0
    overall_progress: float = 0.0
    message: str = "Queued"
    error: Optional[str] = None
    result_path: Optional[Path] = None
    source_url: str = ""
    video_title: str = ""
    segments: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    events: List[Dict] = field(default_factory=list)


class JobCreateRequest(BaseModel):
    url: str
    voice: str = "hi-IN-SwaraNeural"
    asr_model: str = "small"
    tts_rate: str = "-5%"
    mix_original: bool = False
    original_volume: float = 0.10
    time_aligned: bool = True


# ── Step weights for overall progress ────────────────────────────────────────

STEP_ORDER = ["download", "extract", "transcribe", "translate", "synthesize", "assemble"]
STEP_WEIGHTS = {
    "download": 0.15,
    "extract": 0.05,
    "transcribe": 0.25,
    "translate": 0.15,
    "synthesize": 0.30,
    "assemble": 0.10,
}

# ── Storage ──────────────────────────────────────────────────────────────────

JOBS: Dict[str, Job] = {}
BASE_DIR = Path(__file__).resolve().parent
WORK_ROOT = BASE_DIR / "work"
OUTPUTS = WORK_ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="VoiceDub API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output files for video playback
OUTPUTS.mkdir(parents=True, exist_ok=True)
app.mount("/static/jobs", StaticFiles(directory=str(OUTPUTS)), name="job-files")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _calc_overall(step: str, step_progress: float) -> float:
    """Calculate overall progress from current step and its progress."""
    overall = 0.0
    for s in STEP_ORDER:
        if s == step:
            overall += STEP_WEIGHTS.get(s, 0) * step_progress
            break
        overall += STEP_WEIGHTS.get(s, 0)
    return min(overall, 1.0)


def _make_progress_callback(job: Job):
    """Create a progress callback that updates the job and appends events."""
    def callback(step: str, progress: float, message: str):
        job.current_step = step
        job.step_progress = progress
        job.overall_progress = _calc_overall(step, progress)
        job.message = message
        job.events.append({
            "step": step,
            "progress": round(progress, 3),
            "overall": round(job.overall_progress, 3),
            "message": message,
        })
    return callback


def _run_job(job: Job, req: JobCreateRequest):
    """Run the dubbing pipeline in a background thread."""
    try:
        job.state = "running"
        job.message = "Starting..."

        job_dir = OUTPUTS / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        work_dir = job_dir / "work"
        work_dir.mkdir(exist_ok=True)

        out_path = job_dir / "dubbed.mp4"

        cfg = PipelineConfig(
            source=req.url,
            work_dir=work_dir,
            output_path=out_path,
            target_language="hi",
            asr_model=req.asr_model,
            tts_voice=req.voice,
            tts_rate=req.tts_rate,
            mix_original=req.mix_original,
            original_volume=req.original_volume,
            time_aligned=req.time_aligned,
        )

        pipeline = Pipeline(cfg, on_progress=_make_progress_callback(job))
        pipeline.run()

        if not out_path.exists():
            raise RuntimeError("Pipeline finished but output file not found")

        job.result_path = out_path
        job.video_title = pipeline.video_title or "Untitled"
        job.segments = pipeline.segments
        job.overall_progress = 1.0
        job.state = "done"
        job.message = "Complete"
        job.events.append({"type": "complete", "state": "done"})

    except Exception as e:
        job.state = "error"
        job.error = str(e)
        job.message = f"Error: {e}"
        job.events.append({"type": "complete", "state": "error", "error": str(e)})


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/voices")
async def voices(lang: str = "hi") -> Any:
    return await list_voices(lang)


@app.get("/api/jobs")
def list_jobs():
    """List all jobs, newest first."""
    jobs = sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)
    return [
        {
            "id": j.id,
            "state": j.state,
            "current_step": j.current_step,
            "step_progress": j.step_progress,
            "overall_progress": j.overall_progress,
            "message": j.message,
            "error": j.error,
            "source_url": j.source_url,
            "video_title": j.video_title,
            "created_at": j.created_at,
        }
        for j in jobs
    ]


@app.post("/api/jobs")
def create_job(req: JobCreateRequest):
    """Create a new dubbing job from a YouTube URL."""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, source_url=url)
    JOBS[job_id] = job

    t = threading.Thread(target=_run_job, args=(job, req), daemon=True)
    t.start()

    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "state": job.state,
        "current_step": job.current_step,
        "step_progress": round(job.step_progress, 3),
        "overall_progress": round(job.overall_progress, 3),
        "message": job.message,
        "error": job.error,
        "source_url": job.source_url,
        "video_title": job.video_title,
        "created_at": job.created_at,
    }


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """SSE endpoint for real-time progress updates."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_index = 0
        while True:
            if last_index < len(job.events):
                for event in job.events[last_index:]:
                    yield {"data": json.dumps(event)}
                    if event.get("type") == "complete":
                        return
                last_index = len(job.events)
            if job.state in ("done", "error"):
                if last_index >= len(job.events):
                    yield {"data": json.dumps({"type": "complete", "state": job.state})}
                    return
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.get("/api/jobs/{job_id}/transcript")
def get_transcript(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"segments": job.segments}


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state != "done" or not job.result_path:
        raise HTTPException(status_code=409, detail="Job not complete")

    return FileResponse(
        path=str(job.result_path),
        media_type="video/mp4",
        filename=f"dubbed_{job_id}.mp4",
    )


@app.get("/api/jobs/{job_id}/srt")
def get_srt(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    srt_path = OUTPUTS / job_id / "subtitles_hi.srt"
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="Subtitles not found")

    return FileResponse(
        path=str(srt_path),
        media_type="text/plain",
        filename=f"subtitles_{job_id}.srt",
    )


@app.get("/api/jobs/{job_id}/original")
def get_original_video(job_id: str):
    """Serve the original downloaded video for preview."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    work_dir = OUTPUTS / job_id / "work"
    source = work_dir / "source.mp4"
    if not source.exists():
        sources = list(work_dir.glob("source.*"))
        source = sources[0] if sources else None

    if not source or not source.exists():
        raise HTTPException(status_code=404, detail="Original video not found")

    return FileResponse(path=str(source), media_type="video/mp4")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = OUTPUTS / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    return {"status": "deleted"}

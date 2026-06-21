"""
FastAPI wrapper for the FL runner — Track 4
============================================
Exposes the federated learning pipeline as an HTTP API so the frontend
dashboard can trigger and stream real training runs.

Endpoints
---------
POST   /runs                  Start a new FL run (background thread)
GET    /runs/{run_id}         Run status (queued / running / done / error)
GET    /runs/{run_id}/stream  Server-Sent Events: per-round JSON events
GET    /runs/{run_id}/result  Full fl_results.json equivalent (once done)
GET    /health                API liveness check

Start the server from the repo root:
    uvicorn server.api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Adaptive FL for Bank Fraud Detection — API",
    description="FastAPI wrapper around fl_runner.py for the dashboard frontend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory run state ───────────────────────────────────────────────────────
# run_id → {
#   "status":  "queued" | "running" | "done" | "error",
#   "config":  dict,
#   "queue":   queue.Queue,   ← background thread pushes events here
#   "result":  dict | None,   ← set on completion
#   "error":   str | None,
# }
RUNS: dict[str, dict[str, Any]] = {}
RUNS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    method: str = Field("custom", description="Aggregation method")
    rounds: int = Field(10, ge=1, le=50, description="Number of FL rounds")
    local_epochs: int = Field(3, ge=1, le=20)
    mu: float = Field(0.01, ge=0.0, le=1.0, description="FedProx mu")
    clip_norm: float = Field(1.0, gt=0, description="DP L2 clip norm")
    noise_multiplier: float = Field(1.1, ge=0, description="DP noise multiplier")
    device: str = Field("cpu", description="PyTorch device")


class RunStatus(BaseModel):
    run_id: str
    status: str
    method: str
    rounds: int


class StartRunResponse(BaseModel):
    run_id: str


# ---------------------------------------------------------------------------
# Background training thread
# ---------------------------------------------------------------------------


def _training_thread(run_id: str, config: RunConfig) -> None:
    """
    Runs fl_runner.run_fl_gen() in a background thread.
    Pushes each yielded event dict onto the run's queue.
    Marks the run done/error on finish.
    """
    try:
        with RUNS_LOCK:
            RUNS[run_id]["status"] = "running"

        # Build an argparse.Namespace that run_fl_gen() expects
        args = argparse.Namespace(
            method=config.method,
            rounds=config.rounds,
            local_epochs=config.local_epochs,
            mu=config.mu,
            clip_norm=config.clip_norm,
            noise_multiplier=config.noise_multiplier,
            device=config.device,
        )

        from server.fl_runner import run_fl_gen

        q: queue.Queue = RUNS[run_id]["queue"]

        for event in run_fl_gen(args):
            q.put(event)
            if event.get("event") == "complete":
                with RUNS_LOCK:
                    RUNS[run_id]["result"] = event.get("output")
                    RUNS[run_id]["status"] = "done"

    except Exception as exc:
        with RUNS_LOCK:
            RUNS[run_id]["status"] = "error"
            RUNS[run_id]["error"] = str(exc)
        RUNS[run_id]["queue"].put({"event": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    """Liveness check for the frontend status pill."""
    return {"status": "ok"}


@app.post("/runs", response_model=StartRunResponse, status_code=201)
async def start_run(config: RunConfig) -> StartRunResponse:
    """Start a new FL run. Returns a run_id immediately."""
    run_id = str(uuid.uuid4())
    with RUNS_LOCK:
        RUNS[run_id] = {
            "status": "queued",
            "config": config.model_dump(),
            "queue": queue.Queue(),
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_training_thread,
        args=(run_id, config),
        daemon=True,
        name=f"fl-run-{run_id[:8]}",
    )
    thread.start()

    return StartRunResponse(run_id=run_id)


@app.get("/runs/{run_id}", response_model=RunStatus)
async def get_run(run_id: str) -> RunStatus:
    """Return the current status of a run."""
    with RUNS_LOCK:
        run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    cfg = run["config"]
    return RunStatus(
        run_id=run_id,
        status=run["status"],
        method=cfg.get("method", "?"),
        rounds=cfg.get("rounds", 0),
    )


@app.get("/runs/{run_id}/result")
async def get_run_result(run_id: str) -> dict:
    """Return the full result dict once a run has completed."""
    with RUNS_LOCK:
        run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    if run["status"] == "error":
        raise HTTPException(status_code=500, detail=run.get("error", "Unknown error"))
    if run["status"] != "done":
        raise HTTPException(status_code=202, detail="Run not yet complete")
    return run["result"] or {}


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    """
    Server-Sent Events stream for a run.
    Each event is a JSON object. The stream closes after the 'complete' event.
    """
    with RUNS_LOCK:
        run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    q: queue.Queue = run["queue"]

    async def event_generator():
        while True:
            # Poll the queue without blocking the event loop
            try:
                event = q.get_nowait()
            except queue.Empty:
                # If run is done/error and queue is empty, stop
                with RUNS_LOCK:
                    current_status = RUNS[run_id]["status"]
                if current_status in ("done", "error") and q.empty():
                    break
                await asyncio.sleep(0.2)
                continue

            data = json.dumps(event, default=str)
            yield f"data: {data}\n\n"

            if event.get("event") in ("complete", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

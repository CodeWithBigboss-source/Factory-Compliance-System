"""
api.py
------
FastAPI backend tying Modules 1-4 together into a callable pipeline, and
serving data to the Streamlit dashboard (Module 5).

Run with (from project root, not from inside src/):
    uvicorn src.api:app --reload --port 8000
"""

import os
import sys
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection.model_loader import SafeUnsafeClassifier
from detection.detect import run_detection_on_clip, load_policy_rules
from severity.severity_matrix import build_severity_lookup, assign_severity_batch
from escalation.escalation import escalate_batch
from reports.report_writer import (
    write_event_reports_batch, fetch_all_events, export_to_csv, export_to_json
)

app = FastAPI(title="Factory Compliance & Alert Escalation System")

MODEL_PATH = "models/model.pth"
POLICY_PATH = "data/policy_rules.json"
UPLOAD_DIR = "data/uploaded_clips"

_classifier = None
_policy_rules = None


def get_classifier():
    global _classifier
    if _classifier is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"model.pth not found at {MODEL_PATH}. Place your trained model there.",
            )
        _classifier = SafeUnsafeClassifier(model_path=MODEL_PATH)
    return _classifier


def get_policy_rules():
    global _policy_rules
    if _policy_rules is None:
        if not os.path.exists(POLICY_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"policy_rules.json not found at {POLICY_PATH}. Run policy_parser.py first.",
            )
        _policy_rules = load_policy_rules(POLICY_PATH)
    return _policy_rules


class ProcessClipRequest(BaseModel):
    clip_path: str
    sample_interval_sec: float = 1.0
    confidence_threshold: float = 0.55


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_loaded": _classifier is not None,
        "policy_loaded": _policy_rules is not None,
    }


@app.post("/process_clip")
def process_clip(request: ProcessClipRequest):
    """Runs the full pipeline on a clip already present at clip_path on disk."""
    if not os.path.exists(request.clip_path):
        raise HTTPException(status_code=404, detail=f"Clip not found: {request.clip_path}")

    classifier = get_classifier()
    policy_rules = get_policy_rules()

    detections, clip_meta = run_detection_on_clip(
        request.clip_path, classifier, policy_rules,
        every_n_seconds=request.sample_interval_sec,
        confidence_threshold=request.confidence_threshold,
    )

    detections = assign_severity_batch(detections, policy_rules)
    escalation_result = escalate_batch(detections)
    write_event_reports_batch(escalation_result["events"])

    return {
        "clip_path": request.clip_path,
        "clip_metadata": clip_meta,
        "total_events": len(escalation_result["events"]),
        "alerts_triggered": len(escalation_result["alerts"]),
        "logs_only": len(escalation_result["logs"]),
        "events": escalation_result["events"],
    }


@app.post("/upload_and_process")
async def upload_and_process(
    file: UploadFile = File(...),
    sample_interval_sec: float = 1.0,
    confidence_threshold: float = 0.55,
):
    """Accepts an uploaded clip, saves it, then runs the same pipeline as /process_clip."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    request = ProcessClipRequest(
        clip_path=dest_path,
        sample_interval_sec=sample_interval_sec,
        confidence_threshold=confidence_threshold,
    )
    return process_clip(request)


@app.get("/events")
def get_events(
    severity: Optional[str] = Query(None, description="Filter by severity tier: Low/Medium/High/Critical"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    """Historical log query, used by Dashboard View C."""
    rows = fetch_all_events(
        severity_filter=severity, domain_filter=domain,
        date_from=date_from, date_to=date_to,
    )
    return {"count": len(rows), "events": rows}


@app.get("/events/recent")
def get_recent_events(limit: int = 20):
    """Used by Dashboard View B — Alert Timeline Stream."""
    rows = fetch_all_events()
    return {"count": min(limit, len(rows)), "events": rows[:limit]}


@app.get("/alerts/active")
def get_active_alerts(limit: int = 10):
    """Used by Dashboard View A's real-time alert banner for High/Critical events."""
    rows = fetch_all_events(severity_filter=None)
    alerts = [r for r in rows if r.get("escalation_action") == "ALERT"][:limit]
    return {"count": len(alerts), "alerts": alerts}


@app.get("/export")
def export_log(format: str = Query("csv", pattern="^(csv|json)$")):
    """Used by Dashboard View C's export button."""
    rows = fetch_all_events()
    if not rows:
        raise HTTPException(status_code=404, detail="No events to export yet.")

    if format == "csv":
        path = export_to_csv(rows)
        return FileResponse(path, filename="compliance_export.csv", media_type="text/csv")
    else:
        path = export_to_json(rows)
        return FileResponse(path, filename="compliance_export.json", media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
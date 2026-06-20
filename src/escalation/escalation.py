"""
escalation.py
--------------
Module 3 — Escalation Pipeline.

Mandatory routing logic per the assignment spec:
    Low / Medium severity   -> LOG only (no real-time alert)
    High / Critical severity -> ALERT (real-time notification + log)

Also handles the assignment's own hint scenario directly:
    "What happens if you detect multiple simultaneous violations of
     different severities in the same clip? How should the pipeline
     handle that?"

Design decision (documented, not hidden): each violation is routed
INDEPENDENTLY — a Low-severity walkway violation and a Critical
intervention violation detected in the same clip at different timestamps
each get their own escalation decision and their own report record.
They are NOT merged or deduplicated by clip, only deduplicated when they
represent the same physical event observed across consecutive sampled
frames (see deduplicate_consecutive_detections below) — this prevents
one continuous 10-second violation from generating 10 separate alerts
just because we sample at 1-second intervals.
"""

from datetime import datetime, timezone


ALERT_TIERS = {"High", "Critical"}
LOG_ONLY_TIERS = {"Low", "Medium"}

# How close together (in seconds, within the same clip) two detections of the
# SAME class_id must be to be considered the same continuous event rather
# than two separate violations. Tune based on your frame sampling interval.
DEDUP_WINDOW_SECONDS = 3.0


def deduplicate_consecutive_detections(detection_records, dedup_window=DEDUP_WINDOW_SECONDS):
    """
    Collapses consecutive same-class detections within `dedup_window` seconds
    of each other (within the same clip) into a single event, keeping the
    highest-confidence record and recording the event's duration span.
    """
    if not detection_records:
        return []

    sorted_records = sorted(detection_records, key=lambda r: (r["clip_id"], r["class_id"], r["timestamp_sec"]))
    deduped = []
    current_group = [sorted_records[0]]

    for rec in sorted_records[1:]:
        prev = current_group[-1]
        same_event = (
            rec["clip_id"] == prev["clip_id"]
            and rec["class_id"] == prev["class_id"]
            and (rec["timestamp_sec"] - prev["timestamp_sec"]) <= dedup_window
        )
        if same_event:
            current_group.append(rec)
        else:
            deduped.append(_merge_group(current_group))
            current_group = [rec]

    deduped.append(_merge_group(current_group))
    return deduped


def _merge_group(group):
    """Merges a group of consecutive same-event detections into one record."""
    best = max(group, key=lambda r: r.get("class_router_confidence", 0))
    merged = dict(best)
    merged["event_start_sec"] = min(r["timestamp_sec"] for r in group)
    merged["event_end_sec"] = max(r["timestamp_sec"] for r in group)
    merged["event_duration_sec"] = round(merged["event_end_sec"] - merged["event_start_sec"], 2)
    merged["sample_count_in_event"] = len(group)
    return merged


def escalate(detection_record):
    """
    Applies the mandatory routing rule to a single (already severity-tagged)
    detection record. Must be called AFTER severity_matrix.assign_severity().
    """
    tier = detection_record.get("severity_tier", "Medium")

    if tier in ALERT_TIERS:
        action = "ALERT"
        alert_payload = {
            "alert_id": detection_record["detection_id"],
            "severity": tier,
            "message": (
                f"[{tier.upper()} ALERT] {detection_record.get('rule_breached')} "
                f"detected in {detection_record.get('zone')} at "
                f"{detection_record.get('timestamp_sec')}s in clip "
                f"'{detection_record.get('clip_id')}'."
            ),
            "triggered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        }
    else:
        action = "LOG"
        alert_payload = None

    detection_record["escalation_action"] = action
    detection_record["alert_payload"] = alert_payload
    return detection_record


def escalate_batch(detection_records, deduplicate=True):
    """
    Full Module 3 entry point: optionally deduplicates consecutive same-event
    detections, then applies escalation routing to each resulting event.

    Returns:
        dict: {
            "events": List[dict]  (all events, with escalation_action attached),
            "alerts": List[dict]  (only the ALERT-tier events, alert_payload populated),
            "logs": List[dict]    (only the LOG-tier events),
        }
    """
    records = deduplicate_consecutive_detections(detection_records) if deduplicate else detection_records
    events = [escalate(r) for r in records]

    alerts = [e for e in events if e["escalation_action"] == "ALERT"]
    logs = [e for e in events if e["escalation_action"] == "LOG"]

    return {"events": events, "alerts": alerts, "logs": logs}


if __name__ == "__main__":
    # Smoke test with synthetic multi-severity records to demonstrate the
    # "multiple simultaneous violations of different severities" handling.
    sample_records = [
        {"detection_id": "a1", "clip_id": "clip01", "class_id": 0, "timestamp_sec": 2.0,
         "rule_breached": "Safe Walkway Violation", "zone": "Pedestrian Walkway Zone",
         "severity_tier": "Medium", "class_router_confidence": 0.7},
        {"detection_id": "a2", "clip_id": "clip01", "class_id": 0, "timestamp_sec": 3.0,
         "rule_breached": "Safe Walkway Violation", "zone": "Pedestrian Walkway Zone",
         "severity_tier": "Medium", "class_router_confidence": 0.75},
        {"detection_id": "a3", "clip_id": "clip01", "class_id": 1, "timestamp_sec": 5.0,
         "rule_breached": "Unauthorized Intervention", "zone": "Equipment Intervention Zone",
         "severity_tier": "Critical", "class_router_confidence": 0.9},
    ]

    result = escalate_batch(sample_records)
    print(f"Total events after dedup: {len(result['events'])}")
    print(f"Alerts: {len(result['alerts'])}, Logs: {len(result['logs'])}\n")
    for e in result["events"]:
        print(f"  [{e['severity_tier']}] {e['rule_breached']} -> {e['escalation_action']} "
              f"(duration={e.get('event_duration_sec')}s, samples={e.get('sample_count_in_event')})")
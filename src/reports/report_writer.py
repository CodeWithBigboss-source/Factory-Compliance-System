"""
report_writer.py
-----------------
Module 4 — Automated Report Generation.

Persists every violation event (post severity + escalation) as a structured
compliance report. Reports are written automatically, never manually.

Storage: SQLite (outputs/compliance_log.db) as the primary structured store
("Structured database records: rows in a relational or document store
accessible for export" — directly satisfies the Module 4 output format
spec), with CSV/JSON export functions for the dashboard's export button
(Module 5, View C).

Required report fields (per assignment Module 4 spec, at minimum):
    - clip identifier
    - timestamp of occurrence
    - rule breached
    - description of observed behavior
    - zone / location
    - severity tier
    - escalation action taken
    - detection confidence
    - report generation timestamp
"""

import sqlite3
import json
import csv
import os
from datetime import datetime, timezone

DB_PATH = "outputs/compliance_log.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS compliance_events (
    report_id TEXT PRIMARY KEY,
    clip_id TEXT NOT NULL,
    timestamp_sec REAL,
    event_start_sec REAL,
    event_end_sec REAL,
    rule_breached TEXT,
    class_id INTEGER,
    domain TEXT,
    description TEXT,
    zone TEXT,
    severity_tier TEXT,
    severity_signal TEXT,
    escalation_action TEXT,
    gate_model_confidence REAL,
    class_router_confidence REAL,
    class_router_certain INTEGER,
    policy_source_section TEXT,
    detected_at TEXT,
    report_generated_at TEXT
);
"""


def init_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def write_event_report(event_record, db_path=DB_PATH):
    """
    Writes a single post-escalation event record to the SQLite store.
    Idempotent on report_id (uses detection_id as primary key via INSERT OR
    REPLACE) so re-running detection on the same clip won't create duplicate
    rows.
    """
    conn = init_db(db_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    row = (
        event_record.get("detection_id"),
        event_record.get("clip_id"),
        event_record.get("timestamp_sec"),
        event_record.get("event_start_sec", event_record.get("timestamp_sec")),
        event_record.get("event_end_sec", event_record.get("timestamp_sec")),
        event_record.get("rule_breached"),
        event_record.get("class_id"),
        event_record.get("domain"),
        event_record.get("description"),
        event_record.get("zone"),
        event_record.get("severity_tier"),
        event_record.get("severity_signal"),
        event_record.get("escalation_action"),
        event_record.get("gate_model_confidence"),
        event_record.get("class_router_confidence"),
        int(bool(event_record.get("class_router_certain", False))),
        event_record.get("policy_source_section"),
        event_record.get("detected_at"),
        now,
    )

    conn.execute("""
        INSERT OR REPLACE INTO compliance_events (
            report_id, clip_id, timestamp_sec, event_start_sec, event_end_sec,
            rule_breached, class_id, domain, description, zone,
            severity_tier, severity_signal, escalation_action,
            gate_model_confidence, class_router_confidence, class_router_certain,
            policy_source_section, detected_at, report_generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, row)
    conn.commit()
    conn.close()


def write_event_reports_batch(event_records, db_path=DB_PATH):
    for record in event_records:
        write_event_report(record, db_path=db_path)
    print(f"✅ Wrote {len(event_records)} report(s) to {db_path}")


def fetch_all_events(db_path=DB_PATH, severity_filter=None, domain_filter=None,
                      date_from=None, date_to=None):
    """
    Fetches events from the historical log with optional filters, used by
    Module 5 (View C — Historical Log & Export).
    """
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM compliance_events WHERE 1=1"
    params = []

    if severity_filter:
        query += " AND severity_tier = ?"
        params.append(severity_filter)
    if domain_filter:
        query += " AND domain = ?"
        params.append(domain_filter)
    if date_from:
        query += " AND report_generated_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND report_generated_at <= ?"
        params.append(date_to)

    query += " ORDER BY report_generated_at DESC"

    cursor = conn.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def export_to_csv(rows, out_path="outputs/compliance_export.csv"):
    if not rows:
        print("No rows to export.")
        return None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Exported {len(rows)} rows to {out_path}")
    return out_path


def export_to_json(rows, out_path="outputs/compliance_export.json"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"✅ Exported {len(rows)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    sample_event = {
        "detection_id": "test-001",
        "clip_id": "test_clip",
        "timestamp_sec": 4.0,
        "rule_breached": "Opened Panel Cover",
        "class_id": 2,
        "domain": "Electrical Safety",
        "description": "Test record",
        "zone": "Electrical Panel Zone",
        "severity_tier": "Medium",
        "severity_signal": "Medium",
        "escalation_action": "LOG",
        "gate_model_confidence": 0.81,
        "class_router_confidence": 0.65,
        "class_router_certain": True,
        "policy_source_section": "Section 5",
        "detected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    }
    write_event_report(sample_event)
    rows = fetch_all_events()
    print(f"Rows in DB: {len(rows)}")
    for r in rows[:3]:
        print(" ", r["report_id"], r["rule_breached"], r["severity_tier"])
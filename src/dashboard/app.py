"""
app.py
------
Module 5 — Operations Dashboard (Streamlit).

Implements all three required views:
    View A — Live Feed Monitor: process a clip, show status + alert banner
    View B — Alert Timeline Stream: chronological event stream
    View C — Historical Log & Export: filterable table + export button

This dashboard talks to the FastAPI backend (src/api.py) over HTTP rather
than importing the pipeline modules directly, so the dashboard and backend
can be deployed/run independently.

Run with (from project root, with the API already running separately):
    uvicorn src.api:app --reload --port 8000
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import requests
import pandas as pd
import os
import glob
from datetime import datetime

API_BASE_URL = os.environ.get("COMPLIANCE_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Factory Compliance & Alert Escalation System",
    page_icon="🏭",
    layout="wide",
)

SEVERITY_COLORS = {
    "Low": "#4CAF50",
    "Medium": "#FFC107",
    "High": "#FF5722",
    "Critical": "#D32F2F",
}


def api_get(endpoint, params=None):
    try:
        resp = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}. Is the FastAPI backend running at {API_BASE_URL}?")
        return None


def api_post(endpoint, json_data):
    try:
        resp = requests.post(f"{API_BASE_URL}{endpoint}", json=json_data, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


def severity_badge(tier):
    color = SEVERITY_COLORS.get(tier, "#999999")
    return f'<span style="background-color:{color}; color:white; padding:3px 10px; border-radius:10px; font-weight:600;">{tier}</span>'


st.title("🏭 Factory Compliance & Alert Escalation System")
st.caption("Real-time monitoring grounded in the facility's OHS Compliance Policy Manual")

tab_a, tab_b, tab_c = st.tabs(["📹 Live Feed Monitor", "📡 Alert Timeline Stream", "📊 Historical Log & Export"])


# ===========================================================================
# VIEW A — Live Feed Monitor
# ===========================================================================
with tab_a:
    st.subheader("Live / Simulated Feed Monitor")
    st.write(
        "Select a clip from `data/` to process through the full pipeline "
        "(Detection → Severity → Escalation → Report). High/Critical events "
        "trigger a real-time alert banner below."
    )

    clip_files = sorted(glob.glob("data/*.mp4")) + sorted(glob.glob("data/uploaded_clips/*.mp4"))

    col1, col2 = st.columns([2, 1])
    with col1:
        if clip_files:
            selected_clip = st.selectbox("Select a clip to process", clip_files)
        else:
            selected_clip = None
            st.warning("No .mp4 clips found in data/. Place your sample clips there.")

    with col2:
        sample_interval = st.slider("Sampling interval (seconds)", 0.5, 3.0, 1.0, 0.5)
        confidence_threshold = st.slider("Detection confidence threshold", 0.3, 0.95, 0.55, 0.05)

    if selected_clip and st.button("▶ Process Clip", type="primary"):
        with st.spinner(f"Running detection pipeline on {selected_clip} ..."):
            result = api_post("/process_clip", {
                "clip_path": selected_clip,
                "sample_interval_sec": sample_interval,
                "confidence_threshold": confidence_threshold,
            })

        if result:
            st.success(
                f"Processed clip — {result['total_events']} event(s) found "
                f"({result['alerts_triggered']} alert(s), {result['logs_only']} log-only)."
            )

            meta = result["clip_metadata"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Resolution", meta.get("resolution", "N/A"))
            m2.metric("Duration (s)", meta.get("duration_sec", "N/A"))
            m3.metric("FPS", meta.get("fps", "N/A"))
            m4.metric("Total Events", result["total_events"])

            critical_events = [e for e in result["events"] if e["severity_tier"] in ("High", "Critical")]
            if critical_events:
                for e in critical_events:
                    st.markdown(
                        f"""
                        <div style="background-color:{SEVERITY_COLORS[e['severity_tier']]}; 
                                    color:white; padding:14px; border-radius:8px; margin-bottom:8px;
                                    font-weight:600; font-size:16px;">
                            🚨 {e['severity_tier'].upper()} ALERT — {e['rule_breached']} detected at 
                            {e.get('timestamp_sec', e.get('event_start_sec'))}s in {e['zone']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("✅ No High/Critical alerts triggered for this clip.")

            st.markdown("#### Detected events")
            for e in result["events"]:
                badge = severity_badge(e["severity_tier"])
                st.markdown(
                    f"{badge} &nbsp; **{e['rule_breached']}** — {e['zone']} "
                    f"(t={e.get('timestamp_sec', e.get('event_start_sec'))}s, "
                    f"action=`{e['escalation_action']}`)",
                    unsafe_allow_html=True,
                )


# ===========================================================================
# VIEW B — Alert Timeline Stream
# ===========================================================================
with tab_b:
    st.subheader("Real-time Chronological Event Stream")

    if st.button("🔄 Refresh stream"):
        st.rerun()

    recent = api_get("/events/recent", params={"limit": 30})

    if recent and recent["events"]:
        for e in recent["events"]:
            badge = severity_badge(e["severity_tier"])
            icon = "🚨" if e["escalation_action"] == "ALERT" else "📝"
            st.markdown(
                f"{icon} {badge} &nbsp; `{e['report_generated_at']}` — "
                f"**{e['rule_breached']}** in clip `{e['clip_id']}` ({e['zone']})",
                unsafe_allow_html=True,
            )
            st.divider()
    else:
        st.info("No events recorded yet. Process a clip in the Live Feed Monitor tab first.")


# ===========================================================================
# VIEW C — Historical Log & Export
# ===========================================================================
with tab_c:
    st.subheader("Full Historical Compliance Log")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        severity_filter = st.selectbox("Filter by severity", ["All", "Low", "Medium", "High", "Critical"])
    with filter_col2:
        domain_filter = st.selectbox(
            "Filter by domain",
            ["All", "Pedestrian Movement", "Equipment Interaction", "Electrical Safety", "Forklift Load Management"],
        )
    with filter_col3:
        date_from = st.date_input("From date", value=None)

    params = {}
    if severity_filter != "All":
        params["severity"] = severity_filter
    if domain_filter != "All":
        params["domain"] = domain_filter
    if date_from:
        params["date_from"] = date_from.isoformat()

    events_data = api_get("/events", params=params)

    if events_data and events_data["events"]:
        df = pd.DataFrame(events_data["events"])
        display_cols = [
            "report_generated_at", "clip_id", "rule_breached", "domain", "zone",
            "severity_tier", "escalation_action", "class_router_confidence",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, height=400)

        st.markdown(f"**Total matching records: {len(df)}**")

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Export as CSV", data=csv_data,
                file_name=f"compliance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        with export_col2:
            json_data = df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                "⬇ Export as JSON", data=json_data,
                file_name=f"compliance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
    else:
        st.info("No historical records match the selected filters.")
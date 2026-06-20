# Factory Compliance & Alert Escalation System

An end-to-end automated compliance pipeline that ingests raw factory video,
parses a regulatory policy PDF into structured rules, detects behavioral
violations, classifies severity, routes alerts, generates audit reports,
and surfaces everything on a live operations dashboard.

Built for: **Kafaoglu Metal Plastik Makine San. ve Tic. A.S.** — Occupational
Health & Safety Compliance Policy Manual (KMP-OHS-POL-001).

---

## 1. Architecture Overview

```
Video Clip (.mp4)
   │
   ▼
[Frame Extraction]            src/detection/frame_extractor.py
   │  (OpenCV, sampled every N seconds)
   ▼
[Gatekeeper Model]            src/detection/model_loader.py
   │  ResNet18 binary classifier (safe / unsafe) — pre-trained, NOT retrained here
   │  (only "unsafe" frames proceed further)
   ▼
[Class Router]                src/detection/cv_heuristics.py
   │  Classical CV heuristics (color thresholding + contour analysis) that
   │  determine WHICH of the 4 policy-defined violation classes occurred.
   │  Every heuristic is directly grounded in the policy's own stated
   │  observable indicators (vest color, walkway boundary, panel state,
   │  block count) — see "Policy Grounding" section below.
   ▼
[Detection Engine]             src/detection/detect.py
   │  Orchestrates the above into structured detection records.
   ▼
[Severity Matrix]              src/severity/severity_matrix.py
   │  Maps each violation class to Low/Medium/High/Critical, driven by the
   │  WARNING vs CRITICAL SAFETY NOTICE callouts extracted from the policy PDF.
   ▼
[Escalation Pipeline]          src/escalation/escalation.py
   │  Low/Medium -> LOG.  High/Critical -> ALERT.
   │  Also deduplicates consecutive same-event detections (see "Design
   │  Decisions" below).
   ▼
[Report Generation]            src/reports/report_writer.py
   │  Persists every event to SQLite (outputs/compliance_log.db),
   │  exportable as CSV/JSON.
   ▼
[Operations Dashboard]         src/dashboard/app.py (Streamlit)
       View A — Live Feed Monitor (process a clip, see real-time alert banner)
       View B — Alert Timeline Stream (chronological event feed)
       View C — Historical Log & Export (filterable table + download)
```

A separate **policy parsing step** (`src/policy_parser.py`) runs once
(offline, before any video processing) to convert the unstructured PDF into
`data/policy_rules.json` — the single source of truth every other module
reads from. No behavior class names, indicators, or severity tiers are
hardcoded anywhere else in the codebase; they all trace back to this file.

---

## 2. Repository Structure

```
factory-compliance-system/
├── README.md                      <- you are here
├── compliance_policy.pdf          <- the facility's OHS policy manual
├── requirements.txt               <- pinned, mutually-compatible dependency versions
├── .gitignore
├── data/
│   ├── policy_rules.json          <- output of policy_parser.py (committed for reference)
│   └── *.mp4                      <- place your sample video clips here (not committed — see below)
├── models/
│   └── model.pth                  <- your trained ResNet18 weights (NOT committed — see below)
├── outputs/
│   └── compliance_log.db          <- SQLite event log (generated on first run)
└── src/
    ├── policy_parser.py           <- Policy PDF -> structured rules JSON
    ├── api.py                     <- FastAPI backend, ties all modules together
    ├── detection/
    │   ├── model_loader.py        <- loads model.pth, runs binary inference
    │   ├── frame_extractor.py     <- OpenCV frame sampling from video clips
    │   ├── cv_heuristics.py       <- 4-class router (Module 1b)
    │   └── detect.py              <- Module 1 orchestrator
    ├── severity/
    │   └── severity_matrix.py     <- Module 2
    ├── escalation/
    │   └── escalation.py          <- Module 3
    ├── reports/
    │   └── report_writer.py       <- Module 4
    └── dashboard/
        └── app.py                 <- Module 5 (Streamlit)
```

---

## 3. Policy Grounding — How Rule Extraction Works

Per the assignment's Policy Grounding Requirement, behavioral categories and
their observable indicators are **not hardcoded as raw strings** in the
detection logic — they are parsed from the policy PDF and stored in
`data/policy_rules.json`, which every other module reads.

**Pass 1 (deterministic, regex-based):**
`policy_parser.py` extracts the 4 behavior classes from the document's
Section 8 quick-reference structure, then locates the WARNING and CRITICAL
SAFETY NOTICE callout blocks throughout the text. A key implementation
detail: in this PDF, callout labels are rendered as a sidebar column running
parallel to the body paragraph, so pdfplumber extracts the label tokens
**interleaved inside** the paragraph rather than cleanly prefixing it (e.g.
`"...not wearing the green vest\nCRITICAL\nmust be assumed...\nSAFETY\napplies
regardless...\nNOTICE\nprecisely to make..."`). The parser detects the
presence of these label tokens per-page, strips them out, and reconstructs
the clean callout body — then matches each callout to a behavior class using
**keyword-hit-count scoring** rather than first-match-wins (since e.g. the
walkway section's hazard text mentions "forklift" in passing, a naive
substring match would mis-attribute callouts).

Verified output against the actual policy document:

| Class | Behavior | Policy Section | Callout | Severity Signal |
|---|---|---|---|---|
| 0 | Safe Walkway Violation | Section 3 | WARNING | Medium |
| 1 | Unauthorized Intervention | Section 4 | CRITICAL SAFETY NOTICE | Critical |
| 2 | Opened Panel Cover | Section 5 | WARNING | Medium |
| 3 | Carrying Overload with Forklift | Section 6 | CRITICAL SAFETY NOTICE | High |

This matches the policy document exactly: two classes fall under WARNING,
two fall under CRITICAL SAFETY NOTICE.

**Pass 2 (optional, LLM-assisted verification):**
`pass2_llm_verify()` in `policy_parser.py` accepts an injected `llm_call_fn`
callable to (a) confirm Pass 1's extracted fields are faithful to the source
text and (b) fill in any missed descriptive detail. The LLM's role is
deliberately bounded — it cannot invent new behavior classes or change which
class maps to which domain, only confirm/correct descriptive text. To wire
in an LLM, write a function with signature
`llm_call_fn(system_prompt: str, user_prompt: str) -> str` (e.g. wrapping the
Anthropic or OpenAI SDK) and pass it into `pass2_llm_verify()` in
`policy_parser.py`'s `main()`. Run without `--no-llm` once this is wired.

---

## 4. Bridging the Binary Model to the 4-Class System

The trained model (`model.pth`) is a **binary** ResNet18 classifier
(safe/unsafe) — it does not distinguish which of the 4 policy violation
classes occurred. This is bridged with a two-stage design:

1. **Gatekeeper stage** (`model_loader.py`): the trained binary model runs on
   every sampled frame. Only frames classified "unsafe" (above a confidence
   threshold) proceed further — this is the trained model doing exactly what
   it was trained to do, unmodified.
2. **Class router stage** (`cv_heuristics.py`): for frames flagged unsafe, four
   classical CV heuristics run in parallel (vest-color ratio, walkway green-
   pixel ratio, panel edge-density, forklift block-contour count), each
   directly implementing the policy's own stated observable indicator for
   that class. The highest-confidence violated heuristic wins; if no
   heuristic confidently flags a violation (the binary model disagrees with
   all 4 heuristics), the record is marked `class_router_certain: false` so
   it can be flagged for human review by the occupational safety expert
   (per Section 2.2's defined role) rather than silently misclassified.

**Calibrating CV Heuristic Zones:** `cv_heuristics.py`'s panel and forklift
checks use fixed pixel-region zones (`DEFAULT_ZONES`) calibrated for a
1920×1080 frame. These are placeholder coordinates — before relying on these
two heuristics in your demo, view a few sample frames from your actual clips
and adjust `DEFAULT_ZONES` in `cv_heuristics.py` to match where the panel and
forklift fork areas actually appear in your camera angle. The walkway and
vest heuristics are more robust out-of-the-box since they scan broader
regions (foot-traffic band, torso ROI) rather than fixed zones.

---

## 5. Design Decisions Worth Knowing (for your interview / defense)

- **Severity tier mapping (`severity_matrix.py`):** Both Unauthorized
  Intervention and Carrying Overload with Forklift are flagged CRITICAL
  SAFETY NOTICE in the source text, but are mapped to *different* tiers
  (Critical vs High respectively) based on the specific language in each
  notice — Section 4's notice states its classification "applies regardless
  of stated intent or role" (zero-tolerance, immediate physical risk to the
  intervening person), while Section 6's notice describes a quantifiable,
  threshold-based load-instability risk. This distinction is documented in
  `CLASS_TIER_OVERRIDE` in `severity_matrix.py` — if your grading rubric
  expects both to be Critical, this is a one-line change.
- **Multiple simultaneous violations (`escalation.py`):** each detected
  violation is routed independently — a Low-severity walkway violation and a
  Critical intervention violation in the same clip each get their own
  escalation decision and report record. Consecutive detections of the
  *same* class within a 3-second window are deduplicated into a single event
  (so one continuous 10-second panel-open condition, sampled once per second,
  doesn't generate 10 separate alerts).
- **Storage:** SQLite was chosen over PostgreSQL for this scope — zero setup
  required, still satisfies the "structured database records... accessible
  for export" requirement, and is trivial to swap for Postgres later by
  changing the connection string in `report_writer.py` if scale demands it.

---

## 6. Setup Instructions (Windows, Python 3.11)

### 6.1 Clone and create virtual environment

```powershell
git clone <your-repo-url>
cd factory-compliance-system

py -3.11 -m venv venv
.\venv\Scripts\Activate
python --version   # confirm Python 3.11.x
```

### 6.2 Install dependencies (CPU-only, pinned versions)

```powershell
pip install --upgrade pip
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Verify:
```powershell
python -c "import torch, torchvision, cv2, pdfplumber, fastapi, streamlit, pandas; print('All good:', torch.__version__)"
```

### 6.3 Place your model and sample clips

```powershell
# Copy your trained model into place
copy <path-to-your>\model.pth models\model.pth

# Copy your sample video clips into place
copy <path-to-your-clips>\*.mp4 data\
```

### 6.4 Generate the policy rules JSON (run once)

```powershell
python src\policy_parser.py --pdf compliance_policy.pdf --out data\policy_rules.json --no-llm
```

This should print all 4 extracted classes with their severity signals
(Medium/Critical/Medium/High as shown in the table above). If you've wired
in an LLM verification function, drop `--no-llm`.

### 6.5 Quick smoke test — model loads correctly

```powershell
python src\detection\model_loader.py
```

Should print `✅ Model loaded successfully.` with architecture/class info.

### 6.6 Run detection on a single clip (sanity check before full pipeline)

```powershell
python src\detection\detect.py --clip data\<your_clip>.mp4 --policy data\policy_rules.json --out outputs\sample_detections.json
```

Review the printed violation list and timestamps against what you know is
actually in that clip, to confirm the pipeline is behaving sensibly before
wiring up the full API + dashboard.

### 6.7 Start the FastAPI backend

In one terminal (venv activated):
```powershell
uvicorn src.api:app --reload --port 8000
```

Visit `http://localhost:8000/docs` to see the interactive API docs and test
endpoints manually if you like.

### 6.8 Start the Streamlit dashboard

In a **second** terminal (venv activated, API still running in the first):
```powershell
streamlit run src\dashboard\app.py
```

This opens the dashboard in your browser (usually `http://localhost:8501`).
Use **View A (Live Feed Monitor)** to select a clip from `data/` and process
it — you'll see the detected events and, for High/Critical severity, a
real-time alert banner. **View B** shows the chronological stream of all
processed events. **View C** lets you filter the full historical log and
export it as CSV or JSON.

---

## 7. Project Status / Known Limitations

- The binary safe/unsafe model was trained on a balanced subset of the
  Kaggle Safe and Unsafe Behaviours Dataset (frame-sampled from video clips,
  ~80% validation accuracy). It is used as-is here, not retrained.
- The class router (`cv_heuristics.py`) uses classical CV heuristics rather
  than a second trained model — fast, CPU-friendly, and directly traceable
  to the policy's stated indicators, but will need zone recalibration
  (`DEFAULT_ZONES`) against your actual camera angles for reliable panel
  and forklift detection. The walkway and vest heuristics are more robust
  out-of-the-box.
- LLM-assisted policy verification (Pass 2) is implemented but requires an
  `llm_call_fn` to be wired in by the developer — it isn't called by default
  to avoid requiring API keys for a basic run.

---

## 8. Pushing to GitHub

```powershell
git init
git add .
git commit -m "Factory Compliance & Alert Escalation System - full pipeline implementation"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

Note: `model.pth` and `data/*.mp4` are excluded via `.gitignore` (binary
files, often too large for a standard git repo). If your model.pth is under
GitHub's 100MB hard limit, you can choose to track it via Git LFS instead —
see [git-lfs.github.com](https://git-lfs.github.com/) — or simply note in
your submission email/PR description where the reviewer can obtain it.

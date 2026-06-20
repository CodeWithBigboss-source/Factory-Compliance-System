"""
cv_heuristics.py
-----------------
Module 1b — Class Router.

Once the binary model (model_loader.py) flags a frame as "unsafe", this
module determines WHICH of the 4 policy-defined violation classes it most
likely is, using simple, explainable computer-vision heuristics that map
DIRECTLY onto the observable indicators stated in the policy document
(data/policy_rules.json), satisfying the assignment's "Policy Grounding
Requirement":

    "The observable indicators used to classify a behavior as safe or
     unsafe (e.g., vest color, block count, panel state, walkway position)
     must be traceable to the relevant policy section."

Each heuristic function below is annotated with the exact policy section
and indicator it implements. These are intentionally simple/classical CV
(color-space thresholding + contour analysis) rather than a second trained
model — defensible, fast, CPU-friendly, and transparent about its own
confidence and failure modes (documented per-function).

NOTE: Real deployment would calibrate these HSV ranges and zone boundaries
per-camera using actual footage. The thresholds below are reasonable
starting points; the README documents how to recalibrate them.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Class 0 — Safe Walkway Violation (Section 3)
# Indicator: person positioned outside the green-marked walkway boundary.
# ---------------------------------------------------------------------------

def check_walkway_violation(frame_bgr, walkway_mask=None):
    """
    Heuristic: the Designated Safe Walkway is marked with green floor paint
    (Section 3.2). We detect green floor-paint pixels to approximate the
    walkway boundary, then check whether the bottom-center region of the
    frame (proxy for "where the person's feet are") falls outside that
    green-masked area.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = walkway_mask if walkway_mask is not None else cv2.inRange(hsv, lower_green, upper_green)

    region = green_mask[int(h * 0.66):h, int(w * 0.25):int(w * 0.75)]
    green_ratio = float(np.count_nonzero(region)) / max(1, region.size)

    violated = green_ratio < 0.05
    confidence = round(1.0 - green_ratio, 3) if violated else round(green_ratio, 3)

    return {
        "violated": violated,
        "confidence": confidence,
        "reason": f"green_walkway_pixel_ratio={green_ratio:.3f} in foot-traffic region",
        "policy_section": "Section 3.2 / 3.3.2",
    }


# ---------------------------------------------------------------------------
# Class 1 — Unauthorized Intervention (Section 4)
# Indicator: person interacting with equipment NOT wearing the green vest.
# ---------------------------------------------------------------------------

def check_unauthorized_intervention(frame_bgr, person_bbox=None):
    """
    Heuristic: Section 4.2 states the green safety vest is the primary
    observable indicator of authorization. We measure green-pixel ratio
    within the person's bounding box (torso region) — high green ratio
    implies the green authorization vest is present (Authorized); low
    green ratio implies a red-black or other vest (Unauthorized).
    """
    h, w = frame_bgr.shape[:2]
    if person_bbox:
        x1, y1, x2, y2 = person_bbox
        roi = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    else:
        roi = frame_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.35):int(w * 0.65)]

    if roi.size == 0:
        return {"violated": False, "confidence": 0.0, "reason": "empty ROI", "policy_section": "Section 4.2"}

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)
    green_ratio = float(np.count_nonzero(green_mask)) / max(1, green_mask.size)

    lower_red1 = np.array([0, 60, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 60, 40])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_roi, lower_red1, upper_red1) | cv2.inRange(hsv_roi, lower_red2, upper_red2)
    red_ratio = float(np.count_nonzero(red_mask)) / max(1, red_mask.size)

    has_green_vest = green_ratio > 0.08
    violated = not has_green_vest
    confidence = round(max(red_ratio, 1.0 - green_ratio), 3)

    return {
        "violated": violated,
        "confidence": confidence,
        "reason": f"green_vest_ratio={green_ratio:.3f}, red_black_vest_ratio={red_ratio:.3f}",
        "policy_section": "Section 4.2 / 4.3.2",
    }


# ---------------------------------------------------------------------------
# Class 2 — Opened Panel Cover (Section 5)
# Indicator: panel cover observed in the open position.
# ---------------------------------------------------------------------------

def check_panel_cover_open(frame_bgr, panel_zone_bbox):
    """
    Heuristic: an open panel cover changes the geometry/contour profile of
    the panel zone (the cover swings outward, creating a darker recessed
    area + a new rectangular edge not present when closed). We use edge
    density within the known panel zone as a proxy: a closed flat cover has
    a low edge count; an open cover (with visible interior wiring/recess)
    has a high edge count.

    Args:
        panel_zone_bbox: (x1, y1, x2, y2) — the fixed pixel region where
                          this camera sees the electrical panel. Must be
                          calibrated once per camera since panels are at
                          fixed locations (Section 7.1: fixed IP cameras).
    """
    x1, y1, x2, y2 = panel_zone_bbox
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {"violated": False, "confidence": 0.0, "reason": "empty panel ROI", "policy_section": "Section 5.2"}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / max(1, edges.size)

    violated = edge_density > 0.12
    confidence = round(min(edge_density / 0.25, 1.0), 3)

    return {
        "violated": violated,
        "confidence": confidence,
        "reason": f"panel_zone_edge_density={edge_density:.3f}",
        "policy_section": "Section 5.2.2",
    }


# ---------------------------------------------------------------------------
# Class 3 — Carrying Overload with Forklift (Section 6)
# Indicator: 3 or more standardized blocks visible on forklift forks.
# ---------------------------------------------------------------------------

def check_forklift_overload(frame_bgr, fork_zone_bbox, block_threshold=3):
    """
    Heuristic: Section 6.2 defines the threshold precisely (2 blocks safe,
    3+ unsafe). We count distinct rectangular contours within the known
    fork-carrying zone as a proxy for block count.
    """
    x1, y1, x2, y2 = fork_zone_bbox
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "violated": False, "confidence": 0.0, "reason": "empty fork zone ROI",
            "estimated_block_count": 0, "policy_section": "Section 6.2",
        }

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.01 * roi.shape[0] * roi.shape[1]
    significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    estimated_count = len(significant_contours)

    violated = estimated_count >= block_threshold
    confidence = round(min(abs(estimated_count - block_threshold + 0.5) / 3.0 + 0.4, 1.0), 3)

    return {
        "violated": violated,
        "confidence": confidence,
        "reason": f"estimated_block_count={estimated_count} (threshold={block_threshold})",
        "estimated_block_count": estimated_count,
        "policy_section": "Section 6.2 / 6.3.2",
    }


# ---------------------------------------------------------------------------
# Router — runs all 4 heuristics and returns the most likely class
# ---------------------------------------------------------------------------

DEFAULT_ZONES = {
    # Placeholder pixel zones for a 1920x1080 frame. CALIBRATE THESE against
    # your actual sample clips before relying on panel/forklift heuristics —
    # see README "Calibrating CV Heuristic Zones".
    "panel_zone_bbox": (1500, 200, 1900, 600),
    "fork_zone_bbox": (200, 600, 800, 1000),
}


def route_violation_class(frame_pil, zones=None):
    """
    Runs all 4 class heuristics against a single frame (already flagged
    "unsafe" by the binary model) and returns the highest-confidence match.
    """
    zones = zones or DEFAULT_ZONES
    frame_rgb = np.array(frame_pil.convert("RGB"))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    results = {
        0: check_walkway_violation(frame_bgr),
        1: check_unauthorized_intervention(frame_bgr),
        2: check_panel_cover_open(frame_bgr, zones["panel_zone_bbox"]),
        3: check_forklift_overload(frame_bgr, zones["fork_zone_bbox"]),
    }

    class_names = {
        0: "Safe Walkway Violation",
        1: "Unauthorized Intervention",
        2: "Opened Panel Cover",
        3: "Carrying Overload with Forklift",
    }

    # Among classes flagged as violated, pick the highest-confidence one.
    # If none are flagged "violated" by heuristics (binary model said unsafe
    # but heuristics disagree on which class), fall back to highest raw
    # confidence score — this surfaces an "uncertain" case the report/
    # dashboard should flag for human (occupational safety expert) review.
    violated_classes = {cid: r for cid, r in results.items() if r["violated"]}

    if violated_classes:
        best_id = max(violated_classes, key=lambda cid: violated_classes[cid]["confidence"])
        is_certain = True
    else:
        best_id = max(results, key=lambda cid: results[cid]["confidence"])
        is_certain = False

    return {
        "class_id": best_id,
        "class_name": class_names[best_id],
        "confidence": results[best_id]["confidence"],
        "certain": is_certain,
        "all_results": results,
    }


if __name__ == "__main__":
    print("cv_heuristics.py — run via detect.py for end-to-end testing.")
    print("Default calibration zones:", DEFAULT_ZONES)
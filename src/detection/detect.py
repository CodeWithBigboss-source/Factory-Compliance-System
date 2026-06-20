"""
detect.py
---------
Module 1 — Detection Engine (orchestrator).

Pipeline per clip:
    1. Extract frames at a fixed sample interval.
    2. Run each frame through the binary safe/unsafe model (model_loader.py).
    3. For frames flagged "unsafe", run the 4-class CV-heuristic router
       (cv_heuristics.py) to determine WHICH policy-defined class it is.
    4. Emit a structured detection record per violation, containing at
       minimum: clip identifier, timestamp, rule breached, description,
       zone — exactly as required by the assignment's Module 1 spec.

Usage:
    python src/detection/detect.py --clip data/sample_clip.mp4 --policy data/policy_rules.json
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.model_loader import SafeUnsafeClassifier
from detection.frame_extractor import extract_frames, get_clip_metadata
from detection.cv_heuristics import route_violation_class, DEFAULT_ZONES


def load_policy_rules(policy_path):
    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"policy_rules.json not found at {policy_path}. Run policy_parser.py first."
        )
    with open(policy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_rule_lookup(policy_rules):
    """class_id -> rule dict, for attaching policy-grounded fields to detections."""
    return {r["class_id"]: r for r in policy_rules["rules"]}


def zone_name_for_class(class_id):
    """Maps a class_id to a human-readable zone label for the detection record."""
    return {
        0: "Pedestrian Walkway Zone",
        1: "Equipment Intervention Zone",
        2: "Electrical Panel Zone",
        3: "Forklift Load Zone",
    }.get(class_id, "Unknown Zone")


def run_detection_on_clip(clip_path, classifier, policy_rules, every_n_seconds=1.0,
                           confidence_threshold=0.55, zones=None):
    """
    Runs the full Module 1 pipeline on a single video clip.

    Returns:
        List[dict]: one structured detection record per violation found.
    """
    clip_id = os.path.splitext(os.path.basename(clip_path))[0]
    clip_meta = get_clip_metadata(clip_path)
    rule_lookup = build_rule_lookup(policy_rules)

    frames = extract_frames(clip_path, every_n_seconds=every_n_seconds)
    records = []

    for frame_data in frames:
        pil_frame = frame_data["frame"]
        timestamp = frame_data["timestamp_sec"]

        gate_result = classifier.predict_frame(pil_frame)

        if gate_result["label"] != "unsafe":
            continue
        if gate_result["confidence"] < confidence_threshold:
            continue

        route_result = route_violation_class(pil_frame, zones=zones)
        class_id = route_result["class_id"]
        rule = rule_lookup.get(class_id, {})

        record = {
            "detection_id": str(uuid.uuid4()),
            "clip_id": clip_id,
            "clip_path": clip_path,
            "timestamp_sec": timestamp,
            "frame_index": frame_data["frame_index"],
            "rule_breached": rule.get("unsafe_behavior", route_result["class_name"]),
            "class_id": class_id,
            "domain": rule.get("domain", "Unknown"),
            "description": (
                f"{rule.get('unsafe_behavior', route_result['class_name'])} detected. "
                f"Observable indicator: {rule.get('observable_indicator', 'N/A')}"
            ),
            "zone": zone_name_for_class(class_id),
            "gate_model_confidence": gate_result["confidence"],
            "class_router_confidence": route_result["confidence"],
            "class_router_certain": route_result["certain"],
            "policy_source_section": rule.get("source_section", "N/A"),
            "detected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        }
        records.append(record)

    return records, clip_meta


def main():
    parser = argparse.ArgumentParser(description="Run Module 1 detection on a video clip.")
    parser.add_argument("--clip", required=True, help="Path to video clip")
    parser.add_argument("--policy", default="data/policy_rules.json", help="Path to policy_rules.json")
    parser.add_argument("--model", default="models/model.pth", help="Path to trained model.pth")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between sampled frames")
    parser.add_argument("--out", default=None, help="Optional path to write detection records JSON")
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    classifier = SafeUnsafeClassifier(model_path=args.model)

    print(f"Loading policy rules from {args.policy} ...")
    policy_rules = load_policy_rules(args.policy)

    print(f"Running detection on {args.clip} ...")
    records, clip_meta = run_detection_on_clip(args.clip, classifier, policy_rules, every_n_seconds=args.interval)

    print(f"\nClip metadata: {clip_meta}")
    print(f"Found {len(records)} violation(s).\n")
    for r in records:
        print(f"  [{r['timestamp_sec']}s] {r['rule_breached']} (zone={r['zone']}, "
              f"gate_conf={r['gate_model_confidence']:.2f}, router_conf={r['class_router_confidence']:.2f})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"\n✅ Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
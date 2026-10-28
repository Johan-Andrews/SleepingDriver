"""Detector evaluation suite: Confusion matrix, Precision, Recall, F1, Latency, and Ablation analysis."""

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple
import numpy as np

from src.config import DetectionConfig, FusionConfig
from src.decision.fusion import DrowsinessFuser, DrowsinessLevel
from src.vision.features import DriverFeatures

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    mode: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    mean_latency_s: float
    false_alarm_rate_per_hour: float


def generate_synthetic_evaluation_events(num_episodes: int = 20) -> List[Tuple[str, List[DriverFeatures]]]:
    """Generate labeled synthetic episodes representing different driver behaviors."""
    episodes = []
    fps = 30.0
    dt = 1.0 / fps

    behaviors = ["alert_normal", "blinking", "closed_4s", "yawning", "nodding"]

    for b in behaviors:
        for ep_i in range(num_episodes):
            frames = []
            duration_s = 5.0
            n_frames = int(duration_s * fps)

            for f_i in range(n_frames):
                t = f_i * dt
                is_closed = False
                closure_s = 0.0
                mar = 0.15
                pitch_rel = 0.0
                perclos = 0.05
                is_nodding = False
                is_yawning = False

                if b == "alert_normal":
                    pass
                elif b == "blinking":
                    # Periodic 0.2s blinks every 2 seconds
                    if (t % 2.0) < 0.2:
                        is_closed = True
                        closure_s = t % 2.0
                elif b == "closed_4s":
                    # Eyes closed starting at t=0.5s for 4.0s
                    if 0.5 <= t <= 4.5:
                        is_closed = True
                        closure_s = t - 0.5
                        perclos = min(0.8, closure_s / 4.0)
                elif b == "yawning":
                    if 1.0 <= t <= 3.5:
                        mar = 0.75
                        is_yawning = True
                elif b == "nodding":
                    if 1.0 <= t <= 4.0:
                        pitch_rel = -28.0
                        is_nodding = True

                feat = DriverFeatures(
                    timestamp=t,
                    face_detected=True,
                    ear_raw=0.10 if is_closed else 0.32,
                    ear_smooth=0.10 if is_closed else 0.32,
                    is_eye_closed=is_closed,
                    eye_closed_duration_s=closure_s,
                    total_blinks=int(t / 2.0) if b == "blinking" else 1,
                    blink_rate_per_min=15.0,
                    avg_blink_duration_s=0.2,
                    perclos=perclos,
                    mar_raw=mar,
                    mar_smooth=mar,
                    is_yawning=is_yawning,
                    yawn_count=1 if is_yawning else 0,
                    yawns_per_min=2.0 if is_yawning else 0.0,
                    pitch=pitch_rel,
                    yaw=0.0,
                    roll=0.0,
                    pitch_rel=pitch_rel,
                    is_head_nodding=is_nodding,
                    is_distracted=False,
                    no_face_duration_s=0.0,
                )
                frames.append(feat)
            episodes.append((b, frames))

    return episodes


def evaluate_detector_configuration(
    episodes: List[Tuple[str, List[DriverFeatures]]],
    mode: str = "fused",
) -> EvaluationMetrics:
    """Evaluate detector performance on the dataset under a given mode (fused, eyes_only, head_only)."""
    tp = fp = tn = fn = 0
    latencies: List[float] = []
    normal_duration_s = 0.0

    det_cfg = DetectionConfig(eye_closure_trigger_s=3.0)
    fusion_cfg = FusionConfig()

    if mode == "eyes_only":
        fusion_cfg.weights.head_nod = 0.0
        fusion_cfg.weights.yawn = 0.0
        fusion_cfg.weights.eye_closure = 0.6
        fusion_cfg.weights.perclos = 0.4
    elif mode == "head_only":
        fusion_cfg.weights.eye_closure = 0.0
        fusion_cfg.weights.perclos = 0.0
        fusion_cfg.weights.head_nod = 0.8
        fusion_cfg.weights.yawn = 0.2

    for label, frame_list in episodes:
        fuser = DrowsinessFuser(fusion_cfg, det_cfg)
        ground_truth_drowsy = label in ("closed_4s", "nodding")
        alarm_triggered = False
        alarm_time: Optional[float] = None

        if label in ("alert_normal", "blinking"):
            normal_duration_s += frame_list[-1].timestamp

        for feat in frame_list:
            reading = fuser.process(feat)
            if reading.level == DrowsinessLevel.WARNING and not alarm_triggered:
                alarm_triggered = True
                alarm_time = feat.timestamp

        if ground_truth_drowsy:
            if alarm_triggered:
                tp += 1
                if alarm_time is not None:
                    latencies.append(alarm_time)
            else:
                fn += 1
        else:
            if alarm_triggered:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    # False alarms per hour on normal behaviour
    hours = normal_duration_s / 3600.0 if normal_duration_s > 0 else 1.0
    fp_rate = fp / hours if hours > 0 else 0.0

    return EvaluationMetrics(
        mode=mode,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        mean_latency_s=float(mean_lat),
        false_alarm_rate_per_hour=float(fp_rate),
    )


def run_ablation_study() -> Dict[str, EvaluationMetrics]:
    """Execute ablation evaluation comparing eyes-only, head-pose-only, and multi-signal fusion."""
    episodes = generate_synthetic_evaluation_events(num_episodes=25)
    modes = ["eyes_only", "head_only", "fused"]
    results = {}
    for m in modes:
        results[m] = evaluate_detector_configuration(episodes, mode=m)
    return results

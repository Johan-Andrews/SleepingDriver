"""Unit tests for multi-signal fusion, hard trigger rule, and hysteresis."""

import pytest
from src.config import FusionConfig, DetectionConfig
from src.decision.fusion import DrowsinessFuser, DrowsinessLevel
from src.vision.features import DriverFeatures


def make_features(
    eye_closed_s: float = 0.0,
    perclos: float = 0.0,
    is_head_nodding: bool = False,
    is_distracted: bool = False,
    yawns_per_min: float = 0.0,
    pitch_rel: float = 0.0,
) -> DriverFeatures:
    return DriverFeatures(
        timestamp=100.0,
        face_detected=True,
        ear_raw=0.30,
        ear_smooth=0.30,
        is_eye_closed=eye_closed_s > 0,
        eye_closed_duration_s=eye_closed_s,
        total_blinks=5,
        blink_rate_per_min=12.0,
        avg_blink_duration_s=0.2,
        perclos=perclos,
        mar_raw=0.15,
        mar_smooth=0.15,
        is_yawning=False,
        yawn_count=0,
        yawns_per_min=yawns_per_min,
        pitch=0.0,
        yaw=45.0 if is_distracted else 0.0,
        roll=0.0,
        pitch_rel=pitch_rel,
        is_head_nodding=is_head_nodding,
        is_distracted=is_distracted,
        no_face_duration_s=0.0,
    )


def test_three_second_closure_triggers_warning():
    fuser = DrowsinessFuser(FusionConfig(), DetectionConfig(eye_closure_trigger_s=3.0))
    feat = make_features(eye_closed_s=3.2)
    reading = fuser.process(feat)

    assert reading.is_hard_rule_triggered
    assert reading.level == DrowsinessLevel.WARNING


def test_quick_blink_stays_alert():
    fuser = DrowsinessFuser(FusionConfig(), DetectionConfig(blink_max_s=0.5))
    feat = make_features(eye_closed_s=0.2)
    reading = fuser.process(feat)

    assert not reading.is_hard_rule_triggered
    assert reading.level == DrowsinessLevel.ALERT
    assert reading.score < 20.0


def test_distraction_flag_independent_from_drowsiness():
    fuser = DrowsinessFuser(FusionConfig(), DetectionConfig())
    feat = make_features(is_distracted=True)
    reading = fuser.process(feat)

    assert reading.is_distracted
    assert reading.level == DrowsinessLevel.ALERT


def test_hysteresis_prevents_flicker():
    cfg = FusionConfig(caution_score=40.0, warning_score=70.0, hysteresis=10.0)
    fuser = DrowsinessFuser(cfg, DetectionConfig())

    # Raise to caution (score ~50)
    for _ in range(15):
        feat = make_features(eye_closed_s=2.0, perclos=0.25)
        reading = fuser.process(feat)

    assert reading.level == DrowsinessLevel.CAUTION
    assert reading.score >= 40.0

    # Slightly decrease so raw score would be around 35 (below 40, but above 40 - 10 = 30)
    # The state should stay CAUTION due to hysteresis
    for _ in range(3):
        feat = make_features(eye_closed_s=1.2, perclos=0.20)
        reading = fuser.process(feat)

    assert reading.level == DrowsinessLevel.CAUTION

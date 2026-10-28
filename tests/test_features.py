"""Unit tests for EAR, MAR, PERCLOS, Head Pose, and FeatureExtractor."""

import numpy as np
import pytest

from src.config import DetectionConfig
from src.vision.features import (
    calculate_ear,
    calculate_mar,
    calculate_head_pose,
    FeatureExtractor,
    RIGHT_EYE_INDICES,
    LEFT_EYE_INDICES,
)
from src.vision.calibration import Calibrator, CalibrationProfile


def create_synthetic_landmarks(ear_target: float = 0.35, mar_target: float = 0.2) -> np.ndarray:
    """Create a 478-point synthetic landmark array with specified EAR and MAR geometry."""
    lms = np.zeros((478, 2), dtype=np.float32)

    # Set head pose anchor defaults: image center (320, 240)
    lms[1] = [320, 240]    # nose tip
    lms[152] = [320, 350]  # chin
    lms[33] = [260, 190]   # right eye (image left)
    lms[263] = [380, 190]  # left eye (image right)
    lms[61] = [280, 290]   # right mouth (image left)
    lms[291] = [360, 290]  # left mouth (image right)

    # Right eye
    horiz = 20.0
    vert = ear_target * horiz  # (v1 + v2) / (2 * horiz) = 2*vert / (2*horiz) = vert/horiz
    p1 = [250.0, 200.0]
    p4 = [250.0 + horiz, 200.0]
    p2 = [255.0, 200.0 - vert / 2.0]
    p6 = [255.0, 200.0 + vert / 2.0]
    p3 = [265.0, 200.0 - vert / 2.0]
    p5 = [265.0, 200.0 + vert / 2.0]

    for idx, pt in zip(RIGHT_EYE_INDICES, [p1, p2, p3, p4, p5, p6]):
        lms[idx] = pt

    # Left eye
    p1_l = [350.0, 200.0]
    p4_l = [350.0 + horiz, 200.0]
    p2_l = [355.0, 200.0 - vert / 2.0]
    p6_l = [355.0, 200.0 + vert / 2.0]
    p3_l = [365.0, 200.0 - vert / 2.0]
    p5_l = [365.0, 200.0 + vert / 2.0]

    for idx, pt in zip(LEFT_EYE_INDICES, [p1_l, p2_l, p3_l, p4_l, p5_l, p6_l]):
        lms[idx] = pt

    # Mouth
    m_horiz = 40.0
    m_vert = (2.0 * m_horiz * mar_target) / 3.0
    lms[78] = [300.0, 310.0]
    lms[308] = [300.0 + m_horiz, 310.0]
    lms[13] = [320.0, 310.0 - m_vert / 2.0]
    lms[14] = [320.0, 310.0 + m_vert / 2.0]
    lms[81] = [310.0, 310.0 - m_vert / 2.0]
    lms[178] = [310.0, 310.0 + m_vert / 2.0]
    lms[311] = [330.0, 310.0 - m_vert / 2.0]
    lms[402] = [330.0, 310.0 + m_vert / 2.0]

    return lms


def test_ear_calculation_exact():
    lms = create_synthetic_landmarks(ear_target=0.30)
    r_ear, l_ear, avg_ear = calculate_ear(lms)
    assert pytest.approx(avg_ear, rel=1e-2) == 0.30
    assert pytest.approx(r_ear, rel=1e-2) == 0.30
    assert pytest.approx(l_ear, rel=1e-2) == 0.30


def test_mar_calculation_exact():
    lms = create_synthetic_landmarks(mar_target=0.65)
    mar = calculate_mar(lms)
    assert pytest.approx(mar, rel=1e-2) == 0.65


def test_head_pose_estimation():
    lms = create_synthetic_landmarks()
    pitch, yaw, roll = calculate_head_pose(lms, 640, 480)
    # Head facing forward should produce angles near zero
    assert abs(yaw) < 30.0
    assert abs(roll) < 30.0


def test_feature_extractor_blink_vs_long_closure():
    config = DetectionConfig(
        ear_default_threshold=0.21,
        eye_closure_trigger_s=3.0,
        blink_max_s=0.5,
        smoothing_frames=1,
    )
    extractor = FeatureExtractor(config)

    open_lms = create_synthetic_landmarks(ear_target=0.35)
    closed_lms = create_synthetic_landmarks(ear_target=0.10)

    # 1. Simulate a quick 0.2s normal blink
    t0 = 100.0
    extractor.update(open_lms, True, 640, 480, timestamp=t0)
    # Blink begins
    extractor.update(closed_lms, True, 640, 480, timestamp=t0 + 0.1)
    # Blink ends at 0.2s
    feat_open = extractor.update(open_lms, True, 640, 480, timestamp=t0 + 0.2)

    assert feat_open.total_blinks == 1
    assert feat_open.eye_closed_duration_s == 0.0

    # 2. Simulate continuous closure exceeding 3.0 seconds
    t1 = 200.0
    extractor.update(closed_lms, True, 640, 480, timestamp=t1)
    extractor.update(closed_lms, True, 640, 480, timestamp=t1 + 1.5)
    feat_closed = extractor.update(closed_lms, True, 640, 480, timestamp=t1 + 3.2)

    assert feat_closed.is_eye_closed
    assert feat_closed.eye_closed_duration_s >= 3.0


def test_calibrator_success_and_fallback():
    config = DetectionConfig(ear_closed_ratio=0.75, ear_default_threshold=0.21)
    calib = Calibrator(config, calibration_duration_s=1.0)
    calib.start()

    # Collect 20 samples
    for i in range(20):
        calib.update(ear=0.32, mar=0.15, pitch=5.0, yaw=0.0, roll=0.0, face_detected=True)

    profile = calib.get_profile()
    assert profile.is_calibrated
    assert pytest.approx(profile.baseline_ear, rel=1e-2) == 0.32
    assert pytest.approx(profile.closed_threshold, rel=1e-2) == 0.32 * 0.75

    # Fallback test with zero samples
    calib_empty = Calibrator(config, calibration_duration_s=1.0)
    calib_empty.start()
    profile_fallback = calib_empty.get_profile()
    assert not profile_fallback.is_calibrated
    assert profile_fallback.closed_threshold == config.ear_default_threshold

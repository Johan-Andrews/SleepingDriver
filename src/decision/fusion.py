"""Multi-signal drowsiness fusion with hysteresis filtering."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Optional

from src.config import FusionConfig, DetectionConfig
from src.vision.features import DriverFeatures

logger = logging.getLogger(__name__)


class DrowsinessLevel(str, Enum):
    ALERT = "ALERT"
    CAUTION = "CAUTION"
    WARNING = "WARNING"


@dataclass
class DrowsinessReading:
    score: float                      # 0.0 to 100.0
    level: DrowsinessLevel            # ALERT, CAUTION, or WARNING
    eye_closed_s: float
    perclos: float
    yawn_rate: float
    pitch: float
    yaw: float
    is_head_nodding: bool
    is_distracted: bool
    is_hard_rule_triggered: bool     # True if eye closure >= trigger (>= 3s)
    simulated_drowsiness: bool = False


class DrowsinessFuser:
    """Fuses multi-modal visual driver signals into a calibrated drowsiness score."""

    def __init__(self, fusion_config: FusionConfig, detection_config: DetectionConfig) -> None:
        self.config = fusion_config
        self.det_config = detection_config

        self._current_level = DrowsinessLevel.ALERT
        self._filtered_score: float = 0.0
        self._simulated_drowsy: bool = False

    def set_simulated_drowsiness(self, enabled: bool) -> None:
        """Inject keyboard simulated drowsiness (key 'D')."""
        self._simulated_drowsy = enabled
        logger.debug("Simulated drowsiness set to: %s", enabled)

    def process(self, features: DriverFeatures) -> DrowsinessReading:
        """Fuse incoming features into a composite score and evaluated level."""
        weights = self.config.weights

        # 1. Sub-score for continuous eye closure
        eye_s = features.eye_closed_duration_s
        trigger_s = self.det_config.eye_closure_trigger_s
        hard_rule = eye_s >= trigger_s

        if eye_s < self.det_config.blink_max_s:
            s_closure = 0.0
        else:
            # Scale from blink_max_s up to trigger_s
            ratio = (eye_s - self.det_config.blink_max_s) / max(0.1, trigger_s - self.det_config.blink_max_s)
            s_closure = min(100.0, max(0.0, ratio * 100.0))

        # 2. Sub-score for PERCLOS (standard benchmark: 20% to 40%+ indicates drowsy)
        s_perclos = min(100.0, (features.perclos / 0.35) * 100.0)

        # 3. Sub-score for Head Nod
        if features.is_head_nodding:
            s_nod = 100.0
        elif features.pitch_rel < -10.0:
            s_nod = min(80.0, (abs(features.pitch_rel) - 10.0) * 8.0)
        else:
            s_nod = 0.0

        # 4. Sub-score for Yawning
        s_yawn = min(100.0, features.yawns_per_min * 33.3)
        if features.is_yawning:
            s_yawn = max(s_yawn, 70.0)

        # Compute weighted raw score
        raw_score = (
            weights.eye_closure * s_closure
            + weights.perclos * s_perclos
            + weights.head_nod * s_nod
            + weights.yawn * s_yawn
        )

        # Hard rule from Section 4.5: continuous eye closure > trigger_s always raises at least Warning
        if hard_rule:
            raw_score = max(raw_score, self.config.warning_score)

        if self._simulated_drowsy:
            raw_score = 95.0
            eye_s = max(eye_s, trigger_s + 0.5)
            hard_rule = True

        raw_score = min(100.0, max(0.0, raw_score))

        # Exponential moving filter to smooth out score transitions
        alpha = 0.3
        self._filtered_score = (alpha * raw_score) + ((1.0 - alpha) * self._filtered_score)

        # Hysteresis state transitions
        caution_th = self.config.caution_score
        warning_th = self.config.warning_score
        h = self.config.hysteresis

        if hard_rule or self._filtered_score >= warning_th:
            self._current_level = DrowsinessLevel.WARNING
        elif self._current_level == DrowsinessLevel.WARNING:
            # Must fall below (warning - h) to downgrade to CAUTION
            if self._filtered_score < (warning_th - h) and not hard_rule:
                if self._filtered_score >= caution_th:
                    self._current_level = DrowsinessLevel.CAUTION
                else:
                    self._current_level = DrowsinessLevel.ALERT
        elif self._current_level == DrowsinessLevel.CAUTION:
            # Upgrade to WARNING if threshold crossed
            if self._filtered_score >= warning_th or hard_rule:
                self._current_level = DrowsinessLevel.WARNING
            # Must fall below (caution - h) to downgrade to ALERT
            elif self._filtered_score < (caution_th - h):
                self._current_level = DrowsinessLevel.ALERT
        else:  # Currently ALERT
            if hard_rule or self._filtered_score >= warning_th:
                self._current_level = DrowsinessLevel.WARNING
            elif self._filtered_score >= caution_th:
                self._current_level = DrowsinessLevel.CAUTION

        return DrowsinessReading(
            score=float(self._filtered_score),
            level=self._current_level,
            eye_closed_s=eye_s,
            perclos=features.perclos,
            yawn_rate=features.yawns_per_min,
            pitch=features.pitch,
            yaw=features.yaw,
            is_head_nodding=features.is_head_nodding,
            is_distracted=features.is_distracted,
            is_hard_rule_triggered=hard_rule,
            simulated_drowsiness=self._simulated_drowsy,
        )

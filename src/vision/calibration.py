"""Per-driver baseline calibration module for personalized EAR, MAR, and neutral head pose."""

from __future__ import annotations
from dataclasses import dataclass
import logging
import time
from typing import List, Optional
import numpy as np

from src.config import DetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class CalibrationProfile:
    baseline_ear: float
    closed_threshold: float
    baseline_mar: float
    neutral_pitch: float
    neutral_yaw: float
    neutral_roll: float
    is_calibrated: bool


class Calibrator:
    """Collects baseline driver facial features across a calibration period."""

    def __init__(self, config: DetectionConfig, calibration_duration_s: float = 5.0) -> None:
        self.config = config
        self.duration_s = calibration_duration_s

        self.start_time: Optional[float] = None
        self.is_active: bool = False
        self.is_complete: bool = False

        self._ear_samples: List[float] = []
        self._mar_samples: List[float] = []
        self._pitch_samples: List[float] = []
        self._yaw_samples: List[float] = []
        self._roll_samples: List[float] = []

    def start(self) -> None:
        """Start the calibration countdown."""
        self.start_time = time.perf_counter()
        self.is_active = True
        self.is_complete = False
        self._ear_samples.clear()
        self._mar_samples.clear()
        self._pitch_samples.clear()
        self._yaw_samples.clear()
        self._roll_samples.clear()
        logger.info("Driver calibration started (duration: %.1f s).", self.duration_s)

    def update(
        self,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        roll: float,
        face_detected: bool,
    ) -> bool:
        """Collect a sample if face is detected. Returns True when finished."""
        if not self.is_active or self.is_complete:
            return self.is_complete

        now = time.perf_counter()
        if self.start_time is None:
            self.start_time = now

        if face_detected:
            self._ear_samples.append(ear)
            self._mar_samples.append(mar)
            self._pitch_samples.append(pitch)
            self._yaw_samples.append(yaw)
            self._roll_samples.append(roll)

        if (now - self.start_time) >= self.duration_s:
            self.is_active = False
            self.is_complete = True
            logger.info("Calibration finished with %d valid samples.", len(self._ear_samples))
            return True

        return False

    def get_progress(self) -> float:
        """Return calibration progress from 0.0 to 1.0."""
        if not self.is_active or self.start_time is None:
            return 1.0 if self.is_complete else 0.0
        elapsed = time.perf_counter() - self.start_time
        return min(1.0, max(0.0, elapsed / self.duration_s))

    def get_profile(self) -> CalibrationProfile:
        """Compute the final baseline profile or fallback to defaults."""
        min_samples = 15
        if len(self._ear_samples) < min_samples:
            logger.warning(
                "Insufficient samples (%d/%d) during calibration. Using default config thresholds.",
                len(self._ear_samples),
                min_samples,
            )
            base_ear = self.config.ear_default_threshold / self.config.ear_closed_ratio
            return CalibrationProfile(
                baseline_ear=base_ear,
                closed_threshold=self.config.ear_default_threshold,
                baseline_mar=0.2,
                neutral_pitch=0.0,
                neutral_yaw=0.0,
                neutral_roll=0.0,
                is_calibrated=False,
            )

        base_ear = float(np.median(self._ear_samples))
        closed_threshold = base_ear * self.config.ear_closed_ratio
        base_mar = float(np.median(self._mar_samples))
        neutral_pitch = float(np.median(self._pitch_samples))
        neutral_yaw = float(np.median(self._yaw_samples))
        neutral_roll = float(np.median(self._roll_samples))

        logger.info(
            "Calibration successful: baseline_EAR=%.3f, closed_thresh=%.3f, pitch=%.1f deg",
            base_ear,
            closed_threshold,
            neutral_pitch,
        )

        return CalibrationProfile(
            baseline_ear=base_ear,
            closed_threshold=closed_threshold,
            baseline_mar=base_mar,
            neutral_pitch=neutral_pitch,
            neutral_yaw=neutral_yaw,
            neutral_roll=neutral_roll,
            is_calibrated=True,
        )

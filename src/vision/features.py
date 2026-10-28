"""Facial feature extraction: EAR, MAR, Head Pose (solvePnP), PERCLOS, and Blink Dynamics."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import Deque, Optional, Tuple
import cv2
import numpy as np

from src.config import DetectionConfig
from src.vision.calibration import CalibrationProfile

logger = logging.getLogger(__name__)

# Standard 3D model points in camera coordinates (X right, Y down, Z forward)
FACE_3D_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),         # Nose tip (index 1)
        (0.0, 110.0, -50.0),     # Chin (index 152)
        (-60.0, -50.0, -40.0),   # Right eye outer (index 33 - subject's right, image left)
        (60.0, -50.0, -40.0),    # Left eye outer (index 263 - subject's left, image right)
        (-40.0, 50.0, -30.0),    # Right mouth corner (index 61)
        (40.0, 50.0, -30.0),     # Left mouth corner (index 291)
    ],
    dtype=np.float64,
)

HEAD_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def compute_ear_for_eye(landmarks: np.ndarray, indices: list[int]) -> float:
    """Compute Eye Aspect Ratio (EAR) for a single eye given 6 landmarks."""
    p1, p2, p3, p4, p5, p6 = [landmarks[idx] for idx in indices]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * horiz))


def calculate_ear(landmarks: np.ndarray) -> Tuple[float, float, float]:
    """Calculate right EAR, left EAR, and their average."""
    right_ear = compute_ear_for_eye(landmarks, RIGHT_EYE_INDICES)
    left_ear = compute_ear_for_eye(landmarks, LEFT_EYE_INDICES)
    avg_ear = (right_ear + left_ear) / 2.0
    return right_ear, left_ear, avg_ear


def calculate_mar(landmarks: np.ndarray) -> float:
    """Calculate Mouth Aspect Ratio (MAR) using multi-point lip contours."""
    # Top 13, Bottom 14, Left 78, Right 308, pairs 81/178 and 311/402
    p13, p14 = landmarks[13], landmarks[14]
    p81, p178 = landmarks[81], landmarks[178]
    p311, p402 = landmarks[311], landmarks[402]
    p78, p308 = landmarks[78], landmarks[308]

    v1 = np.linalg.norm(p13 - p14)
    v2 = np.linalg.norm(p81 - p178)
    v3 = np.linalg.norm(p311 - p402)
    horiz = np.linalg.norm(p78 - p308)

    if horiz < 1e-6:
        return 0.0
    return float((v1 + v2 + v3) / (2.0 * horiz))


def calculate_head_pose(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float, float]:
    """Estimate head pose (pitch, yaw, roll in degrees) via cv2.solvePnP."""
    image_points = np.array([landmarks[idx][:2] for idx in HEAD_POSE_LANDMARK_INDICES], dtype=np.float64)

    focal_length = float(frame_width)
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        FACE_3D_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    rot_matrix, _ = cv2.Rodrigues(rotation_vector)
    # Decompose rotation matrix into Euler angles
    # pitch (x-axis), yaw (y-axis), roll (z-axis)
    sy = np.sqrt(rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
        yaw = np.arctan2(-rot_matrix[2, 0], sy)
        roll = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    else:
        pitch = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
        yaw = np.arctan2(-rot_matrix[2, 0], sy)
        roll = 0.0

    pitch_deg = float(np.degrees(pitch))
    yaw_deg = float(np.degrees(yaw))
    roll_deg = float(np.degrees(roll))
    return pitch_deg, yaw_deg, roll_deg


@dataclass
class DriverFeatures:
    timestamp: float
    face_detected: bool
    ear_raw: float
    ear_smooth: float
    is_eye_closed: bool
    eye_closed_duration_s: float
    total_blinks: int
    blink_rate_per_min: float
    avg_blink_duration_s: float
    perclos: float
    mar_raw: float
    mar_smooth: float
    is_yawning: bool
    yawn_count: int
    yawns_per_min: float
    pitch: float
    yaw: float
    roll: float
    pitch_rel: float
    is_head_nodding: bool
    is_distracted: bool
    no_face_duration_s: float


class FeatureExtractor:
    """Processes frame landmarks over time to extract smoothed driver dynamics."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.profile: Optional[CalibrationProfile] = None

        # Smoothing buffers
        n_smooth = max(1, config.smoothing_frames)
        self._ear_buffer: Deque[float] = deque(maxlen=n_smooth)
        self._mar_buffer: Deque[float] = deque(maxlen=n_smooth)
        self._pitch_buffer: Deque[float] = deque(maxlen=n_smooth)
        self._yaw_buffer: Deque[float] = deque(maxlen=n_smooth)
        self._roll_buffer: Deque[float] = deque(maxlen=n_smooth)

        # Timers and state
        self._eye_closed_start: Optional[float] = None
        self._current_closure_duration: float = 0.0

        # Blink tracking
        self._blink_history: Deque[Tuple[float, float]] = deque()  # (timestamp, duration)
        self._total_blinks: int = 0

        # PERCLOS tracking window
        self._perclos_window_s: float = config.perclos_window_s
        self._perclos_history: Deque[Tuple[float, bool]] = deque()  # (timestamp, is_closed)

        # Yawn tracking
        self._yawn_start: Optional[float] = None
        self._yawn_history: Deque[float] = deque()  # timestamps of completed yawns
        self._total_yawns: int = 0

        # Head nod tracking
        self._nod_start: Optional[float] = None
        self._is_head_nodding: bool = False

        # No-face tracking
        self._no_face_start: Optional[float] = None
        self._no_face_duration: float = 0.0

    def set_profile(self, profile: CalibrationProfile) -> None:
        """Apply driver baseline profile."""
        self.profile = profile

    def update(
        self,
        landmarks: Optional[np.ndarray],
        face_detected: bool,
        frame_width: int,
        frame_height: int,
        timestamp: Optional[float] = None,
    ) -> DriverFeatures:
        """Update feature buffers and return the latest driver features snapshot."""
        now = time.perf_counter() if timestamp is None else timestamp
        closed_thresh = self.profile.closed_threshold if self.profile else self.config.ear_default_threshold

        if not face_detected or landmarks is None:
            if self._no_face_start is None:
                self._no_face_start = now
            self._no_face_duration = now - self._no_face_start

            # Reset closure timer when face is not visible (Section 4.7 rule)
            self._eye_closed_start = None
            self._current_closure_duration = 0.0
            self._yawn_start = None
            self._nod_start = None
            self._is_head_nodding = False

            return DriverFeatures(
                timestamp=now,
                face_detected=False,
                ear_raw=0.0,
                ear_smooth=0.0,
                is_eye_closed=False,
                eye_closed_duration_s=0.0,
                total_blinks=self._total_blinks,
                blink_rate_per_min=self._calc_rate(self._blink_history, now, 60.0),
                avg_blink_duration_s=self._calc_avg_blink_duration(),
                perclos=self._calc_perclos(now),
                mar_raw=0.0,
                mar_smooth=0.0,
                is_yawning=False,
                yawn_count=self._total_yawns,
                yawns_per_min=self._calc_rate(self._yawn_history, now, 60.0),
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                pitch_rel=0.0,
                is_head_nodding=False,
                is_distracted=False,
                no_face_duration_s=self._no_face_duration,
            )

        # Face is detected
        self._no_face_start = None
        self._no_face_duration = 0.0

        # 1. EAR
        _, _, raw_ear = calculate_ear(landmarks)
        self._ear_buffer.append(raw_ear)
        smooth_ear = float(np.mean(self._ear_buffer))
        is_eye_closed = smooth_ear < closed_thresh

        # Eye closure timer & blink stats
        if is_eye_closed:
            if self._eye_closed_start is None:
                self._eye_closed_start = now
            self._current_closure_duration = now - self._eye_closed_start
        else:
            if self._eye_closed_start is not None:
                closed_time = now - self._eye_closed_start
                if closed_time <= self.config.blink_max_s:
                    # Counts as a normal blink
                    self._total_blinks += 1
                    self._blink_history.append((now, closed_time))
                self._eye_closed_start = None
            self._current_closure_duration = 0.0

        # PERCLOS
        self._perclos_history.append((now, is_eye_closed))
        perclos_val = self._calc_perclos(now)

        # 2. MAR and Yawn
        raw_mar = calculate_mar(landmarks)
        self._mar_buffer.append(raw_mar)
        smooth_mar = float(np.mean(self._mar_buffer))
        yawn_thresh = self.config.mar_yawn_threshold

        is_yawning = False
        if smooth_mar > yawn_thresh:
            if self._yawn_start is None:
                self._yawn_start = now
            elif (now - self._yawn_start) >= self.config.yawn_min_duration_s:
                is_yawning = True
        else:
            if self._yawn_start is not None:
                if (now - self._yawn_start) >= self.config.yawn_min_duration_s:
                    self._total_yawns += 1
                    self._yawn_history.append(now)
                self._yawn_start = None

        # 3. Head Pose
        raw_pitch, raw_yaw, raw_roll = calculate_head_pose(landmarks, frame_width, frame_height)
        self._pitch_buffer.append(raw_pitch)
        self._yaw_buffer.append(raw_yaw)
        self._roll_buffer.append(raw_roll)

        smooth_pitch = float(np.mean(self._pitch_buffer))
        smooth_yaw = float(np.mean(self._yaw_buffer))
        smooth_roll = float(np.mean(self._roll_buffer))

        neutral_pitch = self.profile.neutral_pitch if self.profile else 0.0
        pitch_rel = smooth_pitch - neutral_pitch

        # Head nod detection (pitch downwards)
        if pitch_rel < -self.config.pitch_down_deg:
            if self._nod_start is None:
                self._nod_start = now
            elif (now - self._nod_start) >= self.config.nod_duration_s:
                self._is_head_nodding = True
        else:
            self._nod_start = None
            self._is_head_nodding = False

        # Distraction (large horizontal yaw away from center)
        is_distracted = abs(smooth_yaw) > 30.0

        return DriverFeatures(
            timestamp=now,
            face_detected=True,
            ear_raw=raw_ear,
            ear_smooth=smooth_ear,
            is_eye_closed=is_eye_closed,
            eye_closed_duration_s=self._current_closure_duration,
            total_blinks=self._total_blinks,
            blink_rate_per_min=self._calc_rate(self._blink_history, now, 60.0),
            avg_blink_duration_s=self._calc_avg_blink_duration(),
            perclos=perclos_val,
            mar_raw=raw_mar,
            mar_smooth=smooth_mar,
            is_yawning=is_yawning,
            yawn_count=self._total_yawns,
            yawns_per_min=self._calc_rate(self._yawn_history, now, 60.0),
            pitch=smooth_pitch,
            yaw=smooth_yaw,
            roll=smooth_roll,
            pitch_rel=pitch_rel,
            is_head_nodding=self._is_head_nodding,
            is_distracted=is_distracted,
            no_face_duration_s=0.0,
        )

    def _calc_perclos(self, current_time: float) -> float:
        """Calculate PERCLOS over sliding window."""
        cutoff = current_time - self._perclos_window_s
        while self._perclos_history and self._perclos_history[0][0] < cutoff:
            self._perclos_history.popleft()
        if not self._perclos_history:
            return 0.0
        closed_count = sum(1 for _, closed in self._perclos_history if closed)
        return float(closed_count / len(self._perclos_history))

    def _calc_rate(self, history: Deque, current_time: float, window_s: float) -> float:
        """Calculate event occurrences per minute over the given window."""
        cutoff = current_time - window_s
        while history and (history[0] if isinstance(history[0], (int, float)) else history[0][0]) < cutoff:
            history.popleft()
        count = len(history)
        return float((count / window_s) * 60.0)

    def _calc_avg_blink_duration(self) -> float:
        """Calculate average blink duration in the active window."""
        if not self._blink_history:
            return 0.0
        durations = [dur for _, dur in self._blink_history]
        return float(np.mean(durations))

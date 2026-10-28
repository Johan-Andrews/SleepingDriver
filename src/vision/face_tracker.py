"""Face landmark detection wrapper isolating MediaPipe implementation details."""

from __future__ import annotations
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceTrackerResult:
    face_detected: bool
    landmarks: Optional[np.ndarray]  # (N, 2) in pixel coordinates
    landmarks_3d: Optional[np.ndarray]  # (N, 3) in pixel coordinates
    confidence: float
    frame_width: int
    frame_height: int


class FaceTracker:
    """Wraps MediaPipe FaceLandmarker to provide unified landmark arrays."""

    def __init__(
        self,
        model_path: str = "data/models/face_landmarker.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self._landmarker = None
        self._init_landmarker(min_detection_confidence, min_tracking_confidence)

    def _init_landmarker(self, min_det: float, min_track: float) -> None:
        """Initialize MediaPipe Tasks FaceLandmarker."""
        if not Path(self.model_path).is_file():
            logger.error("FaceLandmarker model asset not found at '%s'. Tracking disabled.", self.model_path)
            return

        try:
            from mediapipe import tasks
            from mediapipe.tasks.python import vision

            base_options = tasks.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_det,
                min_face_presence_confidence=min_track,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize MediaPipe FaceLandmarker: %s", e)
            self._landmarker = None

    def process_frame(self, bgr_frame: np.ndarray) -> FaceTrackerResult:
        """Process a BGR frame and return detected 2D and 3D landmarks."""
        h, w = bgr_frame.shape[:2]
        if self._landmarker is None or bgr_frame is None:
            return FaceTrackerResult(
                face_detected=False,
                landmarks=None,
                landmarks_3d=None,
                confidence=0.0,
                frame_width=w,
                frame_height=h,
            )

        try:
            import mediapipe as mp
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self._landmarker.detect(mp_image)

            if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
                return FaceTrackerResult(
                    face_detected=False,
                    landmarks=None,
                    landmarks_3d=None,
                    confidence=0.0,
                    frame_width=w,
                    frame_height=h,
                )

            raw_lms = detection_result.face_landmarks[0]
            landmarks_2d = np.empty((len(raw_lms), 2), dtype=np.float32)
            landmarks_3d = np.empty((len(raw_lms), 3), dtype=np.float32)

            for i, lm in enumerate(raw_lms):
                px = lm.x * w
                py = lm.y * h
                landmarks_2d[i, 0] = px
                landmarks_2d[i, 1] = py
                landmarks_3d[i, 0] = px
                landmarks_3d[i, 1] = py
                landmarks_3d[i, 2] = lm.z * w  # MediaPipe scale depth roughly with image width

            confidence = 0.95
            return FaceTrackerResult(
                face_detected=True,
                landmarks=landmarks_2d,
                landmarks_3d=landmarks_3d,
                confidence=confidence,
                frame_width=w,
                frame_height=h,
            )

        except Exception as e:
            logger.warning("Error during face landmark inference: %s", e)
            return FaceTrackerResult(
                face_detected=False,
                landmarks=None,
                landmarks_3d=None,
                confidence=0.0,
                frame_width=w,
                frame_height=h,
            )

    def close(self) -> None:
        """Free resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

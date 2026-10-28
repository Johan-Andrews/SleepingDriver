"""Video source abstraction for live webcam and pre-recorded video playback."""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoSource:
    """Provides frames from either a hardware webcam or a video file."""

    def __init__(
        self,
        source_type: str = "webcam",
        index: int = 0,
        file_path: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        loop: bool = True,
    ) -> None:
        self.source_type = source_type
        self.index = index
        self.file_path = file_path
        self.target_width = width
        self.target_height = height
        self.loop = loop

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self._open()

    def _open(self) -> None:
        """Open the video capture device or file."""
        if self.source_type == "video":
            if not self.file_path or not Path(self.file_path).is_file():
                logger.warning("Video file '%s' not found. Falling back to synthetic source.", self.file_path)
                self.is_opened = False
                return
            self.cap = cv2.VideoCapture(self.file_path)
        else:
            # Webcam
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if cv2.CAP_DSHOW is not None else 0)
            if not self.cap.isOpened():
                # Fallback to default backend if CAP_DSHOW fails
                self.cap = cv2.VideoCapture(self.index)

        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.is_opened = True
            logger.info("VideoSource opened (%s, index/file=%s)", self.source_type, self.file_path or self.index)
        else:
            self.is_opened = False
            logger.warning("Could not open video source (%s, %s)", self.source_type, self.file_path or self.index)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame. If video finishes and loop is enabled, rewinds."""
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            if self.source_type == "video" and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return False, None
            else:
                return False, None

        if frame is not None and (frame.shape[1] != self.target_width or frame.shape[0] != self.target_height):
            frame = cv2.resize(frame, (self.target_width, self.target_height))

        return True, frame

    def release(self) -> None:
        """Release capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False

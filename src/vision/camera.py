"""Threaded video capture module to keep UI and simulation decoupled from camera latency."""

from __future__ import annotations
import logging
import threading
import time
from typing import Optional, Tuple
import numpy as np

from src.vision.video_source import VideoSource

logger = logging.getLogger(__name__)


class ThreadedCamera:
    """Runs frame acquisition in a dedicated daemon thread to prevent UI blocking."""

    def __init__(
        self,
        source: VideoSource,
        target_fps: float = 30.0,
    ) -> None:
        self.source = source
        self.target_fps = target_fps
        self._interval = 1.0 / max(1.0, target_fps)

        self._frame: Optional[np.ndarray] = None
        self._ret: bool = False
        self._timestamp: float = 0.0
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Metrics
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.perf_counter()

    def start(self) -> ThreadedCamera:
        """Start the background capture loop."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True, name="CameraWorker")
        self._thread.start()
        logger.info("Threaded camera capture started.")
        return self

    def _update_loop(self) -> None:
        """Continuously grab frames and maintain the latest frame."""
        while self._running:
            t_start = time.perf_counter()
            ret, frame = self.source.read()
            t_capture = time.perf_counter()

            with self._lock:
                self._ret = ret
                if ret and frame is not None:
                    self._frame = frame
                    self._timestamp = t_capture
                    self._frame_count += 1
                else:
                    self._frame = None

            # Calculate FPS periodically
            elapsed = t_capture - self._fps_timer
            if elapsed >= 1.0:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = t_capture

            # Throttle if source is faster than target or if video read
            cycle_time = time.perf_counter() - t_start
            sleep_time = self._interval - cycle_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Get the latest captured frame, retrieval status, and latency in ms."""
        now = time.perf_counter()
        with self._lock:
            if not self._ret or self._frame is None:
                return False, None, 0.0
            frame = self._frame.copy()
            latency = (now - self._timestamp) * 1000.0
            self.latency_ms = latency
            return True, frame, latency

    def stop(self) -> None:
        """Stop capture thread and release source."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.source.release()
        logger.info("Threaded camera stopped.")

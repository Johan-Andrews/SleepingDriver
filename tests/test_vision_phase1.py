"""Unit tests for Phase 1 vision components: video source, camera, and face tracker."""

import time
import numpy as np
import pytest

from src.vision.video_source import VideoSource
from src.vision.camera import ThreadedCamera
from src.vision.face_tracker import FaceTracker, FaceTrackerResult


def test_missing_video_file_handled_cleanly():
    source = VideoSource(source_type="video", file_path="non_existent_file.mp4")
    assert not source.is_opened
    ret, frame = source.read()
    assert not ret
    assert frame is None
    source.release()


def test_missing_webcam_handled_cleanly():
    # Attempting to open a high index camera that doesn't exist
    source = VideoSource(source_type="webcam", index=999)
    # Shouldn't raise any unhandled exceptions
    ret, frame = source.read()
    assert not ret or frame is None
    source.release()


def test_video_source_reading():
    source = VideoSource(source_type="video", file_path="data/test_videos/sample.mp4", width=320, height=240)
    assert source.is_opened
    ret, frame = source.read()
    assert ret
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    source.release()


def test_threaded_camera_lifecycle():
    source = VideoSource(source_type="video", file_path="data/test_videos/sample.mp4", width=320, height=240)
    cam = ThreadedCamera(source=source, target_fps=30.0)
    cam.start()

    time.sleep(0.1)  # allow loop to grab frame
    ret, frame, latency = cam.read()
    assert ret
    assert frame is not None
    assert latency >= 0.0

    cam.stop()
    assert not cam._running


def test_face_tracker_handles_empty_frame():
    tracker = FaceTracker()
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = tracker.process_frame(black_frame)

    assert isinstance(result, FaceTrackerResult)
    assert not result.face_detected
    assert result.landmarks is None
    assert result.confidence == 0.0
    tracker.close()

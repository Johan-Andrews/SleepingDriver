"""Tests for configuration loading and validation."""

import pytest
from src.config import load_config, AppConfig, CameraConfig, VehicleConfig


def test_default_config_loads():
    cfg = load_config("config/default.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.camera.source in ("webcam", "video")
    assert cfg.vehicle.initial_speed_kmh == 100.0
    assert cfg.detection.eye_closure_trigger_s == 3.0
    assert cfg.fusion.warning_score == 60.0


def test_invalid_friction_raises():
    cfg = AppConfig()
    cfg.vehicle.friction = "invalid_surface"
    with pytest.raises(ValueError, match="Invalid friction"):
        cfg.vehicle.validate()


def test_invalid_camera_source_raises():
    cfg = AppConfig()
    cfg.camera.source = "rtsp_stream"
    with pytest.raises(ValueError, match="Invalid camera source"):
        cfg.camera.validate()

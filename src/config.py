"""Configuration loader and schema validator for driver drowsiness detection and braking simulation."""

from __future__ import annotations
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    source: str = "webcam"  # "webcam" or "video"
    index: int = 0
    video_path: str = "data/test_videos/sample.mp4"
    width: int = 640
    height: int = 480

    def validate(self) -> None:
        if self.source not in ("webcam", "video"):
            raise ValueError(f"Invalid camera source '{self.source}'. Must be 'webcam' or 'video'.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive integers.")


@dataclass
class DetectionConfig:
    ear_closed_ratio: float = 0.75
    ear_default_threshold: float = 0.21
    eye_closure_trigger_s: float = 3.0
    blink_max_s: float = 0.5
    perclos_window_s: float = 60.0
    mar_yawn_threshold: float = 0.6
    yawn_min_duration_s: float = 1.5
    pitch_down_deg: float = 20.0
    nod_duration_s: float = 2.0
    no_face_timeout_s: float = 2.0
    smoothing_frames: int = 5

    def validate(self) -> None:
        if not (0.0 < self.ear_closed_ratio < 1.0):
            raise ValueError("ear_closed_ratio must be between 0.0 and 1.0.")
        if self.ear_default_threshold <= 0.0:
            raise ValueError("ear_default_threshold must be positive.")
        if self.eye_closure_trigger_s <= 0.0:
            raise ValueError("eye_closure_trigger_s must be positive.")
        if self.smoothing_frames < 1:
            raise ValueError("smoothing_frames must be at least 1.")


@dataclass
class FusionWeights:
    eye_closure: float = 0.4
    perclos: float = 0.3
    head_nod: float = 0.2
    yawn: float = 0.1

    def validate(self) -> None:
        total = self.eye_closure + self.perclos + self.head_nod + self.yawn
        if not (0.95 <= total <= 1.05):
            logger.warning("Fusion weights sum to %.3f (expected ~1.0)", total)


@dataclass
class FusionConfig:
    weights: FusionWeights = field(default_factory=FusionWeights)
    caution_score: float = 35.0
    warning_score: float = 60.0
    hysteresis: float = 8.0

    def validate(self) -> None:
        self.weights.validate()
        if not (0.0 <= self.caution_score < self.warning_score <= 100.0):
            raise ValueError("Thresholds must satisfy: 0 <= caution_score < warning_score <= 100.")
        if self.hysteresis < 0.0:
            raise ValueError("hysteresis must be non-negative.")


@dataclass
class AlertsConfig:
    response_interval_s: float = 5.0
    recovery_hold_s: float = 2.0

    def validate(self) -> None:
        if self.response_interval_s <= 0.0:
            raise ValueError("response_interval_s must be positive.")
        if self.recovery_hold_s <= 0.0:
            raise ValueError("recovery_hold_s must be positive.")


@dataclass
class VehicleConfig:
    initial_speed_kmh: float = 100.0
    length_m: float = 4.5
    accel_max: float = 3.0
    friction: str = "dry"  # "dry", "wet", "icy"
    g: float = 9.81

    def validate(self) -> None:
        if self.initial_speed_kmh < 0.0:
            raise ValueError("initial_speed_kmh cannot be negative.")
        if self.length_m <= 0.0:
            raise ValueError("length_m must be positive.")
        if self.friction not in ("dry", "wet", "icy"):
            raise ValueError(f"Invalid friction '{self.friction}'. Must be 'dry', 'wet', or 'icy'.")
        if self.accel_max <= 0.0:
            raise ValueError("accel_max must be positive.")


@dataclass
class BrakingConfig:
    strategy: str = "adaptive"  # "abrupt", "gradual", "adaptive"
    gradual_jerk: float = 2.0
    gradual_cap: float = 4.0
    v2v_alert: bool = True

    def validate(self) -> None:
        if self.strategy not in ("abrupt", "gradual", "adaptive"):
            raise ValueError(f"Invalid braking strategy '{self.strategy}'.")
        if self.gradual_jerk <= 0.0:
            raise ValueError("gradual_jerk must be positive.")
        if self.gradual_cap <= 0.0:
            raise ValueError("gradual_cap must be positive.")


@dataclass
class FollowerConfig:
    type: str = "attentive"  # "attentive", "distracted", "acc"
    initial_gap_m: float = 30.0
    reaction_time_s: float = 1.0
    brake_fraction: float = 0.6

    def validate(self) -> None:
        if self.type not in ("attentive", "distracted", "acc"):
            raise ValueError(f"Invalid follower type '{self.type}'.")
        if self.initial_gap_m <= 0.0:
            raise ValueError("initial_gap_m must be positive.")
        if self.reaction_time_s < 0.0:
            raise ValueError("reaction_time_s must be non-negative.")
        if not (0.0 < self.brake_fraction <= 1.0):
            raise ValueError("brake_fraction must be between 0.0 and 1.0.")


@dataclass
class SimConfig:
    dt: float = 0.01
    seed: int = 42

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("Simulation dt must be positive.")


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    braking: BrakingConfig = field(default_factory=BrakingConfig)
    follower: FollowerConfig = field(default_factory=FollowerConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    def validate(self) -> None:
        self.camera.validate()
        self.detection.validate()
        self.fusion.validate()
        self.alerts.validate()
        self.vehicle.validate()
        self.braking.validate()
        self.follower.validate()
        self.sim.validate()


def _dict_to_dataclass(cls: Any, data: Dict[str, Any]) -> Any:
    """Helper to instantiate dataclass with nested dictionaries."""
    if not isinstance(data, dict):
        return data
    kwargs = {}
    field_types = {f.name: f.type for f in getattr(cls, "__dataclass_fields__", {}).values()}
    for key, val in data.items():
        if key in field_types:
            target_type = field_types[key]
            if target_type == FusionWeights and isinstance(val, dict):
                kwargs[key] = FusionWeights(**val)
            else:
                kwargs[key] = val
        else:
            logger.debug("Unknown config key '%s' ignored for %s", key, cls.__name__)
    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load, merge with defaults, and validate configuration from YAML."""
    app_config = AppConfig()

    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/default.yaml")

    path = Path(config_path)
    if not path.is_file():
        logger.warning("Config file '%s' not found. Using default parameters.", config_path)
        app_config.validate()
        return app_config

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        if "camera" in raw_data and isinstance(raw_data["camera"], dict):
            app_config.camera = _dict_to_dataclass(CameraConfig, raw_data["camera"])

        if "detection" in raw_data and isinstance(raw_data["detection"], dict):
            app_config.detection = _dict_to_dataclass(DetectionConfig, raw_data["detection"])

        if "fusion" in raw_data and isinstance(raw_data["fusion"], dict):
            fusion_dict = raw_data["fusion"].copy()
            if "weights" in fusion_dict and isinstance(fusion_dict["weights"], dict):
                fusion_dict["weights"] = _dict_to_dataclass(FusionWeights, fusion_dict["weights"])
            app_config.fusion = _dict_to_dataclass(FusionConfig, fusion_dict)

        if "alerts" in raw_data and isinstance(raw_data["alerts"], dict):
            app_config.alerts = _dict_to_dataclass(AlertsConfig, raw_data["alerts"])

        if "vehicle" in raw_data and isinstance(raw_data["vehicle"], dict):
            app_config.vehicle = _dict_to_dataclass(VehicleConfig, raw_data["vehicle"])

        if "braking" in raw_data and isinstance(raw_data["braking"], dict):
            app_config.braking = _dict_to_dataclass(BrakingConfig, raw_data["braking"])

        if "follower" in raw_data and isinstance(raw_data["follower"], dict):
            app_config.follower = _dict_to_dataclass(FollowerConfig, raw_data["follower"])

        if "sim" in raw_data and isinstance(raw_data["sim"], dict):
            app_config.sim = _dict_to_dataclass(SimConfig, raw_data["sim"])

    except Exception as e:
        logger.error("Error reading YAML config from '%s': %s. Reverting to safe defaults.", config_path, e)
        raise

    app_config.validate()
    logger.info("Configuration loaded and validated from '%s'.", config_path)
    return app_config

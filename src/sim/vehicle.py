"""Longitudinal vehicle physics simulation model."""

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Optional

from src.config import VehicleConfig

logger = logging.getLogger(__name__)

FRICTION_COEFFICIENTS = {
    "dry": 0.8,
    "wet": 0.5,
    "icy": 0.2,
}


@dataclass
class VehicleState:
    position_m: float
    speed_mps: float
    speed_kmh: float
    acceleration_mps2: float
    brake_lights_on: bool
    hazards_on: bool
    brake_intensity: float  # 0.0 to 1.0


class Vehicle:
    """Simulates 1D longitudinal vehicle dynamics under friction and braking commands."""

    def __init__(
        self,
        config: VehicleConfig,
        initial_position_m: float = 0.0,
        initial_speed_kmh: Optional[float] = None,
    ) -> None:
        self.config = config
        self.length_m = config.length_m
        self.g = config.g
        self.friction_name = config.friction

        # Initial conditions
        speed_kmh = config.initial_speed_kmh if initial_speed_kmh is None else initial_speed_kmh
        self.position_m = initial_position_m
        self.speed_mps = speed_kmh / 3.6
        self.acceleration_mps2 = 0.0

        # Signaling
        self.brake_lights_on = False
        self.hazards_on = False
        self.brake_intensity = 0.0

    @property
    def mu(self) -> float:
        """Tire-road friction coefficient."""
        return FRICTION_COEFFICIENTS.get(self.friction_name.lower(), 0.8)

    @property
    def a_brake_max(self) -> float:
        """Maximum achievable deceleration based on road friction (positive magnitude)."""
        return self.mu * self.g

    @property
    def a_accel_max(self) -> float:
        """Maximum acceleration capability."""
        return self.config.accel_max

    def step(self, a_cmd: float, dt: float, hazards: bool = False) -> VehicleState:
        """
        Advance vehicle physics by fixed time-step dt.
        a_cmd: commanded acceleration (positive for accel, negative for brake).
        """
        # Friction bound: cannot decelerate faster than mu * g
        max_decel = self.a_brake_max
        a_clamped = max(-max_decel, min(self.a_accel_max, a_cmd))

        # Integrate speed and position
        v_prev = self.speed_mps
        self.speed_mps = max(0.0, v_prev + a_clamped * dt)

        # Average velocity across the timestep for higher integration accuracy
        v_avg = 0.5 * (v_prev + self.speed_mps)
        self.position_m += v_avg * dt
        self.acceleration_mps2 = (self.speed_mps - v_prev) / dt if dt > 0 else a_clamped

        # Signaling state
        is_braking = a_clamped < -0.1 and self.speed_mps > 0.01
        self.brake_lights_on = is_braking
        self.hazards_on = hazards
        self.brake_intensity = min(1.0, max(0.0, -a_clamped / max_decel)) if is_braking else 0.0

        return self.get_state()

    def get_state(self) -> VehicleState:
        """Return the current vehicle snapshot."""
        return VehicleState(
            position_m=self.position_m,
            speed_mps=self.speed_mps,
            speed_kmh=self.speed_mps * 3.6,
            acceleration_mps2=self.acceleration_mps2,
            brake_lights_on=self.brake_lights_on,
            hazards_on=self.hazards_on,
            brake_intensity=self.brake_intensity,
        )

    def reset(self, position_m: float = 0.0, speed_kmh: Optional[float] = None) -> None:
        """Reset vehicle state."""
        self.position_m = position_m
        spd = self.config.initial_speed_kmh if speed_kmh is None else speed_kmh
        self.speed_mps = spd / 3.6
        self.acceleration_mps2 = 0.0
        self.brake_lights_on = False
        self.hazards_on = False
        self.brake_intensity = 0.0

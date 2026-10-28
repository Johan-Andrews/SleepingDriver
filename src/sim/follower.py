"""Following vehicle driver models (Attentive, Distracted, ACC / IDM)."""

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Optional

from src.config import FollowerConfig, VehicleConfig
from src.sim.vehicle import Vehicle, VehicleState

logger = logging.getLogger(__name__)


class FollowerVehicle:
    """Simulates the rear following vehicle and human/ACC driver reaction."""

    def __init__(
        self,
        follower_cfg: FollowerConfig,
        vehicle_cfg: VehicleConfig,
        v2v_enabled: bool = True,
    ) -> None:
        self.follower_cfg = follower_cfg
        self.vehicle_cfg = vehicle_cfg
        self.v2v_enabled = v2v_enabled

        # Internal vehicle model
        # Follower starts behind ego by initial_gap_m + length
        init_pos = -(follower_cfg.initial_gap_m + vehicle_cfg.length_m)
        self.vehicle = Vehicle(vehicle_cfg, initial_position_m=init_pos)

        # Driver reaction tracking
        self.perceived_brake_time: Optional[float] = None
        self.has_reacted = False

        # Configure reaction time based on follower type
        self.reaction_delay = follower_cfg.reaction_time_s
        if follower_cfg.type == "distracted":
            self.reaction_delay = max(self.reaction_delay, 1.8)
        elif follower_cfg.type == "attentive":
            self.reaction_delay = min(self.reaction_delay, 0.9)

    def reset(self, initial_gap_m: Optional[float] = None, initial_speed_kmh: Optional[float] = None) -> None:
        """Reset following vehicle to scenario start."""
        gap = self.follower_cfg.initial_gap_m if initial_gap_m is None else initial_gap_m
        spd = self.vehicle_cfg.initial_speed_kmh if initial_speed_kmh is None else initial_speed_kmh
        init_pos = -(gap + self.vehicle_cfg.length_m)
        self.vehicle.reset(position_m=init_pos, speed_kmh=spd)
        self.perceived_brake_time = None
        self.has_reacted = False

    def step(
        self,
        ego_state: VehicleState,
        dt: float,
        sim_time_s: float,
        v2v_alert_active: bool = False,
    ) -> VehicleState:
        """Advance follower driver model and vehicle physics."""
        a_cmd = self._compute_acceleration(ego_state, dt, sim_time_s, v2v_alert_active)
        return self.vehicle.step(a_cmd, dt)

    def _compute_acceleration(
        self,
        ego_state: VehicleState,
        dt: float,
        sim_time: float,
        v2v_alert_active: bool,
    ) -> float:
        """Calculate driver commanded acceleration."""
        foll_state = self.vehicle.get_state()
        gap = ego_state.position_m - foll_state.position_m - self.vehicle_cfg.length_m
        v_rel = foll_state.speed_mps - ego_state.speed_mps

        # ACC / IDM (Intelligent Driver Model) controller
        if self.follower_cfg.type == "acc":
            return self._idm_acceleration(ego_state, foll_state, gap)

        # Human driver model (Attentive or Distracted)
        # Check if lead vehicle has signaled or decelerated
        lead_braking = ego_state.brake_lights_on or ego_state.acceleration_mps2 < -0.5
        v2v_early = self.v2v_enabled and v2v_alert_active

        if (lead_braking or v2v_early) and self.perceived_brake_time is None:
            self.perceived_brake_time = sim_time

        # Effective reaction delay (reduced with V2V early warning)
        eff_delay = self.reaction_delay * 0.6 if (v2v_early and self.v2v_enabled) else self.reaction_delay

        if self.perceived_brake_time is not None:
            if (sim_time - self.perceived_brake_time) >= eff_delay:
                self.has_reacted = True

        if not self.has_reacted:
            # Maintain speed prior to reacting
            return 0.0

        # After reacting, apply braking
        a_brake_max = self.vehicle.a_brake_max
        # Normal moderate braking
        cmd_decel = self.follower_cfg.brake_fraction * a_brake_max

        # Panic / emergency escalation if TTC is critical or gap is dangerously small
        ttc = gap / max(0.01, v_rel) if v_rel > 0 else 999.0
        if ttc < 1.8 or gap < 10.0:
            cmd_decel = a_brake_max  # Maximum emergency braking

        return -cmd_decel

    def _idm_acceleration(self, ego: VehicleState, foll: VehicleState, gap: float) -> float:
        """Intelligent Driver Model car-following acceleration."""
        v0 = self.vehicle_cfg.initial_speed_kmh / 3.6
        s0 = 3.0   # minimum jam distance
        T = 1.4    # desired safe time headway
        a = 2.0    # comfortable acceleration
        b = 3.0    # comfortable deceleration
        delta = 4.0

        v = foll.speed_mps
        delta_v = foll.speed_mps - ego.speed_mps

        # Desired dynamic distance s*
        s_star = s0 + max(0.0, v * T + (v * delta_v) / (2.0 * ((a * b) ** 0.5)))
        gap_safe = max(0.1, gap)

        accel = a * (1.0 - (v / max(0.1, v0)) ** delta - (s_star / gap_safe) ** 2)
        return float(max(-self.vehicle.a_brake_max, min(self.vehicle.a_accel_max, accel)))

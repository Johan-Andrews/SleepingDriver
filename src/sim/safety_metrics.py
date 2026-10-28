"""Safety and collision risk metrics: Gap, Relative Speed, TTC, RSS Safe Distance, and Impact Speed."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional

from src.sim.vehicle import VehicleState


@dataclass
class SafetySnapshot:
    gap_m: float
    v_rel_mps: float
    ttc_s: float
    rss_safe_dist_m: float
    required_decel_mps2: float
    risk_level: str  # "SAFE", "CAUTION", "CRITICAL", "COLLISION"
    is_collision: bool
    impact_speed_mps: float


class SafetyMetricsCalculator:
    """Calculates active safety metrics between ego and following vehicles."""

    def __init__(
        self,
        vehicle_length_m: float = 4.5,
        rho_reaction_s: float = 1.0,
        a_accel_max: float = 2.5,
        b_min_braking: float = 4.0,
        b_max_braking: float = 7.84,  # mu=0.8 * g
    ) -> None:
        self.length = vehicle_length_m
        self.rho = rho_reaction_s
        self.a_accel = a_accel_max
        self.b_min = b_min_braking
        self.b_max = b_max_braking

        # Cumulative run tracking
        self.min_gap_m: float = 1e9
        self.has_collided: bool = False
        self.collision_time_s: Optional[float] = None
        self.impact_speed_mps: float = 0.0

    def reset(self) -> None:
        """Reset historical safety tracking."""
        self.min_gap_m = 1e9
        self.has_collided = False
        self.collision_time_s = None
        self.impact_speed_mps = 0.0

    def compute(
        self,
        ego: VehicleState,
        follower: VehicleState,
        sim_time_s: float = 0.0,
    ) -> SafetySnapshot:
        """Compute metrics for the current time step."""
        # Gap = x_lead - x_follow - length
        gap = ego.position_m - follower.position_m - self.length

        # Relative speed: positive means follower is closing in
        v_rel = follower.speed_mps - ego.speed_mps

        # Time to Collision (TTC)
        if gap <= 0.0:
            ttc = 0.0
        elif v_rel > 0.01:
            ttc = gap / v_rel
        else:
            ttc = float("inf")

        # Required deceleration for follower to avoid collision
        if gap > 0.01 and v_rel > 0.01:
            req_decel = (v_rel ** 2) / (2.0 * gap)
        else:
            req_decel = 0.0

        # RSS (Responsibility-Sensitive Safety) minimum safe distance
        # d_min = max(0, v_r*rho + 0.5*a_accel*rho^2 + (v_r + rho*a_accel)^2 / (2*b_min) - v_f^2 / (2*b_max))
        v_r = follower.speed_mps
        v_f = ego.speed_mps
        rho = self.rho

        term1 = v_r * rho
        term2 = 0.5 * self.a_accel * (rho ** 2)
        term3 = ((v_r + rho * self.a_accel) ** 2) / (2.0 * self.b_min)
        term4 = (v_f ** 2) / (2.0 * self.b_max)
        rss_dist = max(0.0, term1 + term2 + term3 - term4)

        # Collision detection
        is_collision = gap <= 0.0
        if is_collision and not self.has_collided:
            self.has_collided = True
            self.collision_time_s = sim_time_s
            self.impact_speed_mps = max(0.0, v_rel)

        if gap < self.min_gap_m:
            self.min_gap_m = gap

        # Risk classification
        if is_collision:
            risk = "COLLISION"
        elif ttc < 2.0 or gap < (rss_dist * 0.5):
            risk = "CRITICAL"
        elif ttc < 4.0 or gap < rss_dist:
            risk = "CAUTION"
        else:
            risk = "SAFE"

        return SafetySnapshot(
            gap_m=float(gap),
            v_rel_mps=float(v_rel),
            ttc_s=float(ttc),
            rss_safe_dist_m=float(rss_dist),
            required_decel_mps2=float(req_decel),
            risk_level=risk,
            is_collision=is_collision,
            impact_speed_mps=float(self.impact_speed_mps),
        )

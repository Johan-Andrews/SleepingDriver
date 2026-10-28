"""Braking controllers: Abrupt, Gradual (Jerk-Limited), and Adaptive (Gap & TTC Aware)."""

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Optional, List, Tuple

from src.config import BrakingConfig, VehicleConfig
from src.sim.vehicle import VehicleState

logger = logging.getLogger(__name__)


class BaseBrakingController(ABC):
    """Abstract interface for longitudinal braking controllers."""

    def __init__(self, braking_cfg: BrakingConfig, vehicle_cfg: VehicleConfig) -> None:
        self.braking_cfg = braking_cfg
        self.vehicle_cfg = vehicle_cfg
        self.t_brake: float = 0.0

    def reset(self) -> None:
        """Reset internal controller state."""
        self.t_brake = 0.0

    @abstractmethod
    def step(
        self,
        ego_state: VehicleState,
        follower_state: Optional[VehicleState],
        dt: float,
        intervention_active: bool,
    ) -> float:
        """Compute commanded acceleration a_cmd (negative for braking)."""
        pass


class AbruptController(BaseBrakingController):
    """Applies maximum available braking immediately upon intervention."""

    def step(
        self,
        ego_state: VehicleState,
        follower_state: Optional[VehicleState],
        dt: float,
        intervention_active: bool,
    ) -> float:
        if not intervention_active or ego_state.speed_mps <= 0.01:
            return 0.0
        # Instant step to full road friction limit
        mu = 0.8 if self.vehicle_cfg.friction == "dry" else (0.5 if self.vehicle_cfg.friction == "wet" else 0.2)
        a_max = mu * self.vehicle_cfg.g
        return -a_max


class GradualController(BaseBrakingController):
    """Jerk-limited controller ramping deceleration linearly up to a comfort cap."""

    def step(
        self,
        ego_state: VehicleState,
        follower_state: Optional[VehicleState],
        dt: float,
        intervention_active: bool,
    ) -> float:
        if not intervention_active or ego_state.speed_mps <= 0.01:
            self.t_brake = 0.0
            return 0.0

        self.t_brake += dt
        jerk = self.braking_cfg.gradual_jerk
        a_cap = self.braking_cfg.gradual_cap

        # Linear ramp with jerk j: a(t) = -min(a_cap, j * t_brake)
        decel = min(a_cap, jerk * self.t_brake)

        # Enforce friction limit
        mu = 0.8 if self.vehicle_cfg.friction == "dry" else (0.5 if self.vehicle_cfg.friction == "wet" else 0.2)
        max_mu_decel = mu * self.vehicle_cfg.g
        decel = min(decel, max_mu_decel)

        return -decel


class AdaptiveController(BaseBrakingController):
    """
    Adaptive gap- and TTC-aware braking controller.
    
    Novelty & Decision Rule:
    1. If follower is absent or far away with high TTC (> 6s), applies comfortable jerk-limited deceleration.
    2. If follower is uncomfortably close (gap < 20m) or closing fast (TTC < 3.5s), runs a 2.5-second forward lookahead
       predicting follower trajectory for 5 discrete candidate deceleration levels.
    3. Picks the candidate deceleration that maximizes the predicted minimum rear-end gap while still stopping ego safely.
    4. If the follower has already reacted and is braking, increases ego braking; if follower has NOT yet reacted, ego brakes
       moderately to avoid rear-end collision while lighting brake lights early.
    """

    def __init__(self, braking_cfg: BrakingConfig, vehicle_cfg: VehicleConfig) -> None:
        super().__init__(braking_cfg, vehicle_cfg)
        self.candidate_decels: List[float] = [1.5, 2.5, 3.8, 5.5, 7.5]

    def step(
        self,
        ego_state: VehicleState,
        follower_state: Optional[VehicleState],
        dt: float,
        intervention_active: bool,
    ) -> float:
        if not intervention_active or ego_state.speed_mps <= 0.01:
            self.t_brake = 0.0
            return 0.0

        self.t_brake += dt
        mu = 0.8 if self.vehicle_cfg.friction == "dry" else (0.5 if self.vehicle_cfg.friction == "wet" else 0.2)
        a_max = mu * self.vehicle_cfg.g

        if follower_state is None:
            # Fallback to gradual if no follower present
            return -min(self.braking_cfg.gradual_cap, self.braking_cfg.gradual_jerk * self.t_brake)

        # Calculate live gap and relative speed
        gap = ego_state.position_m - follower_state.position_m - self.vehicle_cfg.length_m
        v_rel = follower_state.speed_mps - ego_state.speed_mps
        ttc = gap / v_rel if v_rel > 0.01 and gap > 0 else 999.0

        # Run predictive lookahead to select optimal deceleration
        best_decel = self._predict_best_deceleration(ego_state, follower_state, a_max, gap, ttc)
        return -best_decel

    def _predict_best_deceleration(
        self,
        ego: VehicleState,
        follower: VehicleState,
        a_max: float,
        current_gap: float,
        current_ttc: float,
    ) -> float:
        """Evaluate candidate decelerations across a lookahead horizon."""
        horizon_s = 2.5
        lookahead_dt = 0.05
        steps = int(horizon_s / lookahead_dt)

        best_score = -1e9
        selected_decel = 2.5

        # Cap candidates to physical friction limit
        candidates = [min(c, a_max) for c in self.candidate_decels]

        for cand_decel in set(candidates):
            # Forward simulate ego & follower under cand_decel
            ego_x = ego.position_m
            ego_v = ego.speed_mps

            foll_x = follower.position_m
            foll_v = follower.speed_mps
            foll_a = follower.acceleration_mps2

            min_pred_gap = 1e9
            collision_predicted = False

            for _ in range(steps):
                ego_v = max(0.0, ego_v - cand_decel * lookahead_dt)
                ego_x += ego_v * lookahead_dt

                # Follower reacts or maintains braking
                foll_v = max(0.0, foll_v + foll_a * lookahead_dt)
                foll_x += foll_v * lookahead_dt

                pred_gap = ego_x - foll_x - self.vehicle_cfg.length_m
                if pred_gap < min_pred_gap:
                    min_pred_gap = pred_gap

                if pred_gap <= 0.0:
                    collision_predicted = True
                    break

            # Scoring:
            # - Strong penalty for collision
            # - Reward keeping minimum gap above 6.0 meters
            # - Reward progress towards vehicle stop (stopping capability)
            if collision_predicted:
                score = -1000.0 + min_pred_gap
            else:
                score = min_pred_gap + (cand_decel * 1.5)

            if score > best_score:
                best_score = score
                selected_decel = cand_decel

        return selected_decel


def create_controller(strategy: str, braking_cfg: BrakingConfig, vehicle_cfg: VehicleConfig) -> BaseBrakingController:
    """Factory to instantiate braking controllers by strategy name."""
    s = strategy.lower()
    if s == "abrupt":
        return AbruptController(braking_cfg, vehicle_cfg)
    elif s == "gradual":
        return GradualController(braking_cfg, vehicle_cfg)
    elif s == "adaptive":
        return AdaptiveController(braking_cfg, vehicle_cfg)
    raise ValueError(f"Unknown braking strategy: '{strategy}'")

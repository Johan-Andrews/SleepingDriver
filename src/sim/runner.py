"""Headless simulation runner for unit tests, parameter sweeps, and Monte Carlo runs."""

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
import numpy as np

from src.config import AppConfig, BrakingConfig, FollowerConfig, VehicleConfig
from src.decision.fusion import DrowsinessLevel, DrowsinessReading
from src.decision.state_machine import SafetyStateMachine, SystemState
from src.sim.controllers import create_controller
from src.sim.follower import FollowerVehicle
from src.sim.safety_metrics import SafetyMetricsCalculator, SafetySnapshot
from src.sim.scenarios import ScenarioPreset, PRESET_SCENARIOS
from src.sim.vehicle import Vehicle, VehicleState
from src.analysis.logger import LogStep

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    scenario_name: str
    strategy: str
    has_collided: bool
    collision_time_s: Optional[float]
    impact_speed_kmh: float
    min_gap_m: float
    stopping_distance_m: float
    stopping_time_s: float
    peak_decel_mps2: float
    peak_jerk_mps3: float
    time_series: List[LogStep]
    final_state: str


class HeadlessRunner:
    """Runs longitudinal simulation headlessly without GUI or video dependencies."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()

    def run(
        self,
        strategy: str = "adaptive",
        scenario: Optional[ScenarioPreset] = None,
        drowsiness_trigger_s: float = 1.0,
        max_duration_s: float = 15.0,
        v2v_enabled: bool = True,
        dt: float = 0.01,
    ) -> SimulationResult:
        """Run a single end-to-end scenario simulation."""
        # 1. Setup scenario parameters
        if scenario is None:
            scenario = PRESET_SCENARIOS["highway_close"]

        veh_cfg = VehicleConfig(
            initial_speed_kmh=scenario.ego_speed_kmh,
            friction=scenario.friction,
            g=self.config.vehicle.g,
        )
        brk_cfg = BrakingConfig(
            strategy=strategy,
            gradual_jerk=self.config.braking.gradual_jerk,
            gradual_cap=self.config.braking.gradual_cap,
            v2v_alert=v2v_enabled,
        )
        foll_cfg = FollowerConfig(
            type=scenario.follower_type,
            initial_gap_m=scenario.initial_gap_m,
            reaction_time_s=self.config.follower.reaction_time_s,
            brake_fraction=self.config.follower.brake_fraction,
        )

        ego = Vehicle(veh_cfg, initial_position_m=0.0)
        follower = FollowerVehicle(foll_cfg, veh_cfg, v2v_enabled=v2v_enabled)
        controller = create_controller(strategy, brk_cfg, veh_cfg)
        state_machine = SafetyStateMachine(self.config.alerts, self.config.detection)
        safety_calc = SafetyMetricsCalculator(vehicle_length_m=veh_cfg.length_m)

        sim_time = 0.0
        steps = int(max_duration_s / dt)
        logs: List[LogStep] = []

        peak_decel = 0.0
        peak_jerk = 0.0
        prev_accel = 0.0

        stopped_time: Optional[float] = None
        ego_start_pos = ego.position_m

        for step_idx in range(steps):
            sim_time = step_idx * dt

            # 2. Simulate driver drowsiness trigger
            is_drowsy = sim_time >= drowsiness_trigger_s
            reading = DrowsinessReading(
                score=85.0 if is_drowsy else 10.0,
                level=DrowsinessLevel.WARNING if is_drowsy else DrowsinessLevel.ALERT,
                eye_closed_s=3.5 if is_drowsy else 0.0,
                perclos=0.4 if is_drowsy else 0.0,
                yawn_rate=0.0,
                pitch=0.0,
                yaw=0.0,
                is_head_nodding=False,
                is_distracted=False,
                is_hard_rule_triggered=is_drowsy,
            )

            # Update State Machine
            state = state_machine.update(
                reading=reading,
                no_face_duration_s=0.0,
                vehicle_speed_mps=ego.speed_mps,
                manual_driver_response=False,
                timestamp=sim_time,
            )

            intervention_active = state in (SystemState.INTERVENTION, SystemState.SAFE_STOP)
            hazards_active = intervention_active
            v2v_alert_active = state in (SystemState.WARNING, SystemState.INTERVENTION)

            # 3. Compute Ego Deceleration
            ego_state = ego.get_state()
            foll_state = follower.vehicle.get_state()

            a_cmd = controller.step(ego_state, foll_state, dt, intervention_active)
            ego_state = ego.step(a_cmd, dt, hazards=hazards_active)

            # 4. Compute Follower Reaction
            foll_state = follower.step(ego_state, dt, sim_time, v2v_alert_active)

            # 5. Safety Metrics
            safety = safety_calc.compute(ego_state, foll_state, sim_time)

            # Jerk tracking
            jerk = abs(ego_state.acceleration_mps2 - prev_accel) / dt if dt > 0 else 0.0
            prev_accel = ego_state.acceleration_mps2
            if abs(ego_state.acceleration_mps2) > peak_decel:
                peak_decel = abs(ego_state.acceleration_mps2)
            if jerk > peak_jerk:
                peak_jerk = jerk

            # Log step
            log_record = LogStep(
                time_s=float(sim_time),
                ego_speed_kmh=float(ego_state.speed_kmh),
                ego_accel_mps2=float(ego_state.acceleration_mps2),
                ego_brake_intensity=float(ego_state.brake_intensity),
                follower_speed_kmh=float(foll_state.speed_kmh),
                gap_m=float(safety.gap_m),
                ttc_s=float(safety.ttc_s),
                state=state.value,
                drowsiness_score=float(reading.score),
                risk_level=safety.risk_level,
            )
            logs.append(log_record)

            # Check stop condition
            if ego_state.speed_mps <= 0.01 and stopped_time is None:
                stopped_time = sim_time

            # If both vehicles stopped or collision occurred and simulation can terminate
            if ego_state.speed_mps <= 0.01 and foll_state.speed_mps <= 0.01:
                break
            if safety.is_collision and (sim_time - (safety_calc.collision_time_s or 0.0)) > 1.0:
                break

        stopping_dist = max(0.0, ego.position_m - ego_start_pos)
        stopping_time = (stopped_time - drowsiness_trigger_s) if stopped_time else max_duration_s

        return SimulationResult(
            scenario_name=scenario.name,
            strategy=strategy,
            has_collided=safety_calc.has_collided,
            collision_time_s=safety_calc.collision_time_s,
            impact_speed_kmh=safety_calc.impact_speed_mps * 3.6,
            min_gap_m=float(safety_calc.min_gap_m),
            stopping_distance_m=float(stopping_dist),
            stopping_time_s=float(stopping_time),
            peak_decel_mps2=float(peak_decel),
            peak_jerk_mps3=float(peak_jerk),
            time_series=logs,
            final_state=state_machine.current_state.value,
        )

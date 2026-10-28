"""Batch experiment runner and Monte Carlo parameter sweeps."""

from __future__ import annotations
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.config import AppConfig
from src.sim.runner import HeadlessRunner, SimulationResult
from src.sim.scenarios import ScenarioPreset

logger = logging.getLogger(__name__)


@dataclass
class StrategySummary:
    strategy: str
    total_runs: int
    collision_count: int
    collision_rate: float
    mean_min_gap_m: float
    mean_impact_speed_kmh: float
    mean_stopping_dist_m: float
    mean_stopping_time_s: float
    mean_peak_jerk: float


@dataclass
class BatchExperimentReport:
    seed: int
    runs_per_strategy: int
    summaries: Dict[str, StrategySummary]
    results: List[SimulationResult]


class ExperimentManager:
    """Orchestrates reproducible Monte Carlo sweeps across controllers and parameters."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.runner = HeadlessRunner(self.config)

    def run_monte_carlo_sweep(
        self,
        n_samples: int = 40,
        seed: int = 42,
        v2v_enabled: bool = False,
    ) -> BatchExperimentReport:
        """Execute randomized Monte Carlo parameter sweeps across all three braking strategies."""
        rng = np.random.default_rng(seed)
        strategies = ["abrupt", "gradual", "adaptive"]
        all_results: List[SimulationResult] = []

        # Generate sample parameters
        speeds = rng.uniform(60.0, 130.0, n_samples)
        gaps = rng.uniform(10.0, 50.0, n_samples)
        frictions = rng.choice(["dry", "wet", "icy"], n_samples, p=[0.6, 0.3, 0.1])
        follower_types = rng.choice(["attentive", "distracted", "acc"], n_samples, p=[0.4, 0.4, 0.2])

        logger.info("Starting Monte Carlo sweep (%d samples per controller, seed=%d)", n_samples, seed)

        for i in range(n_samples):
            scenario = ScenarioPreset(
                name=f"MonteCarlo_Run_{i+1:03d}",
                ego_speed_kmh=float(speeds[i]),
                initial_gap_m=float(gaps[i]),
                friction=str(frictions[i]),
                follower_type=str(follower_types[i]),
                description="Randomized test trial",
            )

            for strat in strategies:
                res = self.runner.run(
                    strategy=strat,
                    scenario=scenario,
                    drowsiness_trigger_s=1.0,
                    v2v_enabled=v2v_enabled,
                )
                all_results.append(res)

        # Aggregate metrics per strategy
        summaries: Dict[str, StrategySummary] = {}
        for strat in strategies:
            strat_res = [r for r in all_results if r.strategy == strat]
            collisions = sum(1 for r in strat_res if r.has_collided)
            rate = collisions / len(strat_res) if strat_res else 0.0

            mean_gap = float(np.mean([r.min_gap_m for r in strat_res]))
            impact_speeds = [r.impact_speed_kmh for r in strat_res if r.has_collided]
            mean_impact = float(np.mean(impact_speeds)) if impact_speeds else 0.0
            mean_dist = float(np.mean([r.stopping_distance_m for r in strat_res]))
            mean_time = float(np.mean([r.stopping_time_s for r in strat_res]))
            mean_jerk = float(np.mean([r.peak_jerk_mps3 for r in strat_res]))

            summaries[strat] = StrategySummary(
                strategy=strat,
                total_runs=len(strat_res),
                collision_count=collisions,
                collision_rate=rate,
                mean_min_gap_m=mean_gap,
                mean_impact_speed_kmh=mean_impact,
                mean_stopping_dist_m=mean_dist,
                mean_stopping_time_s=mean_time,
                mean_peak_jerk=mean_jerk,
            )

        logger.info("Monte Carlo sweep finished.")
        return BatchExperimentReport(
            seed=seed,
            runs_per_strategy=n_samples,
            summaries=summaries,
            results=all_results,
        )

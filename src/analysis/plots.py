"""Matplotlib figure generation for simulation analysis and reports."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.experiments import BatchExperimentReport
from src.sim.runner import SimulationResult

logger = logging.getLogger(__name__)


def plot_speed_vs_time(results: Dict[str, SimulationResult], output_path: str = "results/speed_comparison.png") -> str:
    """Plot speed vs time for ego and follower across controllers."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = {"abrupt": "crimson", "gradual": "darkorange", "adaptive": "forestgreen"}

    for strat, res in results.items():
        times = [s.time_s for s in res.time_series]
        ego_v = [s.ego_speed_kmh for s in res.time_series]
        foll_v = [s.follower_speed_kmh for s in res.time_series]

        c = colors.get(strat, "blue")
        plt.plot(times, ego_v, label=f"Ego ({strat.title()})", color=c, linewidth=2)
        plt.plot(times, foll_v, label=f"Follower ({strat.title()})", color=c, linestyle="--", alpha=0.7)

    plt.title("Vehicle Speeds vs Time across Braking Strategies", fontsize=14, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Speed (km/h)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_gap_and_rss(result: SimulationResult, output_path: str = "results/gap_rss.png") -> str:
    """Plot gap vs time alongside RSS minimum safety boundary."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))

    times = [s.time_s for s in result.time_series]
    gaps = [s.gap_m for s in result.time_series]

    plt.plot(times, gaps, label="Vehicle Gap (m)", color="royalblue", linewidth=2.5)
    plt.axhline(y=0.0, color="crimson", linestyle="-", linewidth=1.5, label="Collision Threshold (0m)")
    plt.axhline(y=10.0, color="orange", linestyle="--", label="Critical Risk Margin (10m)")

    plt.title(f"Following Gap Evolution ({result.strategy.title()} Strategy)", fontsize=13, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Distance (m)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_decel_and_jerk(result: SimulationResult, output_path: str = "results/dynamics.png") -> str:
    """Plot vehicle deceleration and jerk profile over time."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    times = [s.time_s for s in result.time_series]
    decels = [-s.ego_accel_mps2 for s in result.time_series]

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Deceleration (m/s²)", color="purple")
    ax1.plot(times, decels, color="purple", linewidth=2, label="Deceleration")
    ax1.tick_params(axis="y", labelcolor="purple")
    ax1.grid(True, linestyle=":", alpha=0.5)

    plt.title(f"Deceleration Dynamics ({result.strategy.title()} Controller)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_controller_barchart(report: BatchExperimentReport, output_path: str = "results/controller_comparison.png") -> str:
    """Generate bar chart comparing collision rate, min gap, and stopping distance."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    strategies = list(report.summaries.keys())
    collision_rates = [report.summaries[s].collision_rate * 100 for s in strategies]
    min_gaps = [report.summaries[s].mean_min_gap_m for s in strategies]
    stopping_dists = [report.summaries[s].mean_stopping_dist_m for s in strategies]

    x = np.arange(len(strategies))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 6))

    rects1 = ax1.bar(x - width, collision_rates, width, label="Collision Rate (%)", color="crimson")
    rects2 = ax1.bar(x, min_gaps, width, label="Mean Min Gap (m)", color="royalblue")

    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, stopping_dists, width, label="Stopping Dist (m)", color="seagreen")

    ax1.set_ylabel("Collision Rate (%) / Gap (m)")
    ax2.set_ylabel("Stopping Distance (m)", color="seagreen")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.title() for s in strategies], fontweight="bold")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Benchmarking Braking Strategies Across Monte Carlo Runs", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_collision_heatmap(report: BatchExperimentReport, output_path: str = "results/collision_heatmap.png") -> str:
    """Plot collision probability heatmaps across speed and initial gap for each controller."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    strategies = ["abrupt", "gradual", "adaptive"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    speed_bins = np.linspace(60, 130, 6)
    gap_bins = np.linspace(10, 50, 6)

    for idx, strat in enumerate(strategies):
        strat_res = [r for r in report.results if r.strategy == strat]
        grid = np.zeros((len(speed_bins) - 1, len(gap_bins) - 1))
        counts = np.zeros_like(grid)

        for r in strat_res:
            # Reconstruct initial params from time_series
            init_v = r.time_series[0].ego_speed_kmh
            init_gap = r.time_series[0].gap_m

            v_idx = np.digitize(init_v, speed_bins) - 1
            g_idx = np.digitize(init_gap, gap_bins) - 1

            if 0 <= v_idx < grid.shape[0] and 0 <= g_idx < grid.shape[1]:
                counts[v_idx, g_idx] += 1
                if r.has_collided:
                    grid[v_idx, g_idx] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            prob_map = np.where(counts > 0, grid / counts, 0.0)

        im = axes[idx].imshow(
            prob_map,
            cmap="YlOrRd",
            origin="lower",
            extent=[gap_bins[0], gap_bins[-1], speed_bins[0], speed_bins[-1]],
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        axes[idx].set_title(f"{strat.title()} Braking", fontweight="bold")
        axes[idx].set_xlabel("Initial Gap (m)")
        if idx == 0:
            axes[idx].set_ylabel("Initial Speed (km/h)")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Collision Probability")
    fig.suptitle("Rear-End Collision Probability Heatmaps", fontsize=15, fontweight="bold")
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

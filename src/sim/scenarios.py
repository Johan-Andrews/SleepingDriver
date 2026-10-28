"""Scenario definitions and presets for simulation benchmarks."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ScenarioPreset:
    name: str
    ego_speed_kmh: float
    initial_gap_m: float
    friction: str          # "dry", "wet", "icy"
    follower_type: str     # "attentive", "distracted", "acc"
    description: str


PRESET_SCENARIOS: Dict[str, ScenarioPreset] = {
    "highway_close": ScenarioPreset(
        name="Highway, close follower",
        ego_speed_kmh=120.0,
        initial_gap_m=15.0,
        friction="dry",
        follower_type="attentive",
        description="High speed with tight following distance.",
    ),
    "highway_safe": ScenarioPreset(
        name="Highway, safe gap",
        ego_speed_kmh=120.0,
        initial_gap_m=60.0,
        friction="dry",
        follower_type="attentive",
        description="Standard highway cruising with recommended 2-second safe headway.",
    ),
    "city": ScenarioPreset(
        name="City",
        ego_speed_kmh=50.0,
        initial_gap_m=10.0,
        friction="dry",
        follower_type="attentive",
        description="Urban road with moderate speeds and close vehicle spacing.",
    ),
    "wet_road": ScenarioPreset(
        name="Wet road",
        ego_speed_kmh=90.0,
        initial_gap_m=30.0,
        friction="wet",
        follower_type="distracted",
        description="Reduced surface grip with a delayed-reaction follower.",
    ),
    "icy_road": ScenarioPreset(
        name="Icy road",
        ego_speed_kmh=60.0,
        initial_gap_m=40.0,
        friction="icy",
        follower_type="attentive",
        description="Low friction (mu=0.2) where stopping distances are substantially extended.",
    ),
    "distracted_tailgater": ScenarioPreset(
        name="Distracted tailgater",
        ego_speed_kmh=100.0,
        initial_gap_m=12.0,
        friction="dry",
        follower_type="distracted",
        description="Extreme rear-end collision hazard scenario testing adaptive mitigation.",
    ),
}


def get_scenario_names() -> List[str]:
    """Return all preset scenario names."""
    return list(PRESET_SCENARIOS.keys())


def load_scenario(name: str) -> ScenarioPreset:
    """Retrieve a preset scenario by key."""
    if name not in PRESET_SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {list(PRESET_SCENARIOS.keys())}")
    return PRESET_SCENARIOS[name]

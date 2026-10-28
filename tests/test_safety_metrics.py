"""Unit tests for safety metrics (gap, TTC, RSS distance, collision metrics)."""

import math
import pytest
from src.sim.safety_metrics import SafetyMetricsCalculator
from src.sim.vehicle import VehicleState


def make_vehicle_state(pos: float, speed_mps: float) -> VehicleState:
    return VehicleState(
        position_m=pos,
        speed_mps=speed_mps,
        speed_kmh=speed_mps * 3.6,
        acceleration_mps2=0.0,
        brake_lights_on=False,
        hazards_on=False,
        brake_intensity=0.0,
    )


def test_gap_and_ttc():
    calc = SafetyMetricsCalculator(vehicle_length_m=4.5)
    ego = make_vehicle_state(pos=100.0, speed_mps=20.0)
    follower = make_vehicle_state(pos=65.5, speed_mps=25.0)

    # gap = 100.0 - 65.5 - 4.5 = 30.0 m
    # v_rel = 25.0 - 20.0 = 5.0 m/s
    # ttc = 30.0 / 5.0 = 6.0 s
    snap = calc.compute(ego, follower)

    assert pytest.approx(snap.gap_m) == 30.0
    assert pytest.approx(snap.v_rel_mps) == 5.0
    assert pytest.approx(snap.ttc_s) == 6.0
    assert not snap.is_collision


def test_collision_detected_when_gap_non_positive():
    calc = SafetyMetricsCalculator(vehicle_length_m=4.5)
    ego = make_vehicle_state(pos=50.0, speed_mps=10.0)
    # follower penetrates ego vehicle zone
    follower = make_vehicle_state(pos=46.0, speed_mps=18.0)

    # gap = 50.0 - 46.0 - 4.5 = -0.5 m
    snap = calc.compute(ego, follower, sim_time_s=4.2)

    assert snap.is_collision
    assert snap.risk_level == "COLLISION"
    assert calc.has_collided
    assert pytest.approx(calc.collision_time_s) == 4.2
    assert pytest.approx(calc.impact_speed_mps) == 8.0  # 18.0 - 10.0


def test_rss_safe_distance_hand_calculated():
    # Parameters: v_r = 20, v_f = 20, rho = 1.0, a_accel = 2.0, b_min = 4.0, b_max = 8.0
    calc = SafetyMetricsCalculator(
        vehicle_length_m=4.5,
        rho_reaction_s=1.0,
        a_accel_max=2.0,
        b_min_braking=4.0,
        b_max_braking=8.0,
    )
    ego = make_vehicle_state(pos=100.0, speed_mps=20.0)
    follower = make_vehicle_state(pos=40.0, speed_mps=20.0)

    # term1: v_r * rho = 20 * 1 = 20
    # term2: 0.5 * a_accel * rho^2 = 0.5 * 2 * 1 = 1
    # term3: (v_r + rho*a_accel)^2 / (2 * b_min) = (20 + 2)^2 / (2 * 4) = 484 / 8 = 60.5
    # term4: v_f^2 / (2 * b_max) = 400 / 16 = 25.0
    # d_min = 20 + 1 + 60.5 - 25.0 = 56.5 m
    expected_rss = 20.0 + 1.0 + 60.5 - 25.0

    snap = calc.compute(ego, follower)
    assert pytest.approx(snap.rss_safe_dist_m) == expected_rss

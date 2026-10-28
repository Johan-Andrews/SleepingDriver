"""Unit tests for vehicle dynamics and stopping distance verification."""

import pytest
from src.config import VehicleConfig
from src.sim.vehicle import Vehicle


def test_vehicle_stopping_distance_analytical_match():
    # 100 km/h = 27.7778 m/s
    cfg = VehicleConfig(initial_speed_kmh=100.0, friction="dry", g=9.81)
    veh = Vehicle(cfg, initial_position_m=0.0)

    v0 = 100.0 / 3.6
    a_cmd = -4.0  # Deceleration of 4.0 m/s^2
    dt = 0.01

    # Theoretical stopping distance: s = v0^2 / (2 * a)
    expected_s = (v0 ** 2) / (2.0 * abs(a_cmd))
    expected_time = v0 / abs(a_cmd)

    total_time = 0.0
    while veh.speed_mps > 1e-4 and total_time < expected_time + 1.0:
        veh.step(a_cmd, dt)
        total_time += dt

    # Stopping distance should match within 0.05 meters
    assert pytest.approx(veh.position_m, abs=0.05) == expected_s
    assert veh.speed_mps == 0.0


def test_friction_clamping():
    # On icy road (mu = 0.2, g = 9.81 -> max decel = 1.962 m/s^2)
    cfg = VehicleConfig(initial_speed_kmh=50.0, friction="icy", g=9.81)
    veh = Vehicle(cfg)

    # Command an aggressive deceleration (-8.0 m/s^2)
    state = veh.step(-8.0, dt=0.01)

    max_allowed_decel = 0.2 * 9.81
    assert abs(state.acceleration_mps2) <= max_allowed_decel + 1e-3
    assert state.brake_intensity == 1.0
    assert state.brake_lights_on

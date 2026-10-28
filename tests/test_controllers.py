"""Unit tests for Abrupt, Gradual, and Adaptive braking controllers."""

import pytest
from src.config import BrakingConfig, VehicleConfig
from src.sim.controllers import AbruptController, GradualController, AdaptiveController
from src.sim.vehicle import Vehicle, VehicleState


def test_abrupt_controller_steps_to_max():
    veh_cfg = VehicleConfig(friction="dry", g=9.81)
    brk_cfg = BrakingConfig(strategy="abrupt")
    ctrl = AbruptController(brk_cfg, veh_cfg)

    ego = VehicleState(10.0, 20.0, 72.0, 0.0, False, False, 0.0)
    # Not active: returns 0.0
    assert ctrl.step(ego, None, dt=0.01, intervention_active=False) == 0.0

    # Active: immediate full braking
    a_cmd = ctrl.step(ego, None, dt=0.01, intervention_active=True)
    assert pytest.approx(a_cmd) == -(0.8 * 9.81)


def test_gradual_controller_ramps_with_jerk():
    veh_cfg = VehicleConfig(friction="dry", g=9.81)
    brk_cfg = BrakingConfig(strategy="gradual", gradual_jerk=2.0, gradual_cap=4.0)
    ctrl = GradualController(brk_cfg, veh_cfg)

    ego = VehicleState(10.0, 20.0, 72.0, 0.0, False, False, 0.0)

    # After 1.0 second: jerk * t = 2.0 * 1.0 = 2.0 m/s^2
    a_1s = 0.0
    for _ in range(100):
        a_1s = ctrl.step(ego, None, dt=0.01, intervention_active=True)
    assert pytest.approx(a_1s, abs=0.05) == -2.0

    # After 3.0 seconds: capped at 4.0 m/s^2
    for _ in range(200):
        a_3s = ctrl.step(ego, None, dt=0.01, intervention_active=True)
    assert pytest.approx(a_3s, abs=0.05) == -4.0


def test_adaptive_controller_never_exceeds_friction_and_brakes():
    veh_cfg = VehicleConfig(friction="dry", g=9.81)
    brk_cfg = BrakingConfig(strategy="adaptive")
    ctrl = AdaptiveController(brk_cfg, veh_cfg)

    ego = VehicleState(100.0, 25.0, 90.0, 0.0, False, False, 0.0)
    foll = VehicleState(60.0, 25.0, 90.0, 0.0, False, False, 0.0)

    a_cmd = ctrl.step(ego, foll, dt=0.01, intervention_active=True)
    max_allowed = 0.8 * 9.81

    assert a_cmd <= 0.0
    assert abs(a_cmd) <= max_allowed + 1e-3

"""Unit tests for the safety state machine transitions and escalation."""

import pytest
from src.config import AlertsConfig, DetectionConfig
from src.decision.fusion import DrowsinessReading, DrowsinessLevel
from src.decision.state_machine import SafetyStateMachine, SystemState


def create_reading(level: DrowsinessLevel, hard_rule: bool = False) -> DrowsinessReading:
    return DrowsinessReading(
        score=75.0 if level == DrowsinessLevel.WARNING else (40.0 if level == DrowsinessLevel.CAUTION else 10.0),
        level=level,
        eye_closed_s=3.2 if hard_rule else 0.0,
        perclos=0.1,
        yawn_rate=0.0,
        pitch=0.0,
        yaw=0.0,
        is_head_nodding=False,
        is_distracted=False,
        is_hard_rule_triggered=hard_rule,
    )


def test_transition_alert_to_warning_to_intervention_to_safe_stop():
    alerts_cfg = AlertsConfig(response_interval_s=3.0, recovery_hold_s=1.0)
    det_cfg = DetectionConfig(no_face_timeout_s=2.0)
    sm = SafetyStateMachine(alerts_cfg, det_cfg)

    t = 100.0
    # 1. Starts in ALERT
    assert sm.current_state == SystemState.ALERT

    # 2. Eye closure triggers WARNING
    warn_reading = create_reading(DrowsinessLevel.WARNING, hard_rule=True)
    state = sm.update(warn_reading, no_face_duration_s=0.0, vehicle_speed_mps=25.0, manual_driver_response=False, timestamp=t)
    assert state == SystemState.WARNING

    # 3. Warning window persists past response_interval_s (3.0s) -> INTERVENTION
    t += 3.5
    state = sm.update(warn_reading, no_face_duration_s=0.0, vehicle_speed_mps=20.0, manual_driver_response=False, timestamp=t)
    assert state == SystemState.INTERVENTION

    # 4. Vehicle brakes to standstill (0 m/s) -> SAFE_STOP
    t += 5.0
    state = sm.update(warn_reading, no_face_duration_s=0.0, vehicle_speed_mps=0.0, manual_driver_response=False, timestamp=t)
    assert state == SystemState.SAFE_STOP

    # 5. Driver presses response key -> RECOVERY
    t += 1.0
    state = sm.update(warn_reading, no_face_duration_s=0.0, vehicle_speed_mps=0.0, manual_driver_response=True, timestamp=t)
    assert state == SystemState.RECOVERY

    # 6. Recovery hold time finishes -> ALERT
    t += 1.5
    alert_reading = create_reading(DrowsinessLevel.ALERT)
    state = sm.update(alert_reading, no_face_duration_s=0.0, vehicle_speed_mps=0.0, manual_driver_response=False, timestamp=t)
    assert state == SystemState.ALERT


def test_no_face_triggers_caution_not_intervention():
    alerts_cfg = AlertsConfig(response_interval_s=3.0)
    det_cfg = DetectionConfig(no_face_timeout_s=2.0)
    sm = SafetyStateMachine(alerts_cfg, det_cfg)

    alert_reading = create_reading(DrowsinessLevel.ALERT)
    t = 10.0

    # Face disappears for 2.5s (exceeds 2.0s timeout)
    state = sm.update(alert_reading, no_face_duration_s=2.5, vehicle_speed_mps=25.0, manual_driver_response=False, timestamp=t)
    assert state == SystemState.CAUTION

    # Remains in CAUTION, does not jump to WARNING or INTERVENTION
    t += 3.0
    state = sm.update(alert_reading, no_face_duration_s=5.5, vehicle_speed_mps=25.0, manual_driver_response=False, timestamp=t)
    assert state == SystemState.CAUTION

    # Face returns
    t += 0.5
    state = sm.update(alert_reading, no_face_duration_s=0.0, vehicle_speed_mps=25.0, manual_driver_response=False, timestamp=t)
    assert state == SystemState.ALERT


def test_warning_recovered_by_driver_response():
    alerts_cfg = AlertsConfig(response_interval_s=5.0, recovery_hold_s=1.0)
    det_cfg = DetectionConfig()
    sm = SafetyStateMachine(alerts_cfg, det_cfg)

    t = 0.0
    warn_reading = create_reading(DrowsinessLevel.WARNING)
    sm.update(warn_reading, no_face_duration_s=0.0, vehicle_speed_mps=20.0, manual_driver_response=False, timestamp=t)
    assert sm.current_state == SystemState.WARNING

    # Driver responds within warning interval
    t += 1.5
    sm.update(warn_reading, no_face_duration_s=0.0, vehicle_speed_mps=20.0, manual_driver_response=True, timestamp=t)
    assert sm.current_state == SystemState.RECOVERY

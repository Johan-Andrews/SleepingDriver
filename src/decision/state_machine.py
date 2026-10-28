"""Escalating driver safety state machine."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import List, Optional

from src.config import AlertsConfig, DetectionConfig
from src.decision.fusion import DrowsinessReading, DrowsinessLevel

logger = logging.getLogger(__name__)


class SystemState(str, Enum):
    ALERT = "ALERT"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    INTERVENTION = "INTERVENTION"
    SAFE_STOP = "SAFE_STOP"
    RECOVERY = "RECOVERY"


@dataclass
class TransitionEvent:
    timestamp: float
    from_state: SystemState
    to_state: SystemState
    reason: str


class SafetyStateMachine:
    """Manages safety escalation states and initiates braking intervention."""

    def __init__(self, alerts_cfg: AlertsConfig, det_cfg: DetectionConfig) -> None:
        self.alerts_cfg = alerts_cfg
        self.det_cfg = det_cfg

        self.current_state = SystemState.ALERT
        self.state_enter_time: float = time.perf_counter()

        # Timers
        self.warning_timer: float = 0.0
        self.recovery_timer: float = 0.0
        self.time_in_state: float = 0.0

        # Event logging
        self.event_log: List[TransitionEvent] = []

    def _transition(self, to_state: SystemState, reason: str, timestamp: float) -> None:
        """Execute and record state transition."""
        if to_state == self.current_state:
            return

        event = TransitionEvent(
            timestamp=timestamp,
            from_state=self.current_state,
            to_state=to_state,
            reason=reason,
        )
        self.event_log.append(event)
        logger.info(
            "State transition: %s -> %s (Reason: %s, at t=%.2f)",
            self.current_state.value,
            to_state.value,
            reason,
            timestamp,
        )
        self.current_state = to_state
        self.state_enter_time = timestamp

    def update(
        self,
        reading: DrowsinessReading,
        no_face_duration_s: float,
        vehicle_speed_mps: float,
        manual_driver_response: bool,
        timestamp: Optional[float] = None,
    ) -> SystemState:
        """Process cycle update and advance state machine."""
        now = time.perf_counter() if timestamp is None else timestamp
        self.time_in_state = now - self.state_enter_time

        # Check driver recovery conditions
        is_alert_reading = reading.level == DrowsinessLevel.ALERT and no_face_duration_s < 0.2
        if is_alert_reading:
            self.recovery_timer += 0.03  # approximate increment if timestamp difference not passed
        else:
            self.recovery_timer = 0.0

        recovery_confirmed = manual_driver_response or (self.recovery_timer >= self.alerts_cfg.recovery_hold_s)

        if self.current_state == SystemState.ALERT:
            if reading.level == DrowsinessLevel.WARNING:
                self._transition(SystemState.WARNING, "Drowsiness Warning threshold crossed", now)
            elif no_face_duration_s >= self.det_cfg.no_face_timeout_s:
                self._transition(SystemState.CAUTION, "Driver not visible", now)
            elif reading.level == DrowsinessLevel.CAUTION:
                self._transition(SystemState.CAUTION, "Caution score threshold crossed", now)

        elif self.current_state == SystemState.CAUTION:
            if reading.level == DrowsinessLevel.WARNING:
                self._transition(SystemState.WARNING, "Escalated to Warning", now)
            elif reading.level == DrowsinessLevel.ALERT and no_face_duration_s < self.det_cfg.no_face_timeout_s:
                self._transition(SystemState.ALERT, "Driver alert restored", now)

        elif self.current_state == SystemState.WARNING:
            if recovery_confirmed:
                self._transition(SystemState.RECOVERY, "Driver confirmed attentive", now)
            elif self.time_in_state >= self.alerts_cfg.response_interval_s:
                self._transition(SystemState.INTERVENTION, "Unresponsive after warning interval", now)

        elif self.current_state == SystemState.INTERVENTION:
            if recovery_confirmed:
                self._transition(SystemState.RECOVERY, "Driver took control during intervention", now)
            elif vehicle_speed_mps <= 0.05:
                self._transition(SystemState.SAFE_STOP, "Vehicle brought to safe stop", now)

        elif self.current_state == SystemState.SAFE_STOP:
            if recovery_confirmed:
                self._transition(SystemState.RECOVERY, "Driver resumed control after safe stop", now)

        elif self.current_state == SystemState.RECOVERY:
            if self.time_in_state >= self.alerts_cfg.recovery_hold_s:
                self._transition(SystemState.ALERT, "Control handover complete", now)

        return self.current_state

    @property
    def warning_urgency(self) -> float:
        """Escalating urgency from 0.0 to 1.0 while in WARNING or INTERVENTION."""
        if self.current_state == SystemState.WARNING:
            return min(1.0, max(0.0, self.time_in_state / self.alerts_cfg.response_interval_s))
        elif self.current_state in (SystemState.INTERVENTION, SystemState.SAFE_STOP):
            return 1.0
        return 0.0

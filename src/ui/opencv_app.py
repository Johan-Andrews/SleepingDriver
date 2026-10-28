"""Lightweight OpenCV-native viewer showing live detection and vehicle telemetry."""

from __future__ import annotations
import logging
import sys
import time
from typing import Optional
import cv2
import numpy as np

from src.config import AppConfig
from src.decision.alerts import AlertManager
from src.decision.fusion import DrowsinessFuser
from src.decision.state_machine import SafetyStateMachine, SystemState
from src.sim.controllers import create_controller
from src.sim.follower import FollowerVehicle
from src.sim.safety_metrics import SafetyMetricsCalculator
from src.sim.scenarios import PRESET_SCENARIOS
from src.sim.vehicle import Vehicle
from src.vision.camera import ThreadedCamera
from src.vision.calibration import Calibrator
from src.vision.face_tracker import FaceTracker
from src.vision.features import FeatureExtractor, calculate_ear, calculate_mar, calculate_head_pose
from src.vision.video_source import VideoSource

logger = logging.getLogger(__name__)


def run_opencv_app(config: AppConfig) -> int:
    """Run interactive viewer using standard OpenCV cv2.imshow window."""
    print("\nStarting OpenCV Native Viewer...")

    video_source = VideoSource(
        source_type=config.camera.source,
        index=config.camera.index,
        file_path=config.camera.video_path,
        width=config.camera.width,
        height=config.camera.height,
    )
    camera = ThreadedCamera(video_source, target_fps=30.0).start()
    tracker = FaceTracker()
    calibrator = Calibrator(config.detection, calibration_duration_s=5.0)
    calibrator.start()
    extractor = FeatureExtractor(config.detection)

    fuser = DrowsinessFuser(config.fusion, config.detection)
    state_machine = SafetyStateMachine(config.alerts, config.detection)
    alert_mgr = AlertManager()

    strategy = config.braking.strategy
    ego = Vehicle(config.vehicle, initial_position_m=0.0)
    follower = FollowerVehicle(config.follower, config.vehicle, v2v_enabled=config.braking.v2v_alert)
    controller = create_controller(strategy, config.braking, config.vehicle)
    safety_calc = SafetyMetricsCalculator(vehicle_length_m=config.vehicle.length_m)

    window_name = "Driver Drowsiness & Adaptive Braking [OpenCV Window]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 680)

    sim_time = 0.0
    dt = 0.033

    print("=" * 60)
    print("  OpenCV Window is now active!")
    print("  Press 'D' to toggle simulated drowsiness.")
    print("  Press 'SPACE' to confirm alert / steering touch.")
    print("  Press 'Q' or 'ESC' to quit.")
    print("=" * 60)

    try:
        while True:
            ret, frame, latency = camera.read()
            if not ret or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for camera...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            tracker_res = tracker.process_frame(frame)
            if tracker_res and tracker_res.face_detected and tracker_res.landmarks is not None:
                lms = tracker_res.landmarks
                if not calibrator.is_complete:
                    _, _, ear = calculate_ear(lms)
                    mar = calculate_mar(lms)
                    p, y, r = calculate_head_pose(lms, frame.shape[1], frame.shape[0])
                    if calibrator.update(ear, mar, p, y, r, True):
                        extractor.set_profile(calibrator.get_profile())

                features = extractor.update(lms, True, frame.shape[1], frame.shape[0])
                # Draw facial landmarks
                for pt in lms[::4]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
            else:
                features = extractor.update(None, False, frame.shape[1], frame.shape[0])

            reading = fuser.process(features)
            state = state_machine.update(
                reading=reading,
                no_face_duration_s=features.no_face_duration_s,
                vehicle_speed_mps=ego.speed_mps,
                manual_driver_response=False,
            )
            alert_mgr.update(state, state_machine.warning_urgency)

            # Step physics
            sim_time += dt
            intervention_active = state in (SystemState.INTERVENTION, SystemState.SAFE_STOP)
            v2v_active = state in (SystemState.WARNING, SystemState.INTERVENTION)

            ego_st = ego.get_state()
            foll_st = follower.vehicle.get_state()
            a_cmd = controller.step(ego_st, foll_st, dt, intervention_active)
            ego_st = ego.step(a_cmd, dt, hazards=intervention_active)
            foll_st = follower.step(ego_st, dt, sim_time, v2v_alert_active=v2v_active)
            safety = safety_calc.compute(ego_st, foll_st, sim_time)

            # Build UI Canvas (Side-by-side: Camera + Telemetry)
            canvas = np.zeros((680, 1100, 3), dtype=np.uint8)
            # Paste camera top-left
            cam_h, cam_w = frame.shape[:2]
            scale = min(480 / cam_h, 600 / cam_w)
            cam_resized = cv2.resize(frame, (int(cam_w * scale), int(cam_h * scale)))
            rh, rw = cam_resized.shape[:2]
            canvas[20 : 20 + rh, 20 : 20 + rw] = cam_resized
            cv2.rectangle(canvas, (20, 20), (20 + rw, 20 + rh), (100, 100, 100), 2)

            # Draw State Banner top-right
            state_colors = {
                SystemState.ALERT: (0, 200, 0),
                SystemState.CAUTION: (0, 200, 255),
                SystemState.WARNING: (0, 120, 255),
                SystemState.INTERVENTION: (0, 0, 255),
                SystemState.SAFE_STOP: (0, 0, 180),
                SystemState.RECOVERY: (255, 200, 0),
            }
            col = state_colors.get(state, (200, 200, 200))
            cv2.rectangle(canvas, (640, 20), (1070, 90), col, -1)
            cv2.putText(canvas, state.value, (660, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

            # Telemetry text
            info_y = 130
            lines = [
                f"Drowsiness Score: {reading.score:.1f} / 100",
                f"Eyes Status: {'CLOSED' if features.is_eye_closed else 'OPEN'} (Closed: {features.eye_closed_duration_s:.1f}s)",
                f"EAR: {features.ear_smooth:.2f} | PERCLOS: {features.perclos * 100:.1f}%",
                f"Head Pitch: {features.pitch_rel:+.1f} deg | Yawns: {features.yawn_count}",
                "-" * 35,
                f"Vehicle Speed: {ego_st.speed_kmh:.1f} km/h",
                f"Braking Intensity: {ego_st.brake_intensity * 100:.0f}% ({strategy.upper()})",
                f"Following Gap: {safety.gap_m:.1f} m",
                f"Collision Risk: {safety.risk_level}",
                f"Collision Occurred: {'YES' if safety_calc.has_collided else 'NO'}",
            ]
            for l in lines:
                cv2.putText(canvas, l, (640, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
                info_y += 30

            # 2D road lane representation at bottom
            cv2.rectangle(canvas, (20, 520), (1070, 650), (45, 45, 45), -1)
            cv2.line(canvas, (20, 585), (1070, 585), (200, 200, 200), 2)
            # Ego car
            ego_px = 750
            cv2.rectangle(canvas, (ego_px, 565), (ego_px + 70, 605), (220, 100, 50), -1)
            cv2.putText(canvas, "EGO", (ego_px + 12, 592), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if ego_st.brake_lights_on:
                cv2.circle(canvas, (ego_px - 4, 572), 6, (0, 0, 255), -1)
                cv2.circle(canvas, (ego_px - 4, 598), 6, (0, 0, 255), -1)

            # Follower car
            foll_px = int(ego_px - 70 - (safety.gap_m * 6.0))
            if foll_px > 30:
                cv2.rectangle(canvas, (foll_px, 565), (foll_px + 70, 605), (130, 130, 130), -1)
                cv2.putText(canvas, "FOLLOWER", (foll_px + 2, 592), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Bottom Controls bar
            cv2.putText(canvas, "Keys: [D] Simulate Drowsiness | [SPACE] Wake Up | [1,2,3] Braking Strategy | [Q] Quit", (30, 672), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('d'), ord('D')):
                fuser.set_simulated_drowsiness(not fuser._simulated_drowsy)
            elif key == 32:  # SPACE
                state_machine.update(reading, 0.0, ego.speed_mps, manual_driver_response=True)
            elif key == ord('1'):
                strategy = "abrupt"
                controller = create_controller(strategy, config.braking, config.vehicle)
            elif key == ord('2'):
                strategy = "gradual"
                controller = create_controller(strategy, config.braking, config.vehicle)
            elif key == ord('3'):
                strategy = "adaptive"
                controller = create_controller(strategy, config.braking, config.vehicle)

    finally:
        camera.stop()
        tracker.close()
        cv2.destroyAllWindows()

    return 0

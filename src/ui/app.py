"""Pygame main application integrating real-time computer vision, simulation, and dashboard."""

from __future__ import annotations
import logging
import sys
import time
from typing import Optional
import pygame

from src.config import AppConfig
from src.decision.alerts import AlertManager
from src.decision.fusion import DrowsinessFuser
from src.decision.state_machine import SafetyStateMachine, SystemState
from src.sim.controllers import create_controller
from src.sim.follower import FollowerVehicle
from src.sim.safety_metrics import SafetyMetricsCalculator
from src.sim.scenarios import PRESET_SCENARIOS, ScenarioPreset
from src.sim.vehicle import Vehicle
from src.ui.panels import (
    DriverMonitoringPanel,
    AlertPanel,
    VehicleDashboardPanel,
    RoadSimulationView,
    PerformanceAnalysisPanel,
    BG_COLOR,
)
from src.vision.camera import ThreadedCamera
from src.vision.calibration import Calibrator
from src.vision.face_tracker import FaceTracker
from src.vision.features import FeatureExtractor
from src.vision.video_source import VideoSource

logger = logging.getLogger(__name__)


class DrowsinessSimulationApp:
    """Main interactive Pygame desktop application."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.width = 1240
        self.height = 720

        import os
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        pygame.init()
        pygame.display.set_caption("Driver Drowsiness Detection & Adaptive Braking Simulation")
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        # Force window to foreground on Windows
        if sys.platform == "win32":
            try:
                import ctypes
                wm_info = pygame.display.get_wm_info()
                hwnd = wm_info.get("window")
                if hwnd:
                    ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception as e:
                logger.debug("Could not bring window to foreground: %s", e)

        print("\n" + "=" * 70)
        print("  >> [SUCCESS] Window is now open on your screen!")
        print("  >> Check your Windows taskbar if it is hidden behind your terminal.")
        print("  >> Press 'D' on keyboard to simulate falling asleep.")
        print("  >> Press 'SPACE' to confirm alert / touch steering wheel.")
        print("  >> Press 'ESC' or 'Q' to close.")
        print("=" * 70 + "\n")

        # Fonts
        self.font_small = pygame.font.SysFont("Segoe UI", 13)
        self.font_med = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI", 26, bold=True)

        # 1. Vision Components
        self.video_source = VideoSource(
            source_type=config.camera.source,
            index=config.camera.index,
            file_path=config.camera.video_path,
            width=config.camera.width,
            height=config.camera.height,
        )
        self.camera = ThreadedCamera(self.video_source, target_fps=30.0).start()
        self.tracker = FaceTracker()
        self.calibrator = Calibrator(config.detection, calibration_duration_s=5.0)
        self.calibrator.start()
        self.extractor = FeatureExtractor(config.detection)

        # 2. Decision Components
        self.fuser = DrowsinessFuser(config.fusion, config.detection)
        self.state_machine = SafetyStateMachine(config.alerts, config.detection)
        self.alert_manager = AlertManager()

        # 3. Simulation Components
        self.scenario: ScenarioPreset = PRESET_SCENARIOS["highway_close"]
        self.active_strategy = config.braking.strategy
        self.is_paused = False
        self.sim_time_s = 0.0

        self.ego = Vehicle(config.vehicle, initial_position_m=0.0)
        self.follower = FollowerVehicle(config.follower, config.vehicle, v2v_enabled=config.braking.v2v_alert)
        self.controller = create_controller(self.active_strategy, config.braking, config.vehicle)
        self.safety_calc = SafetyMetricsCalculator(vehicle_length_m=config.vehicle.length_m)

        # 4. UI Panels
        self.monitor_panel = DriverMonitoringPanel(pygame.Rect(15, 15, 410, 230), self.font_small, self.font_med)
        self.alert_panel = AlertPanel(pygame.Rect(435, 15, 410, 230), self.font_small, self.font_med, self.font_large)
        self.dashboard_panel = VehicleDashboardPanel(pygame.Rect(855, 15, 410, 230), self.font_small, self.font_med, self.font_large)
        self.road_view = RoadSimulationView(pygame.Rect(15, 255, 830, 485), self.font_small, self.font_med)
        self.perf_panel = PerformanceAnalysisPanel(pygame.Rect(855, 255, 410, 485), self.font_small, self.font_med)

        self.running = True

    def reset_simulation(self) -> None:
        """Reset vehicles, state machine, and controller."""
        self.ego.reset(0.0)
        self.follower.reset()
        self.controller.reset()
        self.safety_calc.reset()
        self.state_machine = SafetyStateMachine(self.config.alerts, self.config.detection)
        self.sim_time_s = 0.0
        logger.info("Simulation state reset.")

    def set_strategy(self, strategy: str) -> None:
        """Change active braking controller strategy."""
        self.active_strategy = strategy
        self.config.braking.strategy = strategy
        self.controller = create_controller(strategy, self.config.braking, self.config.vehicle)
        logger.info("Switched braking controller to: %s", strategy)

    def run(self) -> None:
        """Main application execution loop."""
        logger.info("Starting Drowsiness Simulation App main loop.")
        dt = 0.016  # Target 60 FPS update

        while self.running:
            frame_start = time.perf_counter()

            # Handle Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self.running = False
                    elif event.key == pygame.K_d:
                        # Toggle simulated drowsiness
                        curr = self.fuser._simulated_drowsy
                        self.fuser.set_simulated_drowsiness(not curr)
                    elif event.key == pygame.K_SPACE:
                        # Driver manual response (steering wheel touch)
                        self.state_machine.update(
                            self.fuser.process(self.extractor.update(None, False, 640, 480)),
                            no_face_duration_s=0.0,
                            vehicle_speed_mps=self.ego.speed_mps,
                            manual_driver_response=True,
                        )
                    elif event.key == pygame.K_p:
                        self.is_paused = not self.is_paused
                    elif event.key == pygame.K_r:
                        self.reset_simulation()
                    elif event.key == pygame.K_l:
                        self.monitor_panel.show_landmarks = not self.monitor_panel.show_landmarks
                    elif event.key == pygame.K_1:
                        self.set_strategy("abrupt")
                    elif event.key == pygame.K_2:
                        self.set_strategy("gradual")
                    elif event.key == pygame.K_3:
                        self.set_strategy("adaptive")

            # 1. Capture & Vision Processing
            ret, frame_bgr, latency = self.camera.read()
            tracker_res = self.tracker.process_frame(frame_bgr) if (ret and frame_bgr is not None) else None

            # Calibration or Feature extraction
            if tracker_res and tracker_res.face_detected and tracker_res.landmarks is not None:
                if not self.calibrator.is_complete:
                    from src.vision.features import calculate_ear, calculate_mar, calculate_head_pose
                    _, _, ear = calculate_ear(tracker_res.landmarks)
                    mar = calculate_mar(tracker_res.landmarks)
                    pitch, yaw, roll = calculate_head_pose(tracker_res.landmarks, tracker_res.frame_width, tracker_res.frame_height)
                    if self.calibrator.update(ear, mar, pitch, yaw, roll, face_detected=True):
                        self.extractor.set_profile(self.calibrator.get_profile())

                features = self.extractor.update(
                    landmarks=tracker_res.landmarks,
                    face_detected=True,
                    frame_width=tracker_res.frame_width,
                    frame_height=tracker_res.frame_height,
                )
            else:
                features = self.extractor.update(
                    landmarks=None,
                    face_detected=False,
                    frame_width=640,
                    frame_height=480,
                )

            # 2. Decision Logic
            reading = self.fuser.process(features)

            state = self.state_machine.update(
                reading=reading,
                no_face_duration_s=features.no_face_duration_s,
                vehicle_speed_mps=self.ego.speed_mps,
                manual_driver_response=False,
            )

            # 3. Audio & Visual Alerts
            urgency = self.state_machine.warning_urgency
            banner_col, banner_txt, sound_active = self.alert_manager.update(state, urgency)

            # 4. Vehicle & Follower Simulation
            if not self.is_paused:
                self.sim_time_s += dt
                intervention_active = state in (SystemState.INTERVENTION, SystemState.SAFE_STOP)
                hazards_active = intervention_active
                v2v_alert_active = state in (SystemState.WARNING, SystemState.INTERVENTION)

                ego_state = self.ego.get_state()
                foll_state = self.follower.vehicle.get_state()

                # Calculate ego braking command
                a_cmd = self.controller.step(ego_state, foll_state, dt, intervention_active)
                ego_state = self.ego.step(a_cmd, dt, hazards=hazards_active)

                # Follower reaction & physics
                foll_state = self.follower.step(ego_state, dt, self.sim_time_s, v2v_alert_active=v2v_alert_active)
                safety = self.safety_calc.compute(ego_state, foll_state, self.sim_time_s)
            else:
                ego_state = self.ego.get_state()
                foll_state = self.follower.vehicle.get_state()
                safety = self.safety_calc.compute(ego_state, foll_state, self.sim_time_s)

            # 5. Render Panels
            self.screen.fill(BG_COLOR)

            fps = self.clock.get_fps()
            lms = tracker_res.landmarks if tracker_res else None
            self.monitor_panel.draw(self.screen, frame_bgr, lms, features, reading, fps)

            # Time to intervention countdown
            rem_time = max(0.0, self.config.alerts.response_interval_s - self.state_machine.time_in_state)
            self.alert_panel.draw(self.screen, state, reading, urgency, rem_time, sound_active)

            self.dashboard_panel.draw(self.screen, ego_state, foll_state, safety)
            self.road_view.draw(self.screen, ego_state, foll_state, safety, self.active_strategy)

            stopping_dist = max(0.0, ego_state.position_m)
            self.perf_panel.draw(
                self.screen,
                min_gap=self.safety_calc.min_gap_m,
                collision_detected=self.safety_calc.has_collided,
                impact_speed=self.safety_calc.impact_speed_mps * 3.6,
                stopping_dist=stopping_dist,
                strategy=self.active_strategy,
            )

            pygame.display.flip()
            self.clock.tick(60)

        # Cleanup on exit
        self.cleanup()

    def cleanup(self) -> None:
        """Release camera, audio mixer, and pygame resources."""
        logger.info("Cleaning up application resources.")
        self.camera.stop()
        self.tracker.close()
        self.alert_manager.buzzer.stop()
        pygame.quit()

"""UI panels and drawing components for Pygame dashboard."""

from __future__ import annotations
import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pygame

from src.decision.fusion import DrowsinessReading, DrowsinessLevel
from src.decision.state_machine import SystemState
from src.sim.safety_metrics import SafetySnapshot
from src.sim.vehicle import VehicleState
from src.vision.features import DriverFeatures

# Color Palette
BG_COLOR = (24, 26, 31)
PANEL_BG = (33, 37, 43)
PANEL_BORDER = (55, 60, 70)
TEXT_WHITE = (235, 238, 242)
TEXT_MUTED = (160, 165, 175)
ACCENT_BLUE = (61, 139, 253)
ALERT_GREEN = (46, 204, 113)
ALERT_AMBER = (241, 196, 15)
ALERT_RED = (231, 76, 60)
ROAD_GRAY = (45, 48, 54)
ROAD_MARKING = (220, 220, 220)


class DriverMonitoringPanel:
    """Renders the webcam / video feed, facial landmarks overlay, and driver biometric gauges."""

    def __init__(self, rect: pygame.Rect, font_small: pygame.font.Font, font_med: pygame.font.Font) -> None:
        self.rect = rect
        self.font_small = font_small
        self.font_med = font_med
        self.show_landmarks = True

    def draw(
        self,
        surface: pygame.Surface,
        frame_bgr: Optional[np.ndarray],
        landmarks: Optional[np.ndarray],
        features: Optional[DriverFeatures],
        reading: Optional[DrowsinessReading],
        fps: float,
    ) -> None:
        # Panel Background
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=8)

        # Header
        header = self.font_med.render("DRIVER MONITORING", True, ACCENT_BLUE)
        surface.blit(header, (self.rect.x + 12, self.rect.y + 10))

        fps_text = self.font_small.render(f"FPS: {fps:.1f}", True, TEXT_MUTED)
        surface.blit(fps_text, (self.rect.right - 70, self.rect.y + 12))

        # Video Frame Area
        cam_w, cam_h = 240, 180
        cam_rect = pygame.Rect(self.rect.x + 12, self.rect.y + 36, cam_w, cam_h)

        if frame_bgr is not None:
            # Convert OpenCV BGR to Pygame RGB surface
            disp_frame = cv2.resize(frame_bgr, (cam_w, cam_h))
            if self.show_landmarks and landmarks is not None:
                scale_x = cam_w / frame_bgr.shape[1]
                scale_y = cam_h / frame_bgr.shape[0]
                # Draw subtle landmarks
                for pt in landmarks[::3]:  # downsample for performance
                    cv2.circle(disp_frame, (int(pt[0] * scale_x), int(pt[1] * scale_y)), 1, (0, 255, 128), -1)

            frame_rgb = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
            frame_surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
            surface.blit(frame_surf, cam_rect)
        else:
            pygame.draw.rect(surface, (15, 15, 20), cam_rect)
            msg = self.font_small.render("NO CAMERA / NO FEED", True, ALERT_RED)
            surface.blit(msg, (cam_rect.x + 40, cam_rect.y + 80))

        pygame.draw.rect(surface, PANEL_BORDER, cam_rect, width=1)

        # Metrics details on the right of camera
        info_x = cam_rect.right + 14
        y = self.rect.y + 38

        if features and features.face_detected:
            # Eye Status Pill
            eye_status = "CLOSED" if features.is_eye_closed else "OPEN"
            eye_col = ALERT_RED if features.is_eye_closed else ALERT_GREEN
            txt = self.font_small.render(f"Eyes: {eye_status}", True, eye_col)
            surface.blit(txt, (info_x, y))

            y += 22
            txt = self.font_small.render(f"EAR: {features.ear_smooth:.2f}", True, TEXT_WHITE)
            surface.blit(txt, (info_x, y))

            y += 20
            txt = self.font_small.render(f"Closed: {features.eye_closed_duration_s:.1f} s", True, ALERT_RED if features.eye_closed_duration_s > 1.5 else TEXT_WHITE)
            surface.blit(txt, (info_x, y))

            y += 20
            txt = self.font_small.render(f"PERCLOS: {features.perclos * 100:.1f}%", True, TEXT_WHITE)
            surface.blit(txt, (info_x, y))

            y += 20
            txt = self.font_small.render(f"Yawns: {features.yawn_count} ({features.yawns_per_min:.1f}/m)", True, TEXT_WHITE)
            surface.blit(txt, (info_x, y))

            y += 20
            txt = self.font_small.render(f"Head Pitch: {features.pitch_rel:+.1f}°", True, ALERT_AMBER if features.is_head_nodding else TEXT_WHITE)
            surface.blit(txt, (info_x, y))

            y += 20
            if features.is_distracted:
                txt = self.font_small.render("DISTRACTION DETECTED", True, ALERT_AMBER)
                surface.blit(txt, (info_x, y))
        else:
            txt = self.font_med.render("NO FACE DETECTED", True, ALERT_AMBER)
            surface.blit(txt, (info_x, y + 40))
            sub = self.font_small.render("Keep head centered", True, TEXT_MUTED)
            surface.blit(sub, (info_x, y + 65))


class AlertPanel:
    """Displays current system state, drowsiness score bar, and countdown."""

    def __init__(self, rect: pygame.Rect, font_small: pygame.font.Font, font_med: pygame.font.Font, font_large: pygame.font.Font) -> None:
        self.rect = rect
        self.font_small = font_small
        self.font_med = font_med
        self.font_large = font_large

    def draw(
        self,
        surface: pygame.Surface,
        state: SystemState,
        reading: Optional[DrowsinessReading],
        urgency: float,
        time_to_intervention: float,
        buzzer_active: bool,
    ) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=8)

        header = self.font_med.render("SAFETY ESCALATION", True, ACCENT_BLUE)
        surface.blit(header, (self.rect.x + 12, self.rect.y + 10))

        # Large State Banner
        state_colors = {
            SystemState.ALERT: ALERT_GREEN,
            SystemState.CAUTION: ALERT_AMBER,
            SystemState.WARNING: (255, 120, 0),
            SystemState.INTERVENTION: ALERT_RED,
            SystemState.SAFE_STOP: (180, 0, 0),
            SystemState.RECOVERY: (0, 180, 216),
        }
        col = state_colors.get(state, TEXT_WHITE)
        banner_rect = pygame.Rect(self.rect.x + 12, self.rect.y + 36, self.rect.width - 24, 46)
        pygame.draw.rect(surface, col, banner_rect, border_radius=6)

        state_text = self.font_large.render(state.value, True, (255, 255, 255))
        surface.blit(state_text, (banner_rect.centerx - state_text.get_width() // 2, banner_rect.centery - state_text.get_height() // 2))

        # Drowsiness Score Bar (0 to 100)
        score = reading.score if reading else 0.0
        bar_y = self.rect.y + 96
        lbl = self.font_small.render(f"Drowsiness Score: {score:.1f} / 100", True, TEXT_WHITE)
        surface.blit(lbl, (self.rect.x + 12, bar_y))

        bar_rect = pygame.Rect(self.rect.x + 12, bar_y + 20, self.rect.width - 24, 18)
        pygame.draw.rect(surface, (50, 50, 60), bar_rect, border_radius=4)

        fill_w = int((min(100.0, max(0.0, score)) / 100.0) * bar_rect.width)
        fill_col = ALERT_RED if score >= 60 else (ALERT_AMBER if score >= 35 else ALERT_GREEN)
        if fill_w > 0:
            pygame.draw.rect(surface, fill_col, (bar_rect.x, bar_rect.y, fill_w, bar_rect.height), border_radius=4)

        # Countdown & Buzzer Indicator
        y = bar_y + 48
        if state == SystemState.WARNING:
            txt = self.font_med.render(f"Braking Intervention in: {time_to_intervention:.1f}s", True, ALERT_RED)
            surface.blit(txt, (self.rect.x + 12, y))
        else:
            txt = self.font_small.render("Intervention status: STANDBY", True, TEXT_MUTED)
            surface.blit(txt, (self.rect.x + 12, y))

        y += 24
        buzzer_lbl = "BUZZER: ACTIVE" if buzzer_active else "BUZZER: SILENT"
        buzzer_col = ALERT_RED if buzzer_active else TEXT_MUTED
        b_surf = self.font_small.render(buzzer_lbl, True, buzzer_col)
        surface.blit(b_surf, (self.rect.x + 12, y))

        # Key guidance
        y += 24
        hints = self.font_small.render("Keys: [D] Drowsiness | [SPACE] Response | [L] Landmarks", True, TEXT_MUTED)
        surface.blit(hints, (self.rect.x + 12, y))


class VehicleDashboardPanel:
    """Speedometer, brake lights, following distance, and TTC gauges."""

    def __init__(self, rect: pygame.Rect, font_small: pygame.font.Font, font_med: pygame.font.Font, font_large: pygame.font.Font) -> None:
        self.rect = rect
        self.font_small = font_small
        self.font_med = font_med
        self.font_large = font_large

    def draw(
        self,
        surface: pygame.Surface,
        ego_state: VehicleState,
        follower_state: Optional[VehicleState],
        safety: Optional[SafetySnapshot],
    ) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=8)

        header = self.font_med.render("VEHICLE TELEMETRY", True, ACCENT_BLUE)
        surface.blit(header, (self.rect.x + 12, self.rect.y + 10))

        # Speedometer Display
        spd = ego_state.speed_kmh
        spd_text = self.font_large.render(f"{spd:.0f}", True, TEXT_WHITE)
        unit_text = self.font_small.render("km/h", True, TEXT_MUTED)
        surface.blit(spd_text, (self.rect.x + 15, self.rect.y + 40))
        surface.blit(unit_text, (self.rect.x + 85, self.rect.y + 55))

        # Brake Intensity Bar
        b_lbl = self.font_small.render(f"Braking: {ego_state.brake_intensity * 100:.0f}%", True, TEXT_WHITE)
        surface.blit(b_lbl, (self.rect.x + 140, self.rect.y + 40))
        b_rect = pygame.Rect(self.rect.x + 140, self.rect.y + 58, 120, 14)
        pygame.draw.rect(surface, (50, 50, 60), b_rect, border_radius=3)
        if ego_state.brake_intensity > 0:
            fill_w = int(ego_state.brake_intensity * b_rect.width)
            pygame.draw.rect(surface, ALERT_RED, (b_rect.x, b_rect.y, fill_w, b_rect.height), border_radius=3)

        # Brake Lights / Hazard Pill Indicators
        pill_y = self.rect.y + 90
        b_color = ALERT_RED if ego_state.brake_lights_on else (60, 60, 70)
        h_color = ALERT_AMBER if ego_state.hazards_on else (60, 60, 70)

        pygame.draw.rect(surface, b_color, (self.rect.x + 12, pill_y, 90, 24), border_radius=4)
        surface.blit(self.font_small.render("BRAKE", True, TEXT_WHITE), (self.rect.x + 36, pill_y + 4))

        pygame.draw.rect(surface, h_color, (self.rect.x + 110, pill_y, 90, 24), border_radius=4)
        surface.blit(self.font_small.render("HAZARD", True, TEXT_WHITE), (self.rect.x + 130, pill_y + 4))

        # Follower Gap & TTC
        y = pill_y + 36
        if safety:
            gap_col = ALERT_RED if safety.gap_m < 10.0 else (ALERT_AMBER if safety.gap_m < 25.0 else ALERT_GREEN)
            txt = self.font_small.render(f"Following Gap: {safety.gap_m:.1f} m", True, gap_col)
            surface.blit(txt, (self.rect.x + 12, y))

            y += 20
            ttc_str = f"{safety.ttc_s:.1f} s" if safety.ttc_s < 50.0 else "INF"
            txt = self.font_small.render(f"Time to Collision (TTC): {ttc_str}", True, TEXT_WHITE)
            surface.blit(txt, (self.rect.x + 12, y))

            y += 20
            txt = self.font_small.render(f"RSS Safe Dist: {safety.rss_safe_dist_m:.1f} m", True, TEXT_MUTED)
            surface.blit(txt, (self.rect.x + 12, y))

            y += 20
            risk_col = ALERT_RED if safety.risk_level in ("COLLISION", "CRITICAL") else (ALERT_AMBER if safety.risk_level == "CAUTION" else ALERT_GREEN)
            txt = self.font_med.render(f"Collision Risk: {safety.risk_level}", True, risk_col)
            surface.blit(txt, (self.rect.x + 12, y))


class RoadSimulationView:
    """Renders scrolling animated 2D road lane with ego vehicle and following vehicle."""

    def __init__(self, rect: pygame.Rect, font_small: pygame.font.Font, font_med: pygame.font.Font) -> None:
        self.rect = rect
        self.font_small = font_small
        self.font_med = font_med
        self._scroll_offset: float = 0.0

    def draw(
        self,
        surface: pygame.Surface,
        ego_state: VehicleState,
        follower_state: Optional[VehicleState],
        safety: Optional[SafetySnapshot],
        strategy: str,
    ) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=8)

        # Header
        header = self.font_med.render(f"2D ROAD SIMULATION - CONTROLLER: {strategy.upper()}", True, ACCENT_BLUE)
        surface.blit(header, (self.rect.x + 12, self.rect.y + 10))

        # Road Surface
        road_w = self.rect.width - 24
        road_h = 160
        road_rect = pygame.Rect(self.rect.x + 12, self.rect.y + 36, road_w, road_h)
        pygame.draw.rect(surface, ROAD_GRAY, road_rect)
        pygame.draw.rect(surface, (80, 80, 85), road_rect, width=2)

        # Scrolling Road Dash Markings
        self._scroll_offset = (self._scroll_offset + (ego_state.speed_mps * 0.4)) % 60.0
        lane_y = road_rect.centery
        dash_len = 30
        dash_gap = 30
        for x in range(int(road_rect.x - self._scroll_offset), road_rect.right, dash_len + dash_gap):
            if x + dash_len > road_rect.x and x < road_rect.right:
                rx = max(road_rect.x, x)
                rw = min(x + dash_len, road_rect.right) - rx
                pygame.draw.line(surface, ROAD_MARKING, (rx, lane_y), (rx + rw, lane_y), 3)

        # Vehicle drawing coordinates (Pixels per meter)
        ppm = 4.5  # 1 meter = 4.5 pixels
        veh_len_px = int(4.5 * ppm)
        veh_w_px = 36

        # Ego Vehicle (Positioned ahead on the road)
        ego_x = road_rect.x + int(road_w * 0.65)
        ego_y = lane_y - veh_w_px // 2

        # Ego Body (Blue Car)
        pygame.draw.rect(surface, (41, 128, 185), (ego_x, ego_y, veh_len_px, veh_w_px), border_radius=5)
        # Ego Windshield
        pygame.draw.rect(surface, (200, 230, 255), (ego_x + int(veh_len_px * 0.6), ego_y + 4, 6, veh_w_px - 8))
        # Ego Brake Lights
        if ego_state.brake_lights_on:
            pygame.draw.rect(surface, ALERT_RED, (ego_x - 3, ego_y + 2, 4, 8))
            pygame.draw.rect(surface, ALERT_RED, (ego_x - 3, ego_y + veh_w_px - 10, 4, 8))
        # Ego Hazards
        if ego_state.hazards_on and (int(time.time() * 4) % 2 == 0):
            pygame.draw.circle(surface, ALERT_AMBER, (ego_x, ego_y), 5)
            pygame.draw.circle(surface, ALERT_AMBER, (ego_x, ego_y + veh_w_px), 5)

        ego_lbl = self.font_small.render("EGO (LEAD)", True, TEXT_WHITE)
        surface.blit(ego_lbl, (ego_x, ego_y - 18))

        # Following Vehicle
        if follower_state and safety:
            foll_x = ego_x - veh_len_px - int(safety.gap_m * ppm)
            foll_y = ego_y

            if foll_x > road_rect.x - veh_len_px:
                # Follower Body (Gray/Red Car)
                f_col = (140, 140, 150)
                pygame.draw.rect(surface, f_col, (foll_x, foll_y, veh_len_px, veh_w_px), border_radius=5)
                # Follower Windshield
                pygame.draw.rect(surface, (200, 230, 255), (foll_x + int(veh_len_px * 0.6), foll_y + 4, 6, veh_w_px - 8))
                if follower_state.brake_lights_on:
                    pygame.draw.rect(surface, ALERT_RED, (foll_x - 3, foll_y + 2, 4, 8))
                    pygame.draw.rect(surface, ALERT_RED, (foll_x - 3, foll_y + veh_w_px - 10, 4, 8))

                f_lbl = self.font_small.render("FOLLOWER", True, TEXT_WHITE)
                surface.blit(f_lbl, (foll_x, foll_y - 18))

                # Distance gap marker between vehicles
                if safety.gap_m > 0:
                    marker_y = road_rect.bottom - 18
                    pygame.draw.line(surface, TEXT_WHITE, (foll_x + veh_len_px, marker_y), (ego_x, marker_y), 2)
                    gap_text = self.font_small.render(f"Gap: {safety.gap_m:.1f}m", True, TEXT_WHITE)
                    surface.blit(gap_text, ((foll_x + veh_len_px + ego_x) // 2 - 25, marker_y - 16))
                elif safety.is_collision:
                    # Flash collision impact text
                    impact_text = self.font_med.render(f"COLLISION! IMPACT SPEED: {safety.impact_speed_mps * 3.6:.1f} km/h", True, ALERT_RED)
                    surface.blit(impact_text, (road_rect.centerx - impact_text.get_width() // 2, road_rect.y + 10))


class PerformanceAnalysisPanel:
    """Shows live telemetry comparison and benchmarking stats."""

    def __init__(self, rect: pygame.Rect, font_small: pygame.font.Font, font_med: pygame.font.Font) -> None:
        self.rect = rect
        self.font_small = font_small
        self.font_med = font_med

    def draw(
        self,
        surface: pygame.Surface,
        min_gap: float,
        collision_detected: bool,
        impact_speed: float,
        stopping_dist: float,
        strategy: str,
    ) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=8)

        header = self.font_med.render("PERFORMANCE & STRATEGY EVALUATION", True, ACCENT_BLUE)
        surface.blit(header, (self.rect.x + 12, self.rect.y + 10))

        y = self.rect.y + 36
        txt = self.font_small.render(f"Active Strategy: {strategy.upper()}", True, TEXT_WHITE)
        surface.blit(txt, (self.rect.x + 12, y))

        y += 22
        gap_col = ALERT_RED if min_gap < 5.0 else ALERT_GREEN
        txt = self.font_small.render(f"Minimum Gap So Far: {min_gap:.2f} m", True, gap_col)
        surface.blit(txt, (self.rect.x + 12, y))

        y += 22
        col_text = f"YES (Speed: {impact_speed:.1f} km/h)" if collision_detected else "NO (0 Collisions)"
        col_color = ALERT_RED if collision_detected else ALERT_GREEN
        txt = self.font_small.render(f"Collision Occurred: {col_text}", True, col_color)
        surface.blit(txt, (self.rect.x + 12, y))

        y += 22
        txt = self.font_small.render(f"Stopping Distance: {stopping_dist:.1f} m", True, TEXT_WHITE)
        surface.blit(txt, (self.rect.x + 12, y))

        # Comparison summary guide
        y += 28
        txt = self.font_small.render("Controller Characteristics:", True, ACCENT_BLUE)
        surface.blit(txt, (self.rect.x + 12, y))
        y += 18
        txt = self.font_small.render("• Abrupt: Shortest stop, highest rear-end collision risk", True, TEXT_MUTED)
        surface.blit(txt, (self.rect.x + 12, y))
        y += 18
        txt = self.font_small.render("• Gradual: Smoother stop, but increases stopping distance", True, TEXT_MUTED)
        surface.blit(txt, (self.rect.x + 12, y))
        y += 18
        txt = self.font_small.render("• Adaptive: Balances follower safety & braking authority", True, TEXT_MUTED)
        surface.blit(txt, (self.rect.x + 12, y))

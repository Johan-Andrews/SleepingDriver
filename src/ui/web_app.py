"""Web-based Dashboard streaming live video feed, dual-engine Audio alarm, and advanced traffic simulation."""

from __future__ import annotations
import base64
import io
import json
import logging
import threading
import time
import wave
import cv2
from flask import Flask, Response, jsonify, render_template_string, request
import numpy as np

from src.config import AppConfig
from src.decision.alerts import AlertManager
from src.decision.fusion import DrowsinessFuser
from src.decision.state_machine import SafetyStateMachine, SystemState
from src.sim.controllers import create_controller
from src.sim.follower import FollowerVehicle
from src.sim.safety_metrics import SafetyMetricsCalculator
from src.sim.vehicle import Vehicle
from src.vision.camera import ThreadedCamera
from src.vision.calibration import Calibrator
from src.vision.face_tracker import FaceTracker
from src.vision.features import FeatureExtractor, calculate_ear, calculate_mar, calculate_head_pose
from src.vision.video_source import VideoSource

logger = logging.getLogger(__name__)

# Pre-generate 180ms 750Hz PCM WAV beep sound as base64 data URI
def _generate_beep_base64() -> str:
    sr = 44100
    t = np.linspace(0, 0.18, int(sr * 0.18), False)
    # Sine wave with smooth envelope
    env = np.sin(np.pi * np.linspace(0, 1, len(t)))
    wave_pcm = (np.sin(2 * np.pi * 750 * t) * env * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(wave_pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

BEEP_DATA_URI = _generate_beep_base64()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Driver Safety & Adaptive Braking Simulation</title>
    <style>
        :root {
            --bg-dark: #090c12;
            --panel-bg: #131722;
            --card-border: #202738;
            --accent-blue: #2563eb;
            --alert-green: #10b981;
            --alert-amber: #f59e0b;
            --alert-orange: #f97316;
            --alert-red: #ef4444;
            --text-light: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-dark); color: var(--text-light); min-height: 100vh; padding: 10px 16px; overflow-x: hidden; }

        /* TOP NAVBAR */
        .top-navbar {
            display: flex; justify-content: space-between; align-items: center;
            background: var(--panel-bg); border: 1px solid var(--card-border);
            padding: 8px 16px; border-radius: 10px; margin-bottom: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .app-title { font-size: 17px; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

        /* PROMINENT SLEEP TIMER BANNER */
        .sleep-timer-banner {
            display: flex; align-items: center; justify-content: center; gap: 14px;
            border-radius: 10px; padding: 10px 20px; margin-bottom: 10px;
            text-align: center; transition: all 0.25s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.25);
            background: rgba(16, 185, 129, 0.18);
            border: 2px solid #10b981;
        }
        .sleep-timer-label { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #d1fae5; }
        .sleep-timer-val {
            font-size: 26px; font-weight: 900; font-family: monospace;
            padding: 3px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2);
            background: #064e3b; color: #a7f3d0;
            transition: all 0.25s ease;
        }
        .sleep-timer-rule { font-size: 11px; opacity: 0.9; }

        /* MAIN GRID */
        .main-container { display: grid; grid-template-columns: 350px 1fr; gap: 12px; }

        /* PANELS */
        .panel {
            background: var(--panel-bg); border: 1px solid var(--card-border);
            border-radius: 12px; padding: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.25);
            display: flex; flex-direction: column; gap: 10px;
        }
        .panel-header { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #93c5fd; display: flex; align-items: center; justify-content: space-between; }

        /* CAMERA BOX */
        .camera-container {
            position: relative; width: 100%; border-radius: 8px; overflow: hidden;
            background: #000; border: 2px solid #374151; aspect-ratio: 4 / 3;
        }
        .camera-container img { width: 100%; height: 100%; object-fit: cover; display: block; }

        .sim-override-badge {
            position: absolute; top: 8px; right: 8px; background: #ea580c; color: white;
            padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: bold;
            display: none; box-shadow: 0 0 10px #ea580c;
        }

        /* TELEMETRY CARD */
        .telemetry-card {
            background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
            border-radius: 8px; padding: 8px 10px;
        }
        .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 11px; }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: var(--text-muted); }
        .metric-val { font-weight: 700; color: #fff; }

        /* STATE BANNER */
        .alert-banner {
            padding: 8px; border-radius: 8px; text-align: center;
            font-size: 14px; font-weight: 800; letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }
        .state-alert { background: linear-gradient(135deg, #059669, #10b981); color: #fff; }
        .state-caution { background: linear-gradient(135deg, #d97706, #f59e0b); color: #fff; }
        .state-warning { background: linear-gradient(135deg, #ea580c, #f97316); color: #fff; animation: warningPulse 0.5s infinite alternate; }
        .state-intervention { background: linear-gradient(135deg, #dc2626, #ef4444); color: #fff; animation: brakeFlash 0.3s infinite alternate; }
        @keyframes warningPulse { from { opacity: 0.85; } to { opacity: 1; transform: scale(1.01); } }
        @keyframes brakeFlash { from { filter: brightness(1); } to { filter: brightness(1.35); transform: scale(1.02); } }

        /* CANVAS */
        .canvas-wrapper {
            position: relative; width: 100%; border-radius: 10px; overflow: hidden;
            background: #1e293b; border: 2px solid #334155; box-shadow: inset 0 0 30px rgba(0,0,0,0.6);
        }
        canvas { display: block; width: 100%; height: 320px; }

        /* CONTROLS BUTTONS */
        .controls-toolbar { display: flex; gap: 8px; flex-wrap: wrap; }
        .btn {
            flex: 1; min-width: 110px; padding: 9px 12px; border: none; border-radius: 6px;
            font-size: 11px; font-weight: 700; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px;
            transition: all 0.2s ease;
        }
        .btn:active { transform: translateY(2px); }
        .btn-sleep { background: linear-gradient(135deg, #f97316, #ef4444); color: white; }
        .btn-sleep:hover { filter: brightness(1.15); }
        .btn-awake { background: linear-gradient(135deg, #059669, #10b981); color: white; }
        .btn-awake:hover { filter: brightness(1.15); }
        .btn-drive { background: linear-gradient(135deg, #2563eb, #3b82f6); color: white; font-weight: 800; }
        .btn-drive:hover { filter: brightness(1.15); }
        .btn-audio { background: #1f2937; border: 1px solid #374151; color: #e5e7eb; }
        .btn-strat { background: #1e3a8a; color: #93c5fd; padding: 7px; font-size: 11px; border: 1px solid transparent; }
        .btn-strat.active { border-color: #60a5fa; background: #2563eb; color: #fff; box-shadow: 0 0 10px rgba(37,99,235,0.5); }

        .direction-badge {
            background: rgba(0,0,0,0.6); border: 1px solid #3b82f6; border-radius: 6px;
            padding: 4px 10px; font-size: 11px; font-weight: bold; color: #93c5fd;
        }

        .audio-unmute-modal {
            background: #f59e0b; color: #000; padding: 5px 10px; border-radius: 6px; font-weight: bold; font-size: 11px;
            cursor: pointer; display: none;
        }
    </style>
</head>
<body onclick="userClickedBody()">

    <!-- AUDIO ELEMENT BACKUP FOR GUARANTEED BROWSER BUZZER -->
    <audio id="beepAudio" preload="auto">
        <source src="{{ beep_uri }}" type="audio/wav">
    </audio>

    <!-- TOP NAVBAR -->
    <header class="top-navbar">
        <div class="app-title">🚗 Autonomous Driver Monitoring & Highway Traffic Simulation</div>
        <div style="display:flex; gap:10px; align-items:center;">
            <div id="audioNotice" class="audio-unmute-modal" onclick="enableAudio()">🔔 Click here to Enable Audio Buzzer</div>
            <button class="btn btn-audio" id="audioToggleBtn" onclick="toggleAudio()" style="padding:4px 8px; font-size:11px;">🔊 Buzzer: ACTIVE</button>
            <div class="direction-badge">🧭 DIRECTION: EAST ➔</div>
        </div>
    </header>

    <!-- PROMINENT SLEEP TIMER WITH DYNAMIC THRESHOLD COLORS -->
    <div class="sleep-timer-banner" id="sleepBanner">
        <span class="sleep-timer-label" id="sleepLabel">🟢 Driver Awake & Attentive:</span>
        <span class="sleep-timer-val" id="topSleepTimer">0.0 s</span>
        <span class="sleep-timer-rule" id="sleepRule">[Buzzer at 2.0s | Autonomous Braking at 5.0s | Awake = Accelerate & Resume]</span>
    </div>

    <div class="main-container">

        <!-- LEFT COLUMN: CAMERA & BIOMETRICS -->
        <div class="panel">
            <div class="panel-header">
                <span>Driver Facial Telemetry</span>
                <span id="controlStatusText" style="color:#10b981; font-weight:bold;">DRIVER HAS CONTROL</span>
            </div>

            <!-- LIVE CAMERA -->
            <div class="camera-container">
                <img src="/video_feed" alt="Webcam Driver Feed">
                <div class="sim-override-badge" id="simBadge">SIMULATION OVERRIDE (D KEY)</div>
            </div>

            <!-- STATE BANNER -->
            <div id="stateBanner" class="alert-banner state-alert">DRIVER HAS CONTROL (NORMAL DRIVING)</div>

            <!-- BIOMETRIC DATA -->
            <div class="telemetry-card">
                <div class="metric-row">
                    <span class="metric-label">Input Source</span>
                    <span class="metric-val" id="inputSourceLabel" style="color:#38bdf8;">LIVE WEBCAM AI</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Fused Drowsiness Score</span>
                    <span class="metric-val" id="scoreText">0.0 / 100</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Eye State</span>
                    <span class="metric-val" id="eyeStatus">OPEN (EAR: 0.33)</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">PERCLOS</span>
                    <span class="metric-val" id="perclosVal">0.0%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Head Pitch Pose</span>
                    <span class="metric-val" id="pitchVal">0.0° (Upright)</span>
                </div>
            </div>

            <!-- SYNCED CONTROLS -->
            <div class="controls-toolbar">
                <button class="btn btn-sleep" id="btnSleep" onclick="triggerDrowsy()">😴 Simulate Falling Asleep [D]</button>
                <button class="btn btn-awake" id="btnAwake" onclick="triggerAwake()">✋ Wake Up / Camera Control [SPACE]</button>
            </div>
            <div style="font-size:10px; color:var(--text-muted); display:flex; justify-content:space-between; margin-top:2px;">
                <span>Press <b>[D]</b>: Asleep 2s &amp; count up</span>
                <span>Press <b>[SPACE]</b>: Back to Camera</span>
            </div>
        </div>

        <!-- RIGHT COLUMN: 2-LANE HIGHWAY WITH INDEPENDENT TRAFFIC & OVERTAKING -->
        <div class="panel">
            <div class="panel-header">
                <span style="display:flex; align-items:center; gap:8px;">
                    <span>Top-View 2-Lane Highway</span>
                    <span style="font-size:11px; color:#38bdf8;">(Traffic, Bikes, Trucks &amp; Lane Changing)</span>
                </span>
                <span id="strategyBadge" style="background:#1e3a8a; color:#93c5fd; padding:3px 8px; border-radius:4px; font-size:11px;">ADAPTIVE BRAKING</span>
            </div>

            <!-- ROAD CANVAS -->
            <div class="canvas-wrapper">
                <canvas id="roadCanvas" width="850" height="320"></canvas>
            </div>

            <!-- RESUME / ACCELERATE BUTTON -->
            <div class="controls-toolbar">
                <button class="btn btn-drive" id="btnDrive" onclick="startDriving()">🚀 Start / Accelerate Driving [R]</button>
            </div>

            <!-- TELEMETRY WITH FOLLOWER STATUS -->
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:8px;">
                <div class="telemetry-card">
                    <div style="font-size:10px; color:var(--text-muted);">YOUR SPEED</div>
                    <div style="font-size:17px; font-weight:800; color:#60a5fa;" id="egoSpeedText">100 km/h</div>
                </div>
                <div class="telemetry-card">
                    <div style="font-size:10px; color:var(--text-muted);">FOLLOWER SPEED</div>
                    <div style="font-size:17px; font-weight:800; color:#f87171;" id="follSpeedText">100 km/h</div>
                </div>
                <div class="telemetry-card">
                    <div style="font-size:10px; color:var(--text-muted);">FOLLOWER GAP / STATUS</div>
                    <div style="font-size:17px; font-weight:800; color:#fbbf24;" id="gapDistanceText">30.0 m</div>
                </div>
                <div class="telemetry-card">
                    <div style="font-size:10px; color:var(--text-muted);">COLLISION RISK</div>
                    <div style="font-size:17px; font-weight:800; color:#34d399;" id="riskLevelText">SAFE</div>
                </div>
            </div>

            <!-- BRAKING STRATEGY SELECTORS -->
            <div class="controls-toolbar">
                <button class="btn btn-strat" id="stratAbrupt" onclick="setStrategy('abrupt')">1. Abrupt (Sudden Stop)</button>
                <button class="btn btn-strat" id="stratGradual" onclick="setStrategy('gradual')">2. Gradual (Jerk-Limited)</button>
                <button class="btn btn-strat active" id="stratAdaptive" onclick="setStrategy('adaptive')">3. Adaptive (Lookahead Gap-Aware)</button>
            </div>
        </div>

    </div>

    <script>
        // 1. UNIVERSAL COMPATIBILITY ROUNDED RECT HELPER (NEVER THROWS)
        function drawRoundedRect(c, x, y, w, h, r) {
            if (w < 2 * r) r = w / 2;
            if (h < 2 * r) r = h / 2;
            c.beginPath();
            c.moveTo(x + r, y);
            c.arcTo(x + w, y, x + w, y + h, r);
            c.arcTo(x + w, y + h, x, y + h, r);
            c.arcTo(x, y + h, x, y, r);
            c.arcTo(x, y, x + w, y, r);
            c.closePath();
        }

        // 2. DUAL AUDIO ENGINE (WEB AUDIO + HTML5 AUDIO ELEMENT)
        let audioCtx = null;
        let isAudioMuted = false;
        let buzzerInterval = null;

        function ensureAudioUnlocked() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            document.getElementById('audioNotice').style.display = 'none';
        }

        function userClickedBody() {
            ensureAudioUnlocked();
        }

        function playAlertTone(freq, durationMs) {
            if (isAudioMuted) return;
            ensureAudioUnlocked();

            // 1. Try HTML5 Audio element
            try {
                const aud = document.getElementById('beepAudio');
                if (aud) {
                    aud.currentTime = 0;
                    aud.play().catch(e => {});
                }
            } catch(e) {}

            // 2. Try Web Audio API oscillator
            if (audioCtx && audioCtx.state === 'running') {
                try {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = "sawtooth";
                    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);

                    gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + (durationMs / 1000));

                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.start();
                    osc.stop(audioCtx.currentTime + (durationMs / 1000));
                } catch (e) {}
            }
        }

        function toggleAudio() {
            ensureAudioUnlocked();
            isAudioMuted = !isAudioMuted;
            document.getElementById('audioToggleBtn').innerText = isAudioMuted ? "🔇 Buzzer: OFF" : "🔊 Buzzer: ACTIVE";
        }

        function enableAudio() {
            ensureAudioUnlocked();
            playAlertTone(750, 150);
        }

        // 3. KEYBOARD SHORTCUTS
        window.addEventListener('keydown', (e) => {
            ensureAudioUnlocked();
            if (e.code === 'KeyD') {
                triggerDrowsy();
            } else if (e.code === 'Space') {
                e.preventDefault();
                triggerAwake();
            } else if (e.code === 'KeyR') {
                startDriving();
            } else if (e.code === 'Digit1') {
                setStrategy('abrupt');
            } else if (e.code === 'Digit2') {
                setStrategy('gradual');
            } else if (e.code === 'Digit3') {
                setStrategy('adaptive');
            }
        });

        function triggerDrowsy() {
            ensureAudioUnlocked();
            fetch('/api/simulate_drowsy', { method: 'POST' });
        }
        function triggerAwake() {
            ensureAudioUnlocked();
            fetch('/api/wakeup', { method: 'POST' });
        }
        function startDriving() {
            ensureAudioUnlocked();
            fetch('/api/start_driving', { method: 'POST' });
        }
        function setStrategy(strat) {
            fetch('/api/strategy', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ strategy: strat })
            }).then(() => {
                document.querySelectorAll('.btn-strat').forEach(b => b.classList.remove('active'));
                if (strat === 'abrupt') document.getElementById('stratAbrupt').classList.add('active');
                if (strat === 'gradual') document.getElementById('stratGradual').classList.add('active');
                if (strat === 'adaptive') document.getElementById('stratAdaptive').classList.add('active');
            });
        }

        // 4. CANVAS 2D SIMULATION WITH DIVERSE TRAFFIC, ZERO-COLLISION INTELLIGENT OVERTAKING
        const canvas = document.getElementById('roadCanvas');
        const ctx = canvas.getContext('2d');

        let worldX = 0;
        let telemetry = {
            state: "ALERT",
            state_desc: "DRIVER HAS CONTROL",
            drowsiness_score: 0,
            is_closed: false,
            ear: 0.33,
            closed_s: 0,
            ego_speed_kmh: 100,
            ego_brake_intensity: 0,
            foll_speed_kmh: 100,
            foll_brake: false,
            gap_m: 30,
            risk_level: "SAFE",
            strategy: "adaptive",
            sim_active: false
        };

        // Vehicles on the 2-lane highway
        let egoVehicle = {
            id: 'ego',
            isEgo: true,
            lane: 1,
            targetLane: 1,
            laneY: 0,
            x: 380,
            length: 74,
            width: 34,
            type: 'car',
            color: '#2563eb',
            label: 'YOU (DRIVER)',
            turnSignal: 'none',
            isBraking: false,
            speed: 100
        };

        let followerVehicle = {
            id: 'follower',
            isFollower: true,
            lane: 1,
            targetLane: 1,
            laneY: 0,
            x: 180,
            length: 74,
            width: 34,
            type: 'car',
            color: '#ef4444',
            label: 'FOLLOWER',
            turnSignal: 'none',
            isBraking: false,
            speed: 100,
            desiredSpeed: 100,
            hasOvertaken: false
        };

        let trafficVehicles = [
            { id: 1, x: 620, lane: 1, targetLane: 1, laneY: 0, speed: 80, desiredSpeed: 80, length: 125, width: 40, type: 'truck', color: '#8b5cf6', label: 'SEMI-TRUCK', turnSignal: 'none', isBraking: false },
            { id: 2, x: 120, lane: 0, targetLane: 0, laneY: 0, speed: 112, desiredSpeed: 112, length: 42, width: 18, type: 'bike', color: '#38bdf8', label: 'BIKE', turnSignal: 'none', isBraking: false },
            { id: 3, x: 760, lane: 0, targetLane: 0, laneY: 0, speed: 104, desiredSpeed: 104, length: 74, width: 34, type: 'car', color: '#10b981', label: 'SEDAN', turnSignal: 'none', isBraking: false },
            { id: 4, x: -90, lane: 1, targetLane: 1, laneY: 0, speed: 95, desiredSpeed: 95, length: 82, width: 38, type: 'suv', color: '#f59e0b', label: 'SUV', turnSignal: 'none', isBraking: false },
        ];

        // Safe lane clearance check: returns true if target lane is completely free
        function isLaneSafeForChange(veh, targetLane, allVehicles, laneCenters, laneH) {
            const laneYTarget = laneCenters[targetLane];
            for (const other of allVehicles) {
                if (other === veh) continue;

                // Is other vehicle in target lane or moving into it?
                const isOccupyingTarget = (other.targetLane === targetLane) ||
                    (Math.abs(other.laneY - laneYTarget) < laneH * 0.55);
                if (!isOccupyingTarget) continue;

                // 1. Direct longitudinal overlap (alongside)
                const alongside = (veh.x + veh.length + 35 > other.x) && (other.x + other.length + 35 > veh.x);
                if (alongside) return false;

                // 2. Approaching from behind in target lane
                if (other.x < veh.x) {
                    const gapBehind = veh.x - (other.x + other.length);
                    if (gapBehind < 85 && (other.speed >= veh.speed - 5)) return false;
                }

                // 3. Ahead in target lane
                if (other.x > veh.x) {
                    const gapAhead = other.x - (veh.x + veh.length);
                    if (gapAhead < 80) return false;
                }
            }
            return true;
        }

        function drawRoad() {
            try {
                const w = canvas.width;
                const h = canvas.height;

                // 1. ROAD LAYOUT & BACKGROUND
                ctx.fillStyle = '#0f172a';
                ctx.fillRect(0, 0, w, h);

                const roadY = 40;
                const roadH = h - 80;
                const laneH = roadH / 2;
                const laneCenters = [roadY + laneH * 0.5, roadY + laneH * 1.5]; // Lane 0 (Top, y=100), Lane 1 (Bottom, y=220)

                // Road Shoulders (Green grass with gravel edge)
                ctx.fillStyle = '#14532d';
                ctx.fillRect(0, 0, w, roadY);
                ctx.fillRect(0, roadY + roadH, w, h - (roadY + roadH));

                // Asphalt
                ctx.fillStyle = '#1e293b';
                ctx.fillRect(0, roadY, w, roadH);

                // 2. ROAD MOTION: SCROLLS TOWARDS THE LEFT BASED ON DRIVER SPEED
                const speedFactor = Math.max(0, telemetry.ego_speed_kmh / 100);
                worldX += 9 * speedFactor;

                // Animated Rumble Curbs (Red & White blocks) moving LEFT
                const curbW = 28;
                const curbOffset = worldX % (curbW * 2);
                for (let cx = -curbW * 2; cx < w + curbW * 2; cx += curbW) {
                    const drawX = cx - curbOffset;
                    ctx.fillStyle = (Math.floor(cx / curbW) % 2 === 0) ? '#dc2626' : '#f8fafc';
                    ctx.fillRect(drawX, roadY - 7, curbW, 7);
                    ctx.fillRect(drawX, roadY + roadH, curbW, 7);
                }

                // White Solid Edge Lines
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(0, roadY); ctx.lineTo(w, roadY);
                ctx.moveTo(0, roadY + roadH); ctx.lineTo(w, roadY + roadH);
                ctx.stroke();

                // Center Dividing Dashed Line (Flows towards the LEFT)
                ctx.strokeStyle = '#f8fafc';
                ctx.lineWidth = 3;
                ctx.setLineDash([32, 28]);
                ctx.lineDashOffset = worldX; // Shifts dash pattern towards LEFT
                ctx.beginPath();
                ctx.moveTo(0, roadY + laneH);
                ctx.lineTo(w, roadY + laneH);
                ctx.stroke();
                ctx.setLineDash([]);

                // Highway Lane Chevrons & Direction Markings painted on pavement moving towards LEFT
                ctx.fillStyle = 'rgba(255, 255, 255, 0.18)';
                ctx.font = 'bold 13px monospace';
                const markerSpacing = 240;
                const markerOffset = worldX % markerSpacing;
                for (let mx = -markerSpacing; mx < w + markerSpacing; mx += markerSpacing) {
                    const drawX = mx + markerSpacing - markerOffset;
                    ctx.fillText('➔ ➔ ➔  LANE 0 (PASSING)', drawX, roadY + laneH * 0.52);
                    ctx.fillText('➔ ➔ ➔  LANE 1 (CRUISING)', drawX, roadY + laneH * 1.52);
                }

                // 3. MULTI-VEHICLE SIMULATION & COLLISION-AVOIDANCE OVERTAKING
                const allVehicles = [egoVehicle, followerVehicle, ...trafficVehicles];

                for (const v of allVehicles) {
                    if (v.laneY === 0) v.laneY = laneCenters[v.targetLane];
                }

                // --- A. DRIVER (EGO) VEHICLE LOGIC ---
                egoVehicle.speed = telemetry.ego_speed_kmh;
                egoVehicle.isBraking = telemetry.ego_brake_intensity > 0.05 || (telemetry.closed_s >= 5.0);

                const isDriverAlert = (telemetry.state === 'ALERT' || telemetry.closed_s < 2.0) && !egoVehicle.isBraking;

                // Driver overtaking: when a vehicle is in front of driver, change lane and overtake!
                if (isDriverAlert) {
                    let egoLead = null;
                    let egoLeadGap = 99999;
                    for (const other of allVehicles) {
                        if (other === egoVehicle) continue;
                        const sameLane = (other.targetLane === egoVehicle.targetLane) || (Math.abs(other.laneY - egoVehicle.laneY) < laneH * 0.45);
                        if (sameLane && other.x > egoVehicle.x) {
                            const gap = other.x - (egoVehicle.x + egoVehicle.length);
                            if (gap < egoLeadGap) {
                                egoLeadGap = gap;
                                egoLead = other;
                            }
                        }
                    }

                    if (egoLead && egoLeadGap < 150) {
                        // Slower vehicle ahead! Driver initiates lane change to overtake
                        const otherLane = 1 - egoVehicle.targetLane;
                        if (isLaneSafeForChange(egoVehicle, otherLane, allVehicles, laneCenters, laneH)) {
                            egoVehicle.targetLane = otherLane;
                            egoVehicle.turnSignal = (otherLane === 0) ? 'left' : 'right';
                        }
                    } else if (egoVehicle.targetLane === 0 && egoVehicle.lane === 0) {
                        // Return to cruising lane 1 if passing is complete and lane 1 is clear
                        let clearRight = true;
                        for (const other of allVehicles) {
                            if (other === egoVehicle) continue;
                            if (other.targetLane === 1 && Math.abs(other.x - egoVehicle.x) < 140) {
                                clearRight = false;
                                break;
                            }
                        }
                        if (clearRight && isLaneSafeForChange(egoVehicle, 1, allVehicles, laneCenters, laneH)) {
                            egoVehicle.targetLane = 1;
                            egoVehicle.turnSignal = 'right';
                        }
                    }
                } else {
                    egoVehicle.turnSignal = 'none';
                }

                // Driver lane interpolation
                egoVehicle.laneY += (laneCenters[egoVehicle.targetLane] - egoVehicle.laneY) * 0.08;
                if (Math.abs(laneCenters[egoVehicle.targetLane] - egoVehicle.laneY) < 2) {
                    egoVehicle.lane = egoVehicle.targetLane;
                    if (egoVehicle.lane === egoVehicle.targetLane) egoVehicle.turnSignal = 'none';
                }

                // --- B. REAR FOLLOWER VEHICLE LOGIC ---
                // "if the driver brakes, the vehicle behind should also brake."
                let followerLead = null;
                let followerLeadGap = 99999;
                for (const other of allVehicles) {
                    if (other === followerVehicle) continue;
                    const sameLane = (other.targetLane === followerVehicle.targetLane) || (Math.abs(other.laneY - followerVehicle.laneY) < laneH * 0.45);
                    if (sameLane && other.x > followerVehicle.x) {
                        const gap = other.x - (followerVehicle.x + followerVehicle.length);
                        if (gap < followerLeadGap) {
                            followerLeadGap = gap;
                            followerLead = other;
                        }
                    }
                }

                if (followerLead) {
                    // Lead vehicle is braking or close ahead
                    if (followerLead.isBraking || followerLeadGap < 140 || (egoVehicle.isBraking && followerLead === egoVehicle)) {
                        followerVehicle.isBraking = true;
                        // Follower decelerates strongly to match/undershoot lead vehicle speed
                        followerVehicle.speed = Math.max(0, Math.min(followerVehicle.speed - 2.0, followerLead.speed - 2));

                        // If lead vehicle is nearly stopped or braking, follower changes lane to pass!
                        const otherLane = 1 - followerVehicle.targetLane;
                        if ((followerLead.speed < 45 || followerVehicle.isBraking) && isLaneSafeForChange(followerVehicle, otherLane, allVehicles, laneCenters, laneH)) {
                            followerVehicle.targetLane = otherLane;
                            followerVehicle.turnSignal = (otherLane === 0) ? 'left' : 'right';
                            followerVehicle.hasOvertaken = true;
                        }
                    } else {
                        followerVehicle.isBraking = false;
                        if (followerVehicle.speed < followerVehicle.desiredSpeed) followerVehicle.speed += 0.5;
                    }

                    // Strict collision buffer: NEVER touch lead vehicle
                    if (followerLeadGap < 35) {
                        followerVehicle.x = followerLead.x - followerVehicle.length - 35;
                        followerVehicle.speed = Math.min(followerVehicle.speed, followerLead.speed);
                        followerVehicle.isBraking = true;
                    }
                } else {
                    followerVehicle.isBraking = false;
                    if (followerVehicle.speed < followerVehicle.desiredSpeed) followerVehicle.speed += 0.5;
                }

                // Follower position relative to driver speed
                const follRelDiff = (followerVehicle.speed - telemetry.ego_speed_kmh) * 0.07;
                followerVehicle.x += follRelDiff;

                followerVehicle.laneY += (laneCenters[followerVehicle.targetLane] - followerVehicle.laneY) * 0.08;
                if (Math.abs(laneCenters[followerVehicle.targetLane] - followerVehicle.laneY) < 2) {
                    followerVehicle.lane = followerVehicle.targetLane;
                    if (followerVehicle.lane === followerVehicle.targetLane) followerVehicle.turnSignal = 'none';
                }

                // If follower has completed overtake and gone far off right, loop back behind driver
                if (followerVehicle.x > w + 220) {
                    followerVehicle.x = egoVehicle.x - 190;
                    followerVehicle.targetLane = egoVehicle.targetLane;
                    followerVehicle.hasOvertaken = false;
                }

                // --- C. SURROUNDING TRAFFIC (SEDAN, BIKE, TRUCK, SUV) ---
                for (const car of trafficVehicles) {
                    let carLead = null;
                    let carLeadGap = 99999;
                    for (const other of allVehicles) {
                        if (other === car) continue;
                        const sameLane = (other.targetLane === car.targetLane) || (Math.abs(other.laneY - car.laneY) < laneH * 0.45);
                        if (sameLane && other.x > car.x) {
                            const gap = other.x - (car.x + car.length);
                            if (gap < carLeadGap) {
                                carLeadGap = gap;
                                carLead = other;
                            }
                        }
                    }

                    if (carLead && (carLeadGap < 140)) {
                        // Slower or braking car ahead: overtake if lane is safe, otherwise brake!
                        const wantOvertake = (car.speed > carLead.speed - 3) || carLead.isBraking || (carLeadGap < 95);
                        const otherLane = 1 - car.targetLane;

                        if (wantOvertake && isLaneSafeForChange(car, otherLane, allVehicles, laneCenters, laneH)) {
                            car.targetLane = otherLane;
                            car.turnSignal = (otherLane === 0) ? 'left' : 'right';
                            car.isBraking = false;
                            if (car.speed < car.desiredSpeed + 8) car.speed += 0.4;
                        } else {
                            // Target lane blocked: MUST BRAKE behind lead car!
                            car.isBraking = true;
                            car.speed = Math.max(0, Math.min(car.speed - 1.5, carLead.speed - 1));
                            // Absolute collision prevention barrier
                            if (carLeadGap < 35) {
                                car.x = carLead.x - car.length - 35;
                                car.speed = Math.min(car.speed, carLead.speed);
                            }
                        }
                    } else {
                        car.isBraking = false;
                        if (car.speed < car.desiredSpeed) car.speed += 0.4;
                        if (car.speed > car.desiredSpeed) car.speed -= 0.2;
                    }

                    // Move relative to driver vehicle
                    const relDiff = (car.speed - telemetry.ego_speed_kmh) * 0.07;
                    car.x += relDiff;

                    car.laneY += (laneCenters[car.targetLane] - car.laneY) * 0.08;
                    if (Math.abs(laneCenters[car.targetLane] - car.laneY) < 2) {
                        car.lane = car.targetLane;
                        if (car.lane === car.targetLane) car.turnSignal = 'none';
                    }

                    // Wrap-around across screen bounds without overlapping
                    if (car.x > w + 220) {
                        car.x = -160;
                        const lane0Blocked = allVehicles.some(o => o !== car && Math.abs(o.x - (-160)) < 110 && o.targetLane === 0);
                        car.targetLane = lane0Blocked ? 1 : 0;
                        car.laneY = laneCenters[car.targetLane];
                        car.lane = car.targetLane;
                    } else if (car.x < -240) {
                        car.x = w + 160;
                        const lane0Blocked = allVehicles.some(o => o !== car && Math.abs(o.x - (w + 160)) < 110 && o.targetLane === 0);
                        car.targetLane = lane0Blocked ? 1 : 0;
                        car.laneY = laneCenters[car.targetLane];
                        car.lane = car.targetLane;
                    }
                }

                // 4. DRAW ALL VEHICLES (Sorted by Y for realistic depth)
                const sortedVehicles = [...allVehicles].sort((a, b) => a.laneY - b.laneY);
                for (const v of sortedVehicles) {
                    const isEgo = v.isEgo === true;
                    const isHazard = isEgo && (telemetry.state === 'INTERVENTION' || telemetry.state === 'SAFE_STOP');
                    let label = v.label;
                    if (v.isFollower) {
                        label = v.isBraking ? 'FOLLOWER (BRAKING)' : (v.targetLane === 0 ? 'FOLLOWER (PASSING)' : 'FOLLOWER');
                    } else if (v.isEgo) {
                        label = v.isBraking ? 'YOU (BRAKING)' : (v.targetLane === 0 ? 'YOU (OVERTAKING)' : 'YOU (DRIVER)');
                    }
                    drawVehicle(v.x, v.laneY, v.type, v.color, label, v.isBraking, isHazard, v.turnSignal, isEgo);
                }

                // 5. FOLLOWER SAFETY GAP LINE (Drawn when follower is behind driver)
                if (followerVehicle.x < egoVehicle.x && Math.abs(followerVehicle.laneY - egoVehicle.laneY) < 30) {
                    const currentGapM = Math.max(2, (egoVehicle.x - (followerVehicle.x + followerVehicle.length)) * 0.18);
                    const gapColor = currentGapM < 12 ? '#ef4444' : (currentGapM < 22 ? '#f59e0b' : '#34d399');
                    ctx.strokeStyle = gapColor;
                    ctx.lineWidth = 2;
                    ctx.setLineDash([4, 4]);
                    ctx.beginPath();
                    ctx.moveTo(followerVehicle.x + followerVehicle.length, followerVehicle.laneY + 20);
                    ctx.lineTo(egoVehicle.x, egoVehicle.laneY + 20);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    ctx.fillStyle = gapColor;
                    ctx.font = 'bold 10px monospace';
                    ctx.fillText(`GAP: ${currentGapM.toFixed(1)}m`, (followerVehicle.x + followerVehicle.length + egoVehicle.x) / 2 - 25, followerVehicle.laneY + 34);
                }

            } catch(e) {
                console.error("Canvas draw error:", e);
            }

            requestAnimationFrame(drawRoad);
        }

        // Vehicle drawing: handles CAR, BIKE, TRUCK, SUV with Turn Signals & Brake Lights
        function drawVehicle(x, centerY, type, color, label, brakeLights, hazardLights, turnSignal = 'none', isEgo = false) {
            ctx.save();

            if (type === 'bike') {
                // MOTORBIKE
                const len = 42;
                const wid = 16;
                const y = centerY - wid / 2;

                ctx.fillStyle = 'rgba(0,0,0,0.3)';
                ctx.fillRect(x + 2, y + 2, len, wid);

                ctx.fillStyle = color;
                drawRoundedRect(ctx, x, y, len, wid, 4);
                ctx.fill();

                ctx.fillStyle = '#0f172a';
                ctx.fillRect(x + 2, y + 4, 8, 8);
                ctx.fillRect(x + len - 10, y + 4, 8, 8);

                ctx.fillStyle = '#f8fafc';
                ctx.beginPath();
                ctx.arc(x + len / 2, centerY, 5, 0, Math.PI * 2);
                ctx.fill();

                // Headlight Beam Cone (Facing East / Right)
                ctx.fillStyle = 'rgba(254, 240, 138, 0.25)';
                ctx.beginPath();
                ctx.moveTo(x + len, centerY);
                ctx.lineTo(x + len + 40, centerY - 15);
                ctx.lineTo(x + len + 40, centerY + 15);
                ctx.closePath();
                ctx.fill();

                // Brake Light (Taillight on the West / Left)
                if (brakeLights) {
                    ctx.fillStyle = '#ff0000';
                    ctx.shadowColor = '#ff0000';
                    ctx.shadowBlur = 10;
                    ctx.fillRect(x - 2, centerY - 3, 3, 6);
                    ctx.shadowBlur = 0;
                }

                // Turn Signals
                if (turnSignal === 'left' && Math.floor(Date.now() / 200) % 2 === 0) {
                    ctx.fillStyle = '#f59e0b';
                    ctx.fillRect(x + len - 4, y - 2, 4, 3);
                } else if (turnSignal === 'right' && Math.floor(Date.now() / 200) % 2 === 0) {
                    ctx.fillStyle = '#f59e0b';
                    ctx.fillRect(x + len - 4, y + wid - 1, 4, 3);
                }

                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 8px sans-serif';
                ctx.fillText(label, x, y - 4);

            } else if (type === 'truck') {
                // HEAVY SEMI-TRUCK / TRAILER
                const len = 125;
                const wid = 40;
                const y = centerY - wid / 2;

                ctx.fillStyle = 'rgba(0,0,0,0.35)';
                drawRoundedRect(ctx, x + 3, y + 4, len, wid, 4);
                ctx.fill();

                // Cargo Container
                ctx.fillStyle = color;
                drawRoundedRect(ctx, x, y, len - 34, wid, 4);
                ctx.fill();
                ctx.strokeStyle = 'rgba(255,255,255,0.3)';
                ctx.stroke();

                // Ribbed Container Stripes
                ctx.fillStyle = 'rgba(0,0,0,0.15)';
                for (let sx = x + 12; sx < x + len - 40; sx += 14) {
                    ctx.fillRect(sx, y + 3, 2, wid - 6);
                }

                // Cab Front (Facing East / Right)
                ctx.fillStyle = '#374151';
                drawRoundedRect(ctx, x + len - 32, y + 2, 32, wid - 4, 5);
                ctx.fill();

                // Windshield
                ctx.fillStyle = '#94a3b8';
                ctx.fillRect(x + len - 12, y + 5, 5, wid - 10);

                // Wheels (6 wheels)
                ctx.fillStyle = '#0f172a';
                ctx.fillRect(x + 8, y - 2, 12, 3);
                ctx.fillRect(x + 24, y - 2, 12, 3);
                ctx.fillRect(x + len - 20, y - 2, 12, 3);
                ctx.fillRect(x + 8, y + wid - 1, 12, 3);
                ctx.fillRect(x + 24, y + wid - 1, 12, 3);
                ctx.fillRect(x + len - 20, y + wid - 1, 12, 3);

                // Headlights
                ctx.fillStyle = '#fef08a';
                ctx.fillRect(x + len - 2, y + 4, 3, 7);
                ctx.fillRect(x + len - 2, y + wid - 11, 3, 7);

                // Brake Lights
                if (brakeLights) {
                    ctx.fillStyle = '#ff0000';
                    ctx.shadowColor = '#ff0000';
                    ctx.shadowBlur = 14;
                    ctx.fillRect(x - 3, y + 4, 4, 10);
                    ctx.fillRect(x - 3, y + wid - 14, 4, 10);
                    ctx.shadowBlur = 0;
                } else {
                    ctx.fillStyle = '#7f1d1d';
                    ctx.fillRect(x - 2, y + 5, 3, 7);
                    ctx.fillRect(x - 2, y + wid - 12, 3, 7);
                }

                // Turn Signals
                if (turnSignal === 'left' && Math.floor(Date.now() / 200) % 2 === 0) {
                    ctx.fillStyle = '#f59e0b';
                    ctx.fillRect(x + len - 8, y - 2, 6, 4);
                    ctx.fillRect(x + 4, y - 2, 6, 4);
                } else if (turnSignal === 'right' && Math.floor(Date.now() / 200) % 2 === 0) {
                    ctx.fillStyle = '#f59e0b';
                    ctx.fillRect(x + len - 8, y + wid - 2, 6, 4);
                    ctx.fillRect(x + 4, y + wid - 2, 6, 4);
                }

                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 8px sans-serif';
                ctx.fillText(label, x + 5, y - 5);

            } else {
                // SEDAN / CAR / SUV
                const len = type === 'suv' ? 82 : 74;
                const wid = type === 'suv' ? 38 : 34;
                const y = centerY - wid / 2;

                ctx.fillStyle = 'rgba(0,0,0,0.35)';
                drawRoundedRect(ctx, x + 2, y + 3, len, wid, 6);
                ctx.fill();

                // 4 Wheels
                ctx.fillStyle = '#0f172a';
                ctx.fillRect(x + 8, y - 2, 12, 3);
                ctx.fillRect(x + len - 20, y - 2, 12, 3);
                ctx.fillRect(x + 8, y + wid - 1, 12, 3);
                ctx.fillRect(x + len - 20, y + wid - 1, 12, 3);

                // Body
                ctx.fillStyle = color;
                drawRoundedRect(ctx, x, y, len, wid, 6);
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = isEgo ? 2 : 1;
                ctx.stroke();

                // Roof
                ctx.fillStyle = '#1e293b';
                drawRoundedRect(ctx, x + 18, y + 3, len - 38, wid - 6, 4);
                ctx.fill();

                // Windshield Front (Facing East / Right)
                ctx.fillStyle = '#94a3b8';
                drawRoundedRect(ctx, x + len - 27, y + 5, 5, wid - 10, 2);
                ctx.fill();

                // Rear Window
                drawRoundedRect(ctx, x + 20, y + 5, 4, wid - 10, 2);
                ctx.fill();

                // Headlights Cone Beam
                ctx.fillStyle = 'rgba(254, 240, 138, 0.2)';
                ctx.beginPath();
                ctx.moveTo(x + len, centerY);
                ctx.lineTo(x + len + 50, centerY - 20);
                ctx.lineTo(x + len + 50, centerY + 20);
                ctx.closePath();
                ctx.fill();

                // Headlight Bulbs
                ctx.fillStyle = '#fef08a';
                ctx.fillRect(x + len - 3, y + 2, 3, 6);
                ctx.fillRect(x + len - 3, y + wid - 8, 3, 6);

                // Brake Lights (Taillights on West / Left side)
                if (brakeLights) {
                    ctx.fillStyle = '#ff0000';
                    ctx.shadowColor = '#ff0000';
                    ctx.shadowBlur = 14;
                    ctx.fillRect(x - 3, y + 2, 4, 8);
                    ctx.fillRect(x - 3, y + wid - 10, 4, 8);
                    ctx.shadowBlur = 0;
                } else {
                    ctx.fillStyle = '#7f1d1d';
                    ctx.fillRect(x - 2, y + 3, 3, 5);
                    ctx.fillRect(x - 2, y + wid - 8, 3, 5);
                }

                // Turn Signals (Amber Blinking)
                if (turnSignal === 'left' && Math.floor(Date.now() / 200) % 2 === 0) {
                    ctx.fillStyle = '#f59e0b';
                    ctx.shadowColor = '#f59e0b';
                    ctx.shadowBlur = 8;
                    ctx.fillRect(x + 2, y - 2, 6, 4);
                    ctx.fillRect(x + len - 8, y - 2, 6, 4);
                    ctx.shadowBlur = 0;
                } else if (turnSignal === 'right' && Math.floor(Date.now() / 200) % 2 === 0) {
                    ctx.fillStyle = '#f59e0b';
                    ctx.shadowColor = '#f59e0b';
                    ctx.shadowBlur = 8;
                    ctx.fillRect(x + 2, y + wid - 2, 6, 4);
                    ctx.fillRect(x + len - 8, y + wid - 2, 6, 4);
                    ctx.shadowBlur = 0;
                }

                // Hazard Lights
                if (hazardLights && Math.floor(Date.now() / 250) % 2 === 0) {
                    ctx.fillStyle = '#f59e0b';
                    ctx.shadowColor = '#f59e0b';
                    ctx.shadowBlur = 10;
                    ctx.fillRect(x - 2, y - 2, 4, 4);
                    ctx.fillRect(x - 2, y + wid - 2, 4, 4);
                    ctx.fillRect(x + len - 2, y - 2, 4, 4);
                    ctx.fillRect(x + len - 2, y + wid - 2, 4, 4);
                    ctx.shadowBlur = 0;
                }

                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 8px sans-serif';
                ctx.fillText(label, x + 5, y - 5);
            }

            ctx.restore();
        }

        // 5. TELEMETRY POLLING & DYNAMIC TIMER LOGIC
        setInterval(() => {
            fetch('/api/telemetry')
                .then(r => r.json())
                .then(d => {
                    telemetry = d;
                    const closedSec = d.closed_s;

                    // Simulation Mode Badge & Input Source Label
                    const simBadge = document.getElementById('simBadge');
                    const srcLabel = document.getElementById('inputSourceLabel');
                    if (d.sim_active) {
                        simBadge.style.display = 'block';
                        srcLabel.innerText = 'SIMULATION (D KEY)';
                        srcLabel.style.color = '#f97316';
                    } else {
                        simBadge.style.display = 'none';
                        srcLabel.innerText = 'LIVE WEBCAM AI';
                        srcLabel.style.color = '#38bdf8';
                    }

                    // DYNAMIC SLEEP TIMER BACKGROUND & HIGHLIGHT COLORS
                    const banner = document.getElementById('sleepBanner');
                    const timerVal = document.getElementById('topSleepTimer');
                    const label = document.getElementById('sleepLabel');
                    timerVal.innerText = closedSec.toFixed(1) + ' s';

                    if (closedSec <= 0.05) {
                        // Timer = 0: Clean Light Green
                        banner.style.background = 'rgba(16, 185, 129, 0.18)';
                        banner.style.borderColor = '#10b981';
                        timerVal.style.background = '#064e3b';
                        timerVal.style.color = '#a7f3d0';
                        label.innerText = '🟢 Driver Awake & Attentive:';
                    } else if (closedSec < 2.0) {
                        // 0 < Timer < 2.0: Soft Yellow
                        banner.style.background = 'rgba(234, 179, 8, 0.25)';
                        banner.style.borderColor = '#eab308';
                        timerVal.style.background = '#713f12';
                        timerVal.style.color = '#fef08a';
                        label.innerText = '🟡 Eye Closure Detected:';
                    } else if (closedSec < 5.0) {
                        // 2.0 <= Timer < 5.0: Warning Orange (Buzzer ON!)
                        banner.style.background = 'rgba(249, 115, 22, 0.38)';
                        banner.style.borderColor = '#f97316';
                        timerVal.style.background = '#7c2d12';
                        timerVal.style.color = '#ffedd5';
                        label.innerText = '🟠 WARNING: SLEEP DETECTED! BUZZER ON';
                    } else {
                        // Timer >= 5.0: Critical Red (Braking Engaged!)
                        banner.style.background = 'rgba(239, 68, 68, 0.45)';
                        banner.style.borderColor = '#ef4444';
                        timerVal.style.background = '#991b1b';
                        timerVal.style.color = '#fee2e2';
                        label.innerText = '🔴 CRITICAL: ASLEEP > 5.0s! AUTONOMOUS BRAKING ACTIVATED';
                    }

                    // SOUND BUZZER TRIGGER: Plays if closed >= 2.0s or state is WARNING/INTERVENTION
                    if (closedSec >= 2.0 || d.state === 'WARNING' || d.state === 'INTERVENTION') {
                        if (!buzzerInterval) {
                            const freq = closedSec >= 5.0 ? 980 : 750;
                            const speed = closedSec >= 5.0 ? 200 : 320;
                            buzzerInterval = setInterval(() => {
                                playAlertTone(freq, 160);
                            }, speed);
                        }
                    } else {
                        if (buzzerInterval) {
                            clearInterval(buzzerInterval);
                            buzzerInterval = null;
                        }
                    }

                    // LABELS
                    document.getElementById('scoreText').innerText = d.drowsiness_score.toFixed(1) + ' / 100';
                    document.getElementById('eyeStatus').innerText = (d.is_closed ? 'CLOSED' : 'OPEN') + ` (EAR: ${d.ear.toFixed(2)})`;
                    document.getElementById('perclosVal').innerText = (d.perclos * 100).toFixed(1) + '%';
                    document.getElementById('pitchVal').innerText = d.pitch.toFixed(1) + '°';

                    document.getElementById('egoSpeedText').innerText = Math.round(d.ego_speed_kmh) + ' km/h';

                    const followerLiveSpeed = Math.round(followerVehicle.speed);
                    if (followerVehicle.isBraking) {
                        document.getElementById('follSpeedText').innerText = followerLiveSpeed + ' km/h (BRAKING)';
                        document.getElementById('follSpeedText').style.color = '#ef4444';
                    } else {
                        document.getElementById('follSpeedText').innerText = followerLiveSpeed + ' km/h';
                        document.getElementById('follSpeedText').style.color = '#f87171';
                    }

                    if (followerVehicle.targetLane === 0 && followerVehicle.x > egoVehicle.x - 50) {
                        document.getElementById('gapDistanceText').innerText = 'OVERTAKING (LANE 0)';
                        document.getElementById('gapDistanceText').style.color = '#38bdf8';
                    } else if (followerVehicle.isBraking) {
                        const g = Math.max(2, (egoVehicle.x - (followerVehicle.x + followerVehicle.length)) * 0.18);
                        document.getElementById('gapDistanceText').innerText = 'BRAKING (' + g.toFixed(1) + ' m)';
                        document.getElementById('gapDistanceText').style.color = '#f87171';
                    } else {
                        const g = Math.max(2, (egoVehicle.x - (followerVehicle.x + followerVehicle.length)) * 0.18);
                        document.getElementById('gapDistanceText').innerText = g.toFixed(1) + ' m';
                        document.getElementById('gapDistanceText').style.color = '#fbbf24';
                    }

                    document.getElementById('strategyBadge').innerText = d.strategy.toUpperCase() + ' BRAKING';

                    const riskEl = document.getElementById('riskLevelText');
                    if (followerVehicle.isBraking) {
                        riskEl.innerText = 'BRAKING BEHIND';
                        riskEl.style.color = '#f59e0b';
                    } else {
                        riskEl.innerText = d.risk_level;
                        riskEl.style.color = d.risk_level === 'SAFE' ? '#34d399' : (d.risk_level === 'CAUTION' ? '#fbbf24' : '#f87171');
                    }

                    // Control state & banner
                    const stateBanner = document.getElementById('stateBanner');
                    const ctrlText = document.getElementById('controlStatusText');

                    if (d.state === 'ALERT') {
                        const statusMsg = d.ego_speed_kmh < 95 ? 'DRIVER HAS CONTROL (ACCELERATING)' : 'DRIVER HAS CONTROL (NORMAL DRIVING)';
                        stateBanner.innerText = statusMsg;
                        stateBanner.className = 'alert-banner state-alert';
                        ctrlText.innerText = statusMsg;
                        ctrlText.style.color = '#10b981';
                    } else if (d.state === 'CAUTION') {
                        stateBanner.innerText = 'CAUTION: ATTENTION REDUCING';
                        stateBanner.className = 'alert-banner state-caution';
                        ctrlText.innerText = 'ATTENTION SLIPPING';
                        ctrlText.style.color = '#f59e0b';
                    } else if (d.state === 'WARNING') {
                        stateBanner.innerText = 'WARNING: WAKE UP DRIVER! BUZZER ON';
                        stateBanner.className = 'alert-banner state-warning';
                        ctrlText.innerText = 'WARNING: DRIVER ASLEEP';
                        ctrlText.style.color = '#f97316';
                    } else {
                        stateBanner.innerText = 'INTERVENTION: AUTONOMOUS BRAKING ACTIVE';
                        stateBanner.className = 'alert-banner state-intervention';
                        ctrlText.innerText = 'AUTONOMOUS BRAKING';
                        ctrlText.style.color = '#ef4444';
                    }

                    // Button synchronization
                    const btnDrive = document.getElementById('btnDrive');
                    if (d.ego_speed_kmh < 5) {
                        btnDrive.style.filter = 'brightness(1.3)';
                        btnDrive.innerText = '🚀 START DRIVING (ACCELERATE)';
                    } else {
                        btnDrive.style.filter = 'brightness(1)';
                        btnDrive.innerText = '🚀 Accelerate / Cruise [R]';
                    }
                })
                .catch(err => console.debug(err));
        }, 100);

        requestAnimationFrame(drawRoad);
    </script>
</body>
</html>
"""


def start_web_dashboard(config: AppConfig, port: int = 5000) -> None:
    """Launch local web server for dashboard streaming with top-down 2D multi-lane animation."""
    app = Flask(__name__)

    # Vision & Camera
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

    # Simulation components
    current_strategy = [config.braking.strategy]
    ego = Vehicle(config.vehicle, initial_position_m=0.0)
    follower = FollowerVehicle(config.follower, config.vehicle, v2v_enabled=config.braking.v2v_alert)
    controller = [create_controller(current_strategy[0], config.braking, config.vehicle)]
    safety_calc = SafetyMetricsCalculator(vehicle_length_m=config.vehicle.length_m)

    # State flags for simulation mode and sleep timing
    sim_mode = {"active": False, "sim_start_time": 0.0}

    shared = {
        "frame_bytes": None,
        "is_manually_accelerating": False,
        "telemetry": {
            "state": "ALERT",
            "state_desc": "DRIVER HAS CONTROL",
            "drowsiness_score": 0.0,
            "is_closed": False,
            "ear": 0.33,
            "closed_s": 0.0,
            "perclos": 0.0,
            "pitch": 0.0,
            "yawns": 0,
            "ego_speed_kmh": 100.0,
            "ego_brake_intensity": 0.0,
            "foll_speed_kmh": 100.0,
            "foll_brake": False,
            "gap_m": 30.0,
            "risk_level": "SAFE",
            "strategy": current_strategy[0],
            "sim_active": False,
        },
    }

    def sim_worker():
        sim_time = 0.0
        dt = 0.033
        target_cruise_speed_mps = config.vehicle.initial_speed_kmh / 3.6

        while True:
            ret, frame, _ = camera.read()
            if not ret or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for camera...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            res = tracker.process_frame(frame)
            if res and res.face_detected and res.landmarks is not None:
                lms = res.landmarks
                if not calibrator.is_complete:
                    _, _, ear = calculate_ear(lms)
                    mar = calculate_mar(lms)
                    p, y, r = calculate_head_pose(lms, frame.shape[1], frame.shape[0])
                    if calibrator.update(ear, mar, p, y, r, True):
                        extractor.set_profile(calibrator.get_profile())

                features = extractor.update(lms, True, frame.shape[1], frame.shape[0])
                for pt in lms[::3]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 128), -1)
            else:
                features = extractor.update(None, False, frame.shape[1], frame.shape[0])

            # 1. HANDLE SIMULATION (D KEY) VS CAMERA INPUT
            if sim_mode["active"]:
                # When D is pressed, driver is already asleep 2.0s and counts up from there!
                # Camera input is IGNORED while in simulation mode!
                elapsed_sim = time.time() - sim_mode["sim_start_time"]
                sleep_duration = 2.0 + elapsed_sim
                is_eye_closed = True
                reading = fuser.process(features)
                reading.score = min(100.0, 65.0 + (sleep_duration * 6.0))
            else:
                # Real camera input is active!
                reading = fuser.process(features)
                sleep_duration = features.eye_closed_duration_s
                is_eye_closed = features.is_eye_closed

            # 2. STATE MACHINE TRANSITIONS
            if not is_eye_closed and sleep_duration < 5.0:
                if state_machine.current_state in (SystemState.WARNING, SystemState.CAUTION, SystemState.INTERVENTION):
                    state_machine._transition(SystemState.ALERT, "Driver alert restored before 5s braking threshold", sim_time)

            state = state_machine.update(reading, features.no_face_duration_s if not sim_mode["active"] else 0.0, ego.speed_mps, False)

            # Pygame native audio buzzer plays on laptop hardware speakers!
            if sleep_duration >= 2.0 or state in (SystemState.WARNING, SystemState.INTERVENTION):
                alert_mgr.update(state, 1.0 if sleep_duration >= 5.0 else 0.5)

            # Braking intervention engages ONLY if sleep exceeds 5.0 seconds
            is_braking_active = (sleep_duration >= 5.0) or (state == SystemState.INTERVENTION and is_eye_closed)

            sim_time += dt
            v2v_active = is_braking_active or (sleep_duration >= 2.0)

            ego_st = ego.get_state()
            foll_st = follower.vehicle.get_state()

            # 3. DIFFERENT CONTROLLER BRAKING ACTIONS
            if is_braking_active:
                # Controller executes chosen strategy: abrupt, gradual, or adaptive
                a_cmd = controller[0].step(ego_st, foll_st, dt, True)
            else:
                # Driver in control: smoothly accelerate back to cruising speed!
                if ego_st.speed_mps < target_cruise_speed_mps:
                    a_cmd = 2.4  # Positive acceleration back to 100 km/h
                else:
                    a_cmd = 0.0  # Cruising

                if ego_st.speed_kmh > 40.0 and follower.has_reacted:
                    follower.has_reacted = False
                    follower.perceived_brake_time = None

            ego_st = ego.step(a_cmd, dt, hazards=is_braking_active)
            foll_st = follower.step(ego_st, dt, sim_time, v2v_alert_active=v2v_active)
            safety = safety_calc.compute(ego_st, foll_st, sim_time)

            # Text overlay on video
            cv2.putText(frame, f"STATUS: {state.value}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if state == SystemState.ALERT else (0, 0, 255), 2)
            cv2.putText(frame, f"Sleep Timer: {sleep_duration:.1f}s | Mode: {'SIMULATION' if sim_mode['active'] else 'LIVE AI'}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            _, jpeg = cv2.imencode(".jpg", frame)
            shared["frame_bytes"] = jpeg.tobytes()

            state_desc_map = {
                SystemState.ALERT: "DRIVER HAS CONTROL",
                SystemState.CAUTION: "ATTENTION REDUCING",
                SystemState.WARNING: "WARNING: AWAKE DRIVER! BUZZER ON",
                SystemState.INTERVENTION: "AUTONOMOUS BRAKING ENGAGED (>5s)",
                SystemState.SAFE_STOP: "VEHICLE STOPPED",
                SystemState.RECOVERY: "DRIVER RESUMED CONTROL",
            }

            shared["telemetry"] = {
                "state": state.value,
                "state_desc": state_desc_map.get(state, "DRIVER HAS CONTROL"),
                "drowsiness_score": float(reading.score),
                "is_closed": bool(is_eye_closed),
                "ear": float(features.ear_smooth),
                "closed_s": float(sleep_duration),
                "perclos": float(features.perclos),
                "pitch": float(features.pitch_rel),
                "yawns": int(features.yawn_count),
                "ego_speed_kmh": float(ego_st.speed_kmh),
                "ego_brake_intensity": float(ego_st.brake_intensity),
                "foll_speed_kmh": float(foll_st.speed_kmh),
                "foll_brake": bool(foll_st.brake_lights_on),
                "gap_m": float(safety.gap_m),
                "risk_level": safety.risk_level,
                "strategy": current_strategy[0],
                "sim_active": bool(sim_mode["active"]),
            }
            time.sleep(dt)

    t = threading.Thread(target=sim_worker, daemon=True)
    t.start()

    @app.route("/")
    def index():
        return render_template_string(HTML_PAGE, beep_uri=BEEP_DATA_URI)

    @app.route("/video_feed")
    def video_feed():
        def gen():
            while True:
                if shared["frame_bytes"] is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + shared["frame_bytes"] + b"\r\n")
                time.sleep(0.04)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/telemetry")
    def telemetry():
        return jsonify(shared["telemetry"])

    @app.route("/api/simulate_drowsy", methods=["POST"])
    def simulate_drowsy():
        sim_mode["active"] = True
        sim_mode["sim_start_time"] = time.time()
        fuser.set_simulated_drowsiness(True)
        return jsonify({"status": "sim_started"})

    @app.route("/api/wakeup", methods=["POST"])
    def wakeup():
        sim_mode["active"] = False
        fuser.set_simulated_drowsiness(False)
        state_machine._transition(SystemState.ALERT, "Driver manual wake up - camera restored", 0.0)
        follower.has_reacted = False
        follower.perceived_brake_time = None
        return jsonify({"status": "camera_restored"})

    @app.route("/api/start_driving", methods=["POST"])
    def start_driving_api():
        sim_mode["active"] = False
        fuser.set_simulated_drowsiness(False)
        state_machine._transition(SystemState.ALERT, "Driver in control - accelerating", 0.0)
        follower.has_reacted = False
        follower.perceived_brake_time = None
        return jsonify({"status": "driving_resumed"})

    @app.route("/api/strategy", methods=["POST"])
    def change_strategy():
        data = request.get_json() or {}
        new_strat = data.get("strategy", "adaptive")
        if new_strat in ("abrupt", "gradual", "adaptive"):
            current_strategy[0] = new_strat
            config.braking.strategy = new_strat
            controller[0] = create_controller(new_strat, config.braking, config.vehicle)
            logger.info("Switched to strategy: %s", new_strat)
        return jsonify({"strategy": current_strategy[0]})

    import webbrowser
    def auto_open_browser():
        time.sleep(1.2)
        try:
            logger.info("Opening web browser at http://localhost:%d", port)
            webbrowser.open(f"http://localhost:{port}")
        except Exception as e:
            logger.debug("Failed to auto-open browser: %s", e)

    threading.Thread(target=auto_open_browser, daemon=True).start()

    print("\n" + "=" * 70)
    print("  >> LIVE WEB DASHBOARD IS READY!")
    print(f"  >> AUTOMATICALLY OPENING IN YOUR BROWSER: http://localhost:{port}")
    print("  >> (If browser does not pop up, click: http://localhost:5000)")
    print("=" * 70 + "\n")

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

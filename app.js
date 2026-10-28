/**
 * SAFEDRIVE AI - CLIENT-SIDE AUTONOMOUS MONITORING & TRAFFIC SIMULATION ENGINE
 * 100% Client-Side WebAssembly AI • Ready for Vercel Static / Edge Hosting
 */

(function () {
    'use strict';

    // =========================================================================
    // 1. DUAL AUDIO BUZZER ENGINE (WEB AUDIO API SYNTHESIZER)
    // =========================================================================
    let audioCtx = null;
    let isAudioMuted = false;
    let buzzerInterval = null;

    function initAudio() {
        if (!audioCtx) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (AudioContextClass) audioCtx = new AudioContextClass();
        }
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        const notice = document.getElementById('audioNotice');
        if (notice) notice.style.display = 'none';
    }

    window.enableAudio = function () {
        initAudio();
        playAlertTone(750, 160);
    };

    window.toggleAudio = function () {
        initAudio();
        isAudioMuted = !isAudioMuted;
        const btn = document.getElementById('audioToggleBtn');
        if (btn) btn.innerText = isAudioMuted ? '🔇 Buzzer: OFF' : '🔊 Buzzer: ACTIVE';
    };

    function playAlertTone(frequency, durationMs) {
        if (isAudioMuted) return;
        initAudio();
        if (!audioCtx || audioCtx.state !== 'running') return;

        try {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);

            gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + (durationMs / 1000));

            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + (durationMs / 1000));
        } catch (e) {
            // Audio context error fallback
        }
    }

    document.addEventListener('click', initAudio, { once: true });
    document.addEventListener('keydown', initAudio, { once: true });

    // =========================================================================
    // 2. MATHEMATICAL & BIOMETRIC GEOMETRY HELPERS
    // =========================================================================
    function dist2D(p1, p2) {
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function calculateEAR(landmarks) {
        // Left eye indices: 33 (corner), 160, 158 (top), 133 (corner), 153, 144 (bottom)
        const leftH1 = dist2D(landmarks[160], landmarks[144]);
        const leftH2 = dist2D(landmarks[158], landmarks[153]);
        const leftW = dist2D(landmarks[33], landmarks[133]);
        const leftEAR = (leftH1 + leftH2) / (2.0 * Math.max(0.001, leftW));

        // Right eye indices: 362 (corner), 385, 387 (top), 263 (corner), 373, 380 (bottom)
        const rightH1 = dist2D(landmarks[385], landmarks[380]);
        const rightH2 = dist2D(landmarks[387], landmarks[373]);
        const rightW = dist2D(landmarks[362], landmarks[263]);
        const rightEAR = (rightH1 + rightH2) / (2.0 * Math.max(0.001, rightW));

        return {
            left: leftEAR,
            right: rightEAR,
            avg: (leftEAR + rightEAR) / 2.0
        };
    }

    function calculateMAR(landmarks) {
        // Lip vertical: 13 (upper inner), 14 (lower inner)
        // Lip horizontal: 61 (left corner), 291 (right corner)
        const v = dist2D(landmarks[13], landmarks[14]);
        const h = dist2D(landmarks[61], landmarks[291]);
        return v / Math.max(0.001, h);
    }

    function estimatePitch(landmarks) {
        // Nose tip: 1, Chin: 152, Midpoint between eyes: 168
        const nose = landmarks[1];
        const chin = landmarks[152];
        const brow = landmarks[10];

        const faceHeight = Math.max(0.001, dist2D(chin, brow));
        // Vertical offset of nose relative to brow-chin axis
        const midY = (chin.y + brow.y) / 2.0;
        const offset = (nose.y - midY) / faceHeight;
        return offset * 45.0; // Scaled pitch degrees
    }

    // =========================================================================
    // 3. CALIBRATION & FEATURE SMOOTHING ENGINE
    // =========================================================================
    const calibration = {
        isComplete: false,
        durationSeconds: 5.0,
        startTime: null,
        earSamples: [],
        pitchSamples: [],
        baselineEAR: 0.32,
        closedThreshold: 0.24,
        neutralPitch: 0.0
    };

    const eyeHistory = []; // Rolling buffer for PERCLOS (past 180 frames)
    let eyeClosedStartTime = null;
    let eyeClosedDuration = 0.0;
    let earSmooth = 0.33;

    function updateDriverFeatures(landmarks, nowTime) {
        if (!landmarks) {
            return {
                ear: earSmooth,
                mar: 0.1,
                pitch: 0.0,
                isClosed: false,
                closedDuration: 0.0,
                perclos: 0.0
            };
        }

        const earData = calculateEAR(landmarks);
        const mar = calculateMAR(landmarks);
        const pitch = estimatePitch(landmarks);

        // Exponential smoothing on EAR (alpha = 0.3)
        earSmooth = 0.3 * earData.avg + 0.7 * earSmooth;

        // 1. Calibration Phase (First 5 seconds)
        if (!calibration.isComplete) {
            if (calibration.startTime === null) calibration.startTime = nowTime;
            calibration.earSamples.push(earSmooth);
            calibration.pitchSamples.push(pitch);

            const elapsed = (nowTime - calibration.startTime) / 1000.0;
            const progress = Math.min(100, (elapsed / calibration.durationSeconds) * 100);

            const calibEl = document.getElementById('metricCalib');
            if (calibEl) calibEl.innerText = `CALIBRATING (${(calibration.durationSeconds - elapsed).toFixed(1)}s)`;

            if (elapsed >= calibration.durationSeconds && calibration.earSamples.length > 20) {
                const sumEAR = calibration.earSamples.reduce((a, b) => a + b, 0);
                calibration.baselineEAR = sumEAR / calibration.earSamples.length;
                calibration.closedThreshold = calibration.baselineEAR * 0.75;

                const sumPitch = calibration.pitchSamples.reduce((a, b) => a + b, 0);
                calibration.neutralPitch = sumPitch / calibration.pitchSamples.length;
                calibration.isComplete = true;

                if (calibEl) {
                    calibEl.innerText = `READY (Baseline EAR: ${calibration.baselineEAR.toFixed(2)})`;
                    calibEl.className = 'metric-value text-green';
                }
            }
        }

        // 2. Eye Closure Detection
        const isClosed = earSmooth < calibration.closedThreshold;

        if (isClosed) {
            if (eyeClosedStartTime === null) eyeClosedStartTime = nowTime;
            eyeClosedDuration = (nowTime - eyeClosedStartTime) / 1000.0;
        } else {
            eyeClosedStartTime = null;
            eyeClosedDuration = 0.0;
        }

        // 3. PERCLOS Buffer
        eyeHistory.push(isClosed ? 1 : 0);
        if (eyeHistory.length > 180) eyeHistory.shift();
        const perclos = eyeHistory.reduce((a, b) => a + b, 0) / Math.max(1, eyeHistory.length);

        return {
            ear: earSmooth,
            mar: mar,
            pitch: pitch - calibration.neutralPitch,
            isClosed: isClosed,
            closedDuration: eyeClosedDuration,
            perclos: perclos
        };
    }

    // =========================================================================
    // 4. DROWSINESS FUSION & SAFETY ESCALATION STATE MACHINE
    // =========================================================================
    const SystemState = {
        ALERT: 'ALERT',
        CAUTION: 'CAUTION',
        WARNING: 'WARNING',
        INTERVENTION: 'INTERVENTION',
        SAFE_STOP: 'SAFE_STOP',
        RECOVERY: 'RECOVERY'
    };

    let currentState = SystemState.ALERT;
    let drowsinessScore = 0.0;
    let activeStrategy = 'adaptive'; // 'abrupt' | 'gradual' | 'adaptive'

    // Manual Simulation Mode (D key)
    const simMode = {
        active: false,
        startTime: null
    };

    function computeDrowsinessScore(features, isSimulated) {
        if (isSimulated) {
            const elapsed = (performance.now() - simMode.startTime) / 1000.0;
            const dur = 2.0 + elapsed;
            return {
                score: Math.min(100, 65.0 + dur * 6.5),
                closedDuration: dur,
                isClosed: true
            };
        }

        // Multi-signal score fusion: Duration (45%) + PERCLOS (30%) + Pitch nod (15%) + Yawn (10%)
        let score = 0;

        // Eye closure duration component
        if (features.closedDuration > 0) {
            score += Math.min(50, features.closedDuration * 16.0);
        }

        // PERCLOS component
        score += Math.min(30, features.perclos * 100.0 * 0.6);

        // Head pitch nod component (pitch < -10 deg = head drop)
        if (features.pitch < -8.0) {
            score += Math.min(15, Math.abs(features.pitch) * 1.2);
        }

        // Mouth yawn component
        if (features.mar > 0.45) {
            score += 10;
        }

        return {
            score: Math.min(100.0, score),
            closedDuration: features.closedDuration,
            isClosed: features.isClosed
        };
    }

    function updateStateMachine(fusedScore, closedDur, isClosed, egoSpeed) {
        const prev = currentState;

        if (egoSpeed <= 0.5 && (prev === SystemState.INTERVENTION || prev === SystemState.SAFE_STOP)) {
            currentState = SystemState.SAFE_STOP;
        } else if (closedDur >= 5.0 || (prev === SystemState.INTERVENTION && isClosed)) {
            currentState = SystemState.INTERVENTION;
        } else if (closedDur >= 2.0 || fusedScore >= 65.0) {
            currentState = SystemState.WARNING;
        } else if (fusedScore >= 45.0 || closedDur >= 1.0) {
            currentState = SystemState.CAUTION;
        } else {
            currentState = SystemState.ALERT;
        }

        return currentState;
    }

    // =========================================================================
    // 5. VEHICLE DYNAMICS & 3 BRAKING CONTROLLERS
    // =========================================================================
    const vehiclePhysics = {
        speedMps: 100 / 3.6, // 100 km/h cruising
        targetSpeedMps: 100 / 3.6,
        accelMps2: 0.0,
        brakeIntensity: 0.0,
        frictionMu: 0.8, // Dry asphalt
        g: 9.81,

        // Gradual controller tracking
        gradualCurrentDecel: 0.0,

        resetDriving: function () {
            this.targetSpeedMps = 100 / 3.6;
        },

        step: function (dt, isBrakingActive, followerGapM, followerRelSpeedMps) {
            if (isBrakingActive) {
                // Execute chosen braking controller
                let aCmd = 0.0;
                const aMaxBrake = this.frictionMu * this.g; // 7.848 m/s^2

                if (activeStrategy === 'abrupt') {
                    // Controller 1: Abrupt Full Clamp Stop
                    aCmd = -aMaxBrake;
                    this.brakeIntensity = 1.0;
                } else if (activeStrategy === 'gradual') {
                    // Controller 2: Jerk-Limited Smooth Deceleration
                    const jerkLimit = 2.2; // m/s^3
                    this.gradualCurrentDecel = Math.min(4.5, this.gradualCurrentDecel + jerkLimit * dt);
                    aCmd = -this.gradualCurrentDecel;
                    this.brakeIntensity = Math.min(1.0, this.gradualCurrentDecel / aMaxBrake);
                } else {
                    // Controller 3: Adaptive Gap-Aware Lookahead Controller
                    // Modulates deceleration to avoid rear collision with follower
                    const ttc = followerRelSpeedMps > 0 ? followerGapM / followerRelSpeedMps : 999.0;
                    let targetDecel = 3.5;

                    if (followerGapM < 18.0 || ttc < 2.5) {
                        // Follower is close behind: soften braking to allow follower reaction
                        targetDecel = 2.2;
                    } else if (this.speedMps < 15.0) {
                        // Near full stop: firmly halt vehicle
                        targetDecel = 5.0;
                    }

                    aCmd = -Math.min(aMaxBrake, targetDecel);
                    this.brakeIntensity = Math.min(1.0, targetDecel / aMaxBrake);
                }

                this.accelMps2 = aCmd;
                this.speedMps = Math.max(0, this.speedMps + aCmd * dt);
            } else {
                // Driver in Control: smoothly accelerate back to cruising speed (100 km/h)
                this.gradualCurrentDecel = 0.0;
                this.brakeIntensity = 0.0;

                if (this.speedMps < this.targetSpeedMps) {
                    this.accelMps2 = 2.4; // 2.4 m/s^2 positive acceleration
                    this.speedMps = Math.min(this.targetSpeedMps, this.speedMps + this.accelMps2 * dt);
                } else {
                    this.accelMps2 = 0.0;
                }
            }

            return {
                speedKmh: this.speedMps * 3.6,
                isBraking: this.brakeIntensity > 0.05,
                brakeIntensity: this.brakeIntensity
            };
        }
    };

    // =========================================================================
    // 6. HIGHWAY 2D MULTI-VEHICLE SIMULATION & INTELLIGENT OVERTAKING
    // =========================================================================
    const canvas = document.getElementById('roadCanvas');
    const ctx = canvas.getContext('2d');

    let worldX = 0;

    // Universal rounded rectangle helper
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

    const egoCar = {
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

    const followerCar = {
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

    const trafficVehicles = [
        { id: 1, x: 620, lane: 1, targetLane: 1, laneY: 0, speed: 80, desiredSpeed: 80, length: 125, width: 40, type: 'truck', color: '#8b5cf6', label: 'SEMI-TRUCK', turnSignal: 'none', isBraking: false },
        { id: 2, x: 120, lane: 0, targetLane: 0, laneY: 0, speed: 112, desiredSpeed: 112, length: 42, width: 18, type: 'bike', color: '#38bdf8', label: 'BIKE', turnSignal: 'none', isBraking: false },
        { id: 3, x: 760, lane: 0, targetLane: 0, laneY: 0, speed: 104, desiredSpeed: 104, length: 74, width: 34, type: 'car', color: '#10b981', label: 'SEDAN', turnSignal: 'none', isBraking: false },
        { id: 4, x: -90, lane: 1, targetLane: 1, laneY: 0, speed: 95, desiredSpeed: 95, length: 82, width: 38, type: 'suv', color: '#f59e0b', label: 'SUV', turnSignal: 'none', isBraking: false },
    ];

    function isLaneSafe(veh, targetLane, allVehicles, laneCenters, laneH) {
        const laneYTarget = laneCenters[targetLane];
        for (const other of allVehicles) {
            if (other === veh) continue;

            const inTarget = (other.targetLane === targetLane) ||
                (Math.abs(other.laneY - laneYTarget) < laneH * 0.55);
            if (!inTarget) continue;

            // Longitudinal overlap check
            const overlap = (veh.x + veh.length + 35 > other.x) && (other.x + other.length + 35 > veh.x);
            if (overlap) return false;

            // Approaching from behind in target lane
            if (other.x < veh.x) {
                const rearDist = veh.x - (other.x + other.length);
                if (rearDist < 85 && (other.speed >= veh.speed - 5)) return false;
            }

            // Directly ahead in target lane
            if (other.x > veh.x) {
                const frontDist = other.x - (veh.x + veh.length);
                if (frontDist < 80) return false;
            }
        }
        return true;
    }

    function renderSimulation(egoSpeedKmh, isEgoBraking, isHazard) {
        const w = canvas.width;
        const h = canvas.height;

        // 1. CLEAR & ASPHALT
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, w, h);

        const roadY = 40;
        const roadH = h - 80; // 240px
        const laneH = roadH / 2; // 120px
        const laneCenters = [roadY + laneH * 0.5, roadY + laneH * 1.5]; // Lane 0 (Top: 100), Lane 1 (Bottom: 220)

        // Shoulders
        ctx.fillStyle = '#14532d';
        ctx.fillRect(0, 0, w, roadY);
        ctx.fillRect(0, roadY + roadH, w, h - (roadY + roadH));

        // Asphalt Road
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, roadY, w, roadH);

        // 2. ROAD MOTION: SCROLLS CONTINUOUSLY TOWARDS THE LEFT BASED ON SPEED
        const speedFactor = Math.max(0, egoSpeedKmh / 100);
        worldX += 9 * speedFactor;

        // Rumble curbs (Red & White blocks moving LEFT)
        const curbW = 28;
        const curbOffset = worldX % (curbW * 2);
        for (let cx = -curbW * 2; cx < w + curbW * 2; cx += curbW) {
            const drawX = cx - curbOffset;
            ctx.fillStyle = (Math.floor(cx / curbW) % 2 === 0) ? '#dc2626' : '#f8fafc';
            ctx.fillRect(drawX, roadY - 7, curbW, 7);
            ctx.fillRect(drawX, roadY + roadH, curbW, 7);
        }

        // White solid edges
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(0, roadY); ctx.lineTo(w, roadY);
        ctx.moveTo(0, roadY + roadH); ctx.lineTo(w, roadY + roadH);
        ctx.stroke();

        // Center dashed dividing line (flowing LEFT)
        ctx.strokeStyle = '#f8fafc';
        ctx.lineWidth = 3;
        ctx.setLineDash([32, 28]);
        ctx.lineDashOffset = worldX; // Shifts dash pattern LEFT
        ctx.beginPath();
        ctx.moveTo(0, roadY + laneH);
        ctx.lineTo(w, roadY + laneH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Highway direction arrows painted on asphalt moving LEFT
        ctx.fillStyle = 'rgba(255, 255, 255, 0.18)';
        ctx.font = 'bold 13px monospace';
        const markerSpacing = 240;
        const markerOffset = worldX % markerSpacing;
        for (let mx = -markerSpacing; mx < w + markerSpacing; mx += markerSpacing) {
            const drawX = mx + markerSpacing - markerOffset;
            ctx.fillText('➔ ➔ ➔  LANE 0 (PASSING)', drawX, roadY + laneH * 0.52);
            ctx.fillText('➔ ➔ ➔  LANE 1 (CRUISING)', drawX, roadY + laneH * 1.52);
        }

        // 3. MULTI-VEHICLE LOGIC & OVERTAKING
        const allVehicles = [egoCar, followerCar, ...trafficVehicles];

        for (const v of allVehicles) {
            if (v.laneY === 0) v.laneY = laneCenters[v.targetLane];
        }

        // A. DRIVER (EGO)
        egoCar.speed = egoSpeedKmh;
        egoCar.isBraking = isEgoBraking;

        const isDriverAlert = (currentState === SystemState.ALERT || eyeClosedDuration < 2.0) && !isEgoBraking;

        if (isDriverAlert) {
            let egoLead = null;
            let egoLeadGap = 99999;
            for (const other of allVehicles) {
                if (other === egoCar) continue;
                const sameLane = (other.targetLane === egoCar.targetLane) || (Math.abs(other.laneY - egoCar.laneY) < laneH * 0.45);
                if (sameLane && other.x > egoCar.x) {
                    const gap = other.x - (egoCar.x + egoCar.length);
                    if (gap < egoLeadGap) {
                        egoLeadGap = gap;
                        egoLead = other;
                    }
                }
            }

            if (egoLead && egoLeadGap < 150) {
                const otherLane = 1 - egoCar.targetLane;
                if (isLaneSafe(egoCar, otherLane, allVehicles, laneCenters, laneH)) {
                    egoCar.targetLane = otherLane;
                    egoCar.turnSignal = (otherLane === 0) ? 'left' : 'right';
                }
            } else if (egoCar.targetLane === 0 && egoCar.lane === 0) {
                let clearRight = true;
                for (const other of allVehicles) {
                    if (other === egoCar) continue;
                    if (other.targetLane === 1 && Math.abs(other.x - egoCar.x) < 140) {
                        clearRight = false;
                        break;
                    }
                }
                if (clearRight && isLaneSafe(egoCar, 1, allVehicles, laneCenters, laneH)) {
                    egoCar.targetLane = 1;
                    egoCar.turnSignal = 'right';
                }
            }
        } else {
            egoCar.turnSignal = 'none';
        }

        egoCar.laneY += (laneCenters[egoCar.targetLane] - egoCar.laneY) * 0.08;
        if (Math.abs(laneCenters[egoCar.targetLane] - egoCar.laneY) < 2) {
            egoCar.lane = egoCar.targetLane;
            if (egoCar.lane === egoCar.targetLane) egoCar.turnSignal = 'none';
        }

        // B. FOLLOWER (BRAKES WHEN DRIVER BRAKES)
        let followerLead = null;
        let followerLeadGap = 99999;
        for (const other of allVehicles) {
            if (other === followerCar) continue;
            const sameLane = (other.targetLane === followerCar.targetLane) || (Math.abs(other.laneY - followerCar.laneY) < laneH * 0.45);
            if (sameLane && other.x > followerCar.x) {
                const gap = other.x - (followerCar.x + followerCar.length);
                if (gap < followerLeadGap) {
                    followerLeadGap = gap;
                    followerLead = other;
                }
            }
        }

        if (followerLead) {
            if (followerLead.isBraking || followerLeadGap < 140 || (egoCar.isBraking && followerLead === egoCar)) {
                followerCar.isBraking = true;
                followerCar.speed = Math.max(0, Math.min(followerCar.speed - 2.0, followerLead.speed - 2));

                const otherLane = 1 - followerCar.targetLane;
                if ((followerLead.speed < 45 || followerCar.isBraking) && isLaneSafe(followerCar, otherLane, allVehicles, laneCenters, laneH)) {
                    followerCar.targetLane = otherLane;
                    followerCar.turnSignal = (otherLane === 0) ? 'left' : 'right';
                    followerCar.hasOvertaken = true;
                }
            } else {
                followerCar.isBraking = false;
                if (followerCar.speed < followerCar.desiredSpeed) followerCar.speed += 0.5;
            }

            if (followerLeadGap < 35) {
                followerCar.x = followerLead.x - followerCar.length - 35;
                followerCar.speed = Math.min(followerCar.speed, followerLead.speed);
                followerCar.isBraking = true;
            }
        } else {
            followerCar.isBraking = false;
            if (followerCar.speed < followerCar.desiredSpeed) followerCar.speed += 0.5;
        }

        const follRelDiff = (followerCar.speed - egoSpeedKmh) * 0.07;
        followerCar.x += follRelDiff;

        followerCar.laneY += (laneCenters[followerCar.targetLane] - followerCar.laneY) * 0.08;
        if (Math.abs(laneCenters[followerCar.targetLane] - followerCar.laneY) < 2) {
            followerCar.lane = followerCar.targetLane;
            if (followerCar.lane === followerCar.targetLane) followerCar.turnSignal = 'none';
        }

        if (followerCar.x > w + 220) {
            followerCar.x = egoCar.x - 190;
            followerCar.targetLane = egoCar.targetLane;
            followerCar.hasOvertaken = false;
        }

        // C. TRAFFIC VEHICLES
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
                const wantOvertake = (car.speed > carLead.speed - 3) || carLead.isBraking || (carLeadGap < 95);
                const otherLane = 1 - car.targetLane;

                if (wantOvertake && isLaneSafe(car, otherLane, allVehicles, laneCenters, laneH)) {
                    car.targetLane = otherLane;
                    car.turnSignal = (otherLane === 0) ? 'left' : 'right';
                    car.isBraking = false;
                    if (car.speed < car.desiredSpeed + 8) car.speed += 0.4;
                } else {
                    car.isBraking = true;
                    car.speed = Math.max(0, Math.min(car.speed - 1.5, carLead.speed - 1));
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

            const relDiff = (car.speed - egoSpeedKmh) * 0.07;
            car.x += relDiff;

            car.laneY += (laneCenters[car.targetLane] - car.laneY) * 0.08;
            if (Math.abs(laneCenters[car.targetLane] - car.laneY) < 2) {
                car.lane = car.targetLane;
                if (car.lane === car.targetLane) car.turnSignal = 'none';
            }

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

        // 4. DRAW VEHICLES
        const sortedVehicles = [...allVehicles].sort((a, b) => a.laneY - b.laneY);
        for (const v of sortedVehicles) {
            const isEgo = v.isEgo === true;
            let label = v.label;
            if (v.isFollower) {
                label = v.isBraking ? 'FOLLOWER (BRAKING)' : (v.targetLane === 0 ? 'FOLLOWER (PASSING)' : 'FOLLOWER');
            } else if (v.isEgo) {
                label = v.isBraking ? 'YOU (BRAKING)' : (v.targetLane === 0 ? 'YOU (OVERTAKING)' : 'YOU (DRIVER)');
            }
            drawVehicle(v.x, v.laneY, v.type, v.color, label, v.isBraking, isHazard, v.turnSignal, isEgo);
        }

        // 5. FOLLOWER GAP LINE
        if (followerCar.x < egoCar.x && Math.abs(followerCar.laneY - egoCar.laneY) < 30) {
            const currentGapM = Math.max(2, (egoCar.x - (followerCar.x + followerCar.length)) * 0.18);
            const gapColor = currentGapM < 12 ? '#ef4444' : (currentGapM < 22 ? '#f59e0b' : '#34d399');
            ctx.strokeStyle = gapColor;
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(followerCar.x + followerCar.length, followerCar.laneY + 20);
            ctx.lineTo(egoCar.x, egoCar.laneY + 20);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = gapColor;
            ctx.font = 'bold 10px monospace';
            ctx.fillText(`GAP: ${currentGapM.toFixed(1)}m`, (followerCar.x + followerCar.length + egoCar.x) / 2 - 25, followerCar.laneY + 34);
        }
    }

    function drawVehicle(x, centerY, type, color, label, brakeLights, hazardLights, turnSignal, isEgo) {
        ctx.save();

        if (type === 'bike') {
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

            ctx.fillStyle = 'rgba(254, 240, 138, 0.25)';
            ctx.beginPath();
            ctx.moveTo(x + len, centerY);
            ctx.lineTo(x + len + 40, centerY - 15);
            ctx.lineTo(x + len + 40, centerY + 15);
            ctx.closePath();
            ctx.fill();

            if (brakeLights) {
                ctx.fillStyle = '#ff0000';
                ctx.shadowColor = '#ff0000';
                ctx.shadowBlur = 10;
                ctx.fillRect(x - 2, centerY - 3, 3, 6);
                ctx.shadowBlur = 0;
            }

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
            const len = 125;
            const wid = 40;
            const y = centerY - wid / 2;

            ctx.fillStyle = 'rgba(0,0,0,0.35)';
            drawRoundedRect(ctx, x + 3, y + 4, len, wid, 4);
            ctx.fill();

            ctx.fillStyle = color;
            drawRoundedRect(ctx, x, y, len - 34, wid, 4);
            ctx.fill();
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.stroke();

            ctx.fillStyle = 'rgba(0,0,0,0.15)';
            for (let sx = x + 12; sx < x + len - 40; sx += 14) {
                ctx.fillRect(sx, y + 3, 2, wid - 6);
            }

            ctx.fillStyle = '#374151';
            drawRoundedRect(ctx, x + len - 32, y + 2, 32, wid - 4, 5);
            ctx.fill();

            ctx.fillStyle = '#94a3b8';
            ctx.fillRect(x + len - 12, y + 5, 5, wid - 10);

            ctx.fillStyle = '#0f172a';
            ctx.fillRect(x + 8, y - 2, 12, 3);
            ctx.fillRect(x + 24, y - 2, 12, 3);
            ctx.fillRect(x + len - 20, y - 2, 12, 3);
            ctx.fillRect(x + 8, y + wid - 1, 12, 3);
            ctx.fillRect(x + 24, y + wid - 1, 12, 3);
            ctx.fillRect(x + len - 20, y + wid - 1, 12, 3);

            ctx.fillStyle = '#fef08a';
            ctx.fillRect(x + len - 2, y + 4, 3, 7);
            ctx.fillRect(x + len - 2, y + wid - 11, 3, 7);

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
            const len = type === 'suv' ? 82 : 74;
            const wid = type === 'suv' ? 38 : 34;
            const y = centerY - wid / 2;

            ctx.fillStyle = 'rgba(0,0,0,0.35)';
            drawRoundedRect(ctx, x + 2, y + 3, len, wid, 6);
            ctx.fill();

            ctx.fillStyle = '#0f172a';
            ctx.fillRect(x + 8, y - 2, 12, 3);
            ctx.fillRect(x + len - 20, y - 2, 12, 3);
            ctx.fillRect(x + 8, y + wid - 1, 12, 3);
            ctx.fillRect(x + len - 20, y + wid - 1, 12, 3);

            ctx.fillStyle = color;
            drawRoundedRect(ctx, x, y, len, wid, 6);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = isEgo ? 2 : 1;
            ctx.stroke();

            ctx.fillStyle = '#1e293b';
            drawRoundedRect(ctx, x + 18, y + 3, len - 38, wid - 6, 4);
            ctx.fill();

            ctx.fillStyle = '#94a3b8';
            drawRoundedRect(ctx, x + len - 27, y + 5, 5, wid - 10, 2);
            ctx.fill();

            drawRoundedRect(ctx, x + 20, y + 5, 4, wid - 10, 2);
            ctx.fill();

            ctx.fillStyle = 'rgba(254, 240, 138, 0.2)';
            ctx.beginPath();
            ctx.moveTo(x + len, centerY);
            ctx.lineTo(x + len + 50, centerY - 20);
            ctx.lineTo(x + len + 50, centerY + 20);
            ctx.closePath();
            ctx.fill();

            ctx.fillStyle = '#fef08a';
            ctx.fillRect(x + len - 3, y + 2, 3, 6);
            ctx.fillRect(x + len - 3, y + wid - 8, 3, 6);

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

    // =========================================================================
    // 7. WEBCAM & MEDIAPIPE FACE MESH INITIALIZATION
    // =========================================================================
    const videoElement = document.getElementById('webcamVideo');
    const overlayCanvas = document.getElementById('faceOverlayCanvas');
    const overlayCtx = overlayCanvas.getContext('2d');
    const camStatusOverlay = document.getElementById('camStatusOverlay');
    const camStatusText = document.getElementById('camStatusText');

    let latestLandmarks = null;

    function initFaceMesh() {
        if (!window.FaceMesh) {
            if (camStatusText) camStatusText.innerText = 'MediaPipe library loading...';
            setTimeout(initFaceMesh, 500);
            return;
        }

        const faceMesh = new window.FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceMesh.onResults((results) => {
            if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
                latestLandmarks = results.multiFaceLandmarks[0];
                drawFacialOverlay(latestLandmarks);
            } else {
                latestLandmarks = null;
                overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            }
        });

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            }).then((stream) => {
                videoElement.srcObject = stream;
                videoElement.onloadedmetadata = () => {
                    videoElement.play();
                    overlayCanvas.width = videoElement.videoWidth;
                    overlayCanvas.height = videoElement.videoHeight;
                    if (camStatusOverlay) camStatusOverlay.style.display = 'none';

                    const camera = new window.Camera(videoElement, {
                        onFrame: async () => {
                            if (!simMode.active) {
                                await faceMesh.send({ image: videoElement });
                            }
                        },
                        width: 640,
                        height: 480
                    });
                    camera.start();
                };
            }).catch((err) => {
                console.warn('Webcam permission denied or unavailable:', err);
                if (camStatusOverlay) {
                    document.getElementById('camSpinner').style.display = 'none';
                    camStatusText.innerHTML = 'Webcam unavailable or permission denied.<br>Simulation Mode (Press [D]) is active!';
                    camStatusText.style.color = '#f59e0b';
                }
            });
        }
    }

    function drawFacialOverlay(lms) {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        overlayCtx.fillStyle = 'rgba(0, 255, 128, 0.7)';

        const eyeIndices = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380];
        for (const idx of eyeIndices) {
            const pt = lms[idx];
            overlayCtx.beginPath();
            overlayCtx.arc(pt.x * overlayCanvas.width, pt.y * overlayCanvas.height, 2, 0, Math.PI * 2);
            overlayCtx.fill();
        }

        // Mouth outline
        overlayCtx.fillStyle = 'rgba(56, 189, 248, 0.7)';
        const mouthIndices = [61, 291, 13, 14];
        for (const idx of mouthIndices) {
            const pt = lms[idx];
            overlayCtx.beginPath();
            overlayCtx.arc(pt.x * overlayCanvas.width, pt.y * overlayCanvas.height, 2, 0, Math.PI * 2);
            overlayCtx.fill();
        }
    }

    // =========================================================================
    // 8. INTERACTIVE SHORTCUTS & API CONTROLS
    // =========================================================================
    window.triggerDrowsy = function () {
        initAudio();
        simMode.active = true;
        simMode.startTime = performance.now();
        document.getElementById('simBadge').style.display = 'block';
        document.getElementById('metricSource').innerText = 'SIMULATION (D KEY)';
        document.getElementById('metricSource').style.color = '#f97316';
    };

    window.triggerAwake = function () {
        initAudio();
        simMode.active = false;
        document.getElementById('simBadge').style.display = 'none';
        document.getElementById('metricSource').innerText = 'LIVE WEBCAM AI';
        document.getElementById('metricSource').style.color = '#38bdf8';
        vehiclePhysics.resetDriving();
    };

    window.startDriving = function () {
        initAudio();
        simMode.active = false;
        document.getElementById('simBadge').style.display = 'none';
        document.getElementById('metricSource').innerText = 'LIVE WEBCAM AI';
        document.getElementById('metricSource').style.color = '#38bdf8';
        vehiclePhysics.resetDriving();
        followerCar.hasOvertaken = false;
    };

    window.setStrategy = function (strat) {
        if (['abrupt', 'gradual', 'adaptive'].includes(strat)) {
            activeStrategy = strat;
            document.querySelectorAll('.btn-strat').forEach(b => b.classList.remove('active'));
            if (strat === 'abrupt') document.getElementById('stratAbrupt').classList.add('active');
            if (strat === 'gradual') document.getElementById('stratGradual').classList.add('active');
            if (strat === 'adaptive') document.getElementById('stratAdaptive').classList.add('active');

            const badge = document.getElementById('strategyBadge');
            if (badge) badge.innerText = strat.toUpperCase() + ' BRAKING';

            const desc = document.getElementById('strategyDesc');
            if (desc) {
                if (strat === 'abrupt') {
                    desc.innerHTML = '<b>Abrupt Strategy:</b> Instant maximum friction clamp stop (-7.8 m/s²). Halts in minimal distance but carries high rear collision risk.';
                } else if (strat === 'gradual') {
                    desc.innerHTML = '<b>Gradual Strategy:</b> Jerk-limited ramp deceleration (-2.2 m/s³ cap at -4.5 m/s²). Smooth passenger comfort.';
                } else {
                    desc.innerHTML = '<b>Adaptive Strategy:</b> Computes follower time-to-collision (TTC) and smoothly modulates deceleration to prevent rear-end collisions.';
                }
            }
        }
    };

    window.addEventListener('keydown', (e) => {
        if (e.code === 'KeyD') {
            window.triggerDrowsy();
        } else if (e.code === 'Space') {
            e.preventDefault();
            window.triggerAwake();
        } else if (e.code === 'KeyR') {
            window.startDriving();
        } else if (e.code === 'Digit1') {
            window.setStrategy('abrupt');
        } else if (e.code === 'Digit2') {
            window.setStrategy('gradual');
        } else if (e.code === 'Digit3') {
            window.setStrategy('adaptive');
        }
    });

    // =========================================================================
    // 9. UNIFIED MAIN GAME LOOP (60 FPS)
    // =========================================================================
    let lastFrameTime = performance.now();

    function mainLoop(nowTime) {
        const dt = Math.min(0.05, (nowTime - lastFrameTime) / 1000.0);
        lastFrameTime = nowTime;

        // 1. Feature Extraction
        const rawFeatures = updateDriverFeatures(latestLandmarks, nowTime);

        // 2. Score Fusion & Simulation Mode Handling
        const fusionResult = computeDrowsinessScore(rawFeatures, simMode.active);
        drowsinessScore = fusionResult.score;
        const currentClosedDur = fusionResult.closedDuration;
        const currentIsClosed = fusionResult.isClosed;

        // 3. State Machine Transition
        const state = updateStateMachine(drowsinessScore, currentClosedDur, currentIsClosed, vehiclePhysics.speedMps);

        // 4. Autonomous Braking Trigger (Only if asleep > 5.0s or Intervention)
        const isBrakingActive = currentClosedDur >= 5.0 || (state === SystemState.INTERVENTION && currentIsClosed);

        // 5. Vehicle Longitudinal Physics Integration
        const followerGapM = Math.max(2, (egoCar.x - (followerCar.x + followerCar.length)) * 0.18);
        const followerRelMps = (followerCar.speed - vehiclePhysics.speedMps * 3.6) / 3.6;
        const egoPhysicsState = vehiclePhysics.step(dt, isBrakingActive, followerGapM, followerRelMps);

        // 6. Audio Buzzer Trigger
        if (currentClosedDur >= 2.0 || state === SystemState.WARNING || state === SystemState.INTERVENTION) {
            if (!buzzerInterval) {
                const freq = currentClosedDur >= 5.0 ? 980 : 750;
                const speedMs = currentClosedDur >= 5.0 ? 200 : 320;
                buzzerInterval = setInterval(() => {
                    playAlertTone(freq, 160);
                }, speedMs);
            }
        } else {
            if (buzzerInterval) {
                clearInterval(buzzerInterval);
                buzzerInterval = null;
            }
        }

        // 7. Render 2D Canvas Highway Simulation
        const isHazard = state === SystemState.INTERVENTION || state === SystemState.SAFE_STOP;
        renderSimulation(egoPhysicsState.speedKmh, egoPhysicsState.isBraking, isHazard);

        // 8. Update UI Dashboard Elements
        updateDashboardUI(currentClosedDur, currentIsClosed, rawFeatures, egoPhysicsState.speedKmh);

        requestAnimationFrame(mainLoop);
    }

    function updateDashboardUI(closedSec, isClosed, features, egoSpeedKmh) {
        // Sleep Timer Banner
        const banner = document.getElementById('sleepBanner');
        const timerVal = document.getElementById('topSleepTimer');
        const label = document.getElementById('sleepLabel');
        const icon = document.getElementById('sleepIcon');

        if (timerVal) timerVal.innerText = closedSec.toFixed(1) + ' s';

        if (closedSec <= 0.05) {
            banner.style.background = 'rgba(16, 185, 129, 0.15)';
            banner.style.borderColor = 'rgba(16, 185, 129, 0.3)';
            timerVal.style.background = '#064e3b';
            timerVal.style.color = '#a7f3d0';
            icon.innerText = '🟢';
            label.innerText = 'Driver Awake & Attentive:';
        } else if (closedSec < 2.0) {
            banner.style.background = 'rgba(234, 179, 8, 0.25)';
            banner.style.borderColor = 'rgba(234, 179, 8, 0.4)';
            timerVal.style.background = '#713f12';
            timerVal.style.color = '#fef08a';
            icon.innerText = '🟡';
            label.innerText = 'Eye Closure Detected:';
        } else if (closedSec < 5.0) {
            banner.style.background = 'rgba(249, 115, 22, 0.35)';
            banner.style.borderColor = '#f97316';
            timerVal.style.background = '#7c2d12';
            timerVal.style.color = '#ffedd5';
            icon.innerText = '🟠';
            label.innerText = 'WARNING: SLEEP DETECTED! BUZZER ON';
        } else {
            banner.style.background = 'rgba(239, 68, 68, 0.45)';
            banner.style.borderColor = '#ef4444';
            timerVal.style.background = '#991b1b';
            timerVal.style.color = '#fee2e2';
            icon.innerText = '🔴';
            label.innerText = 'CRITICAL: ASLEEP > 5.0s! AUTONOMOUS BRAKING ACTIVATED';
        }

        // Metrics Grid
        const scoreEl = document.getElementById('metricScore');
        if (scoreEl) scoreEl.innerText = drowsinessScore.toFixed(1) + ' / 100';

        const eyeEl = document.getElementById('metricEye');
        if (eyeEl) eyeEl.innerText = (isClosed ? 'CLOSED' : 'OPEN') + ` (EAR: ${features.ear.toFixed(2)})`;

        const pitchEl = document.getElementById('metricPitch');
        if (pitchEl) pitchEl.innerText = features.pitch.toFixed(1) + '°';

        const perclosEl = document.getElementById('metricPerclos');
        if (perclosEl) perclosEl.innerText = (features.perclos * 100).toFixed(1) + '%';

        // Vehicle Telemetry
        const egoSpeedEl = document.getElementById('egoSpeedVal');
        if (egoSpeedEl) egoSpeedEl.innerText = Math.round(egoSpeedKmh) + ' km/h';

        const follSpeedEl = document.getElementById('follSpeedVal');
        if (follSpeedEl) {
            const follSpeed = Math.round(followerCar.speed);
            if (followerCar.isBraking) {
                follSpeedEl.innerText = follSpeed + ' km/h (BRAKING)';
                follSpeedEl.style.color = '#ef4444';
            } else {
                follSpeedEl.innerText = follSpeed + ' km/h';
                follSpeedEl.style.color = '#f87171';
            }
        }

        const follGapEl = document.getElementById('follGapVal');
        if (follGapEl) {
            if (followerCar.targetLane === 0 && followerCar.x > egoCar.x - 50) {
                follGapEl.innerText = 'OVERTAKING (LANE 0)';
                follGapEl.style.color = '#38bdf8';
            } else if (followerCar.isBraking) {
                const g = Math.max(2, (egoCar.x - (followerCar.x + followerCar.length)) * 0.18);
                follGapEl.innerText = 'BRAKING (' + g.toFixed(1) + ' m)';
                follGapEl.style.color = '#f87171';
            } else {
                const g = Math.max(2, (egoCar.x - (followerCar.x + followerCar.length)) * 0.18);
                follGapEl.innerText = g.toFixed(1) + ' m';
                follGapEl.style.color = '#fbbf24';
            }
        }

        const riskEl = document.getElementById('collisionRiskVal');
        if (riskEl) {
            if (followerCar.isBraking) {
                riskEl.innerText = 'BRAKING BEHIND';
                riskEl.style.color = '#f59e0b';
            } else {
                riskEl.innerText = 'SAFE';
                riskEl.style.color = '#34d399';
            }
        }

        // Driver State Banner & Status Pill
        const stateBanner = document.getElementById('stateDescBanner');
        const driverPill = document.getElementById('driverStatusPill');

        if (currentState === SystemState.ALERT) {
            const msg = egoSpeedKmh < 95 ? 'DRIVER HAS CONTROL (ACCELERATING)' : 'DRIVER HAS CONTROL (NORMAL CRUISING)';
            if (stateBanner) {
                stateBanner.innerText = msg;
                stateBanner.className = 'state-banner state-banner-alert';
            }
            if (driverPill) {
                driverPill.innerText = msg;
                driverPill.className = 'status-pill status-pill-alert';
            }
        } else if (currentState === SystemState.CAUTION) {
            if (stateBanner) {
                stateBanner.innerText = 'CAUTION: ATTENTION REDUCING';
                stateBanner.className = 'state-banner state-banner-caution';
            }
            if (driverPill) {
                driverPill.innerText = 'ATTENTION SLIPPING';
                driverPill.className = 'status-pill status-pill-caution';
            }
        } else if (currentState === SystemState.WARNING) {
            if (stateBanner) {
                stateBanner.innerText = 'WARNING: WAKE UP DRIVER! BUZZER ON';
                stateBanner.className = 'state-banner state-banner-warning';
            }
            if (driverPill) {
                driverPill.innerText = 'WARNING: SLEEP DETECTED';
                driverPill.className = 'status-pill status-pill-warning';
            }
        } else {
            if (stateBanner) {
                stateBanner.innerText = 'CRITICAL: AUTONOMOUS BRAKING ACTIVE (>5s)';
                stateBanner.className = 'state-banner state-banner-intervention';
            }
            if (driverPill) {
                driverPill.innerText = 'AUTONOMOUS BRAKING ENGAGED';
                driverPill.className = 'status-pill status-pill-danger';
            }
        }
    }

    // Start App
    initFaceMesh();
    requestAnimationFrame(mainLoop);

})();

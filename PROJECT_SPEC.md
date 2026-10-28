# Driver Drowsiness Detection and Adaptive Braking Simulation

> **Instructions for the Antigravity agent:** This file is the single source of truth for the project.
> Read it fully before writing any code. Work **one phase at a time** (Section 12). After each phase,
> run the acceptance checks, summarise what you did, and wait for confirmation before starting the next phase.
> Do not skip ahead, do not add dependencies that are not listed, and do not rewrite working modules without being asked.

---

## 1. Project Summary

A Python desktop application, built for a scientific student competition, that:

1. Monitors a driver's face through a laptop webcam and detects drowsiness (eye closure, PERCLOS, yawning, head nodding).
2. Fuses those signals into a drowsiness score and runs an escalating warning system (visual and audio).
3. If the driver does not respond, commands a **simulated** vehicle to brake.
4. Simulates a **following vehicle** and measures rear-end collision risk under different braking strategies.
5. Compares three braking strategies: **abrupt**, **gradual (jerk-limited)** and **adaptive (gap-aware)**, and produces graphs, metrics and an automatic report.

**Research question:** *Under what conditions does gradual braking reduce rear-end collision risk compared with abrupt braking, and can a controller that adapts to the following vehicle's gap and speed outperform both?*

**Important:** Gradual braking must NOT be assumed to be safer in every case, since it increases stopping distance. The simulation must be able to show cases where it is worse.

**Scope:** Software simulation only. No real vehicle, no Arduino, no extra hardware. Runs entirely on a laptop with a webcam, display and speakers. Results apply within the simulation only and do not prove real-world safety.

---

## 2. Technology Stack

| Purpose | Library |
|---|---|
| Language | Python 3.10 to 3.12 (use a virtual environment) |
| Webcam capture and image processing | `opencv-python` |
| Face landmarks | `mediapipe` (FaceMesh / Face Landmarker, 468+ landmarks) |
| Simulation and GUI | `pygame` |
| Numerics | `numpy` |
| Graphs and reports | `matplotlib` |
| Config | `pyyaml` |
| Tests | `pytest` |
| Data logging | built-in `csv`, `json`, `dataclasses` |

Pin versions in `requirements.txt` after confirming they install and run together. If the installed MediaPipe version
does not provide the legacy `mp.solutions.face_mesh`, use the MediaPipe Tasks `FaceLandmarker` API instead.
Choose one, and wrap it in the `FaceTracker` class so the rest of the code never depends on the choice.

---

## 3. Repository Structure

```
drowsiness-braking-sim/
├── README.md
├── PROJECT_SPEC.md              # this file
├── requirements.txt
├── config/
│   └── default.yaml             # all thresholds and parameters
├── src/
│   ├── main.py                  # entry point, integrates everything
│   ├── config.py                # loads and validates YAML into dataclasses
│   ├── vision/
│   │   ├── camera.py            # threaded webcam capture
│   │   ├── face_tracker.py      # MediaPipe wrapper -> landmarks
│   │   ├── features.py          # EAR, MAR, head pose, PERCLOS, blink stats
│   │   ├── calibration.py       # per-driver baseline calibration
│   │   └── video_source.py      # webcam OR pre-recorded video file
│   ├── decision/
│   │   ├── fusion.py            # drowsiness score (0-100) with hysteresis
│   │   ├── state_machine.py     # Alert -> Caution -> Warning -> Intervention -> SafeStop -> Recovery
│   │   └── alerts.py            # visual alert levels + audio buzzer
│   ├── sim/
│   │   ├── vehicle.py           # longitudinal vehicle dynamics
│   │   ├── controllers.py       # abrupt, gradual, adaptive braking controllers
│   │   ├── follower.py          # following-vehicle driver model
│   │   ├── safety_metrics.py    # TTC, RSS distance, required deceleration
│   │   ├── scenarios.py         # preset scenarios
│   │   └── runner.py            # headless simulation runner (for batch experiments)
│   ├── ui/
│   │   ├── app.py               # Pygame main window and layout
│   │   ├── panels.py            # monitor, alert, dashboard, analysis panels
│   │   └── assets/              # buzzer.wav, fonts, sprites (generated if missing)
│   └── analysis/
│       ├── logger.py            # CSV/JSON run logging
│       ├── experiments.py       # Monte Carlo / parameter sweeps
│       ├── plots.py             # Matplotlib figures
│       └── report.py            # auto-generated HTML/PDF report
├── tests/
│   ├── test_features.py
│   ├── test_fusion.py
│   ├── test_state_machine.py
│   ├── test_vehicle.py
│   ├── test_controllers.py
│   └── test_safety_metrics.py
├── data/
│   ├── test_videos/             # pre-recorded clips (open eyes, closed eyes, yawn, nod)
│   └── runs/                    # logged runs (gitignored)
└── results/                     # generated graphs and reports
```

---

## 4. Part A: Driver Monitoring (Python and OpenCV)

### 4.1 Video input (`video_source.py`, `camera.py`)
- Capture frames with OpenCV `cv2.VideoCapture`, in a **separate thread** that always keeps the latest frame, so the Pygame loop never blocks.
- Support two sources, switchable via config or CLI flag: **live webcam** (index 0) and **video file** (for repeatable tests).
- Measure and display capture FPS and processing latency.
- Handle camera-not-found and no-face-detected gracefully (status "NO FACE", no crash, and no false alarm. See the safety rule in 4.7).

### 4.2 Face tracking (`face_tracker.py`)
- Wrap MediaPipe FaceMesh (`refine_landmarks=True`, `max_num_faces=1`).
- Output: an `(N, 2)` NumPy array of pixel-space landmarks, plus a `face_detected` flag and a confidence value.
- Convert RGB/BGR correctly. Do not draw on the frame inside the tracker.

### 4.3 Feature extraction (`features.py`)
Use these MediaPipe FaceMesh indices (verify visually with a debug overlay in Phase 1):

| Feature | Landmark indices |
|---|---|
| Right eye (EAR) | `[33, 160, 158, 133, 153, 144]` |
| Left eye (EAR) | `[362, 385, 387, 263, 373, 380]` |
| Mouth (MAR), inner lips | top `13`, bottom `14`, left `78`, right `308` (add `81/178` and `311/402` pairs for a more robust MAR) |
| Head pose (solvePnP) | nose tip `1`, chin `152`, left eye outer `263`, right eye outer `33`, left mouth corner `291`, right mouth corner `61` |

**Eye Aspect Ratio**
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
Average the two eyes. Eye is "closed" when `EAR < closed_threshold` (calibrated per driver, see 4.4).

**PERCLOS**
Percentage of frames in a sliding window (default 60 s, configurable) in which the eye is at least 80% closed. Standard drowsiness metric.
```
PERCLOS = closed_frames_in_window / total_frames_in_window
```

**Blink statistics**
- Count blinks (closure shorter than ~0.5 s) and measure blink duration.
- Long closures (greater than 0.5 s) count as "microsleep candidates".
- Track blink rate per minute and mean blink duration over the window.

**Yawn detection**
- MAR above `yawn_threshold` for more than `yawn_min_duration` (default 1.5 s) counts as one yawn.
- Track yawns per minute.

**Head pose**
- Use `cv2.solvePnP` with a generic 3D face model and approximate camera intrinsics (focal length is approximately the frame width, with the principal point at the frame centre).
- Convert to Euler angles: **pitch, yaw, roll** in degrees.
- "Head nod / sustained downward head" = pitch below `pitch_down_threshold` (relative to the calibrated neutral pitch) for more than `nod_duration` seconds.
- Also report "looking away" (large yaw) as a **distraction** signal, separate from drowsiness.

**Smoothing:** Apply a small moving average or EMA (3 to 5 frames) to EAR/MAR/pose to reduce jitter. Brief blinks (under 0.4 s) and short head movements must NOT trigger alarms.

### 4.4 Calibration (`calibration.py`)
- At startup run a 10 to 15 s calibration (driver looks at the camera, eyes open, neutral mouth).
- Compute the driver's baseline: mean open-eye EAR, baseline neutral head pitch, baseline MAR.
- Set `closed_threshold = baseline_EAR * ratio` (default ratio 0.75), and set the pitch reference to the neutral pitch.
- If calibration fails (no face), fall back to default thresholds from config and show a warning.

### 4.5 Drowsiness fusion (`fusion.py`)
Combine signals into a **drowsiness score from 0 to 100**:

| Signal | Example weight |
|---|---|
| Continuous eye closure duration (target trigger: more than 3 s) | high |
| PERCLOS over the window | high |
| Head nodding / sustained downward pitch | medium |
| Yawn frequency | low |
| Blink duration trend (slower blinks) | low |

- Weights come from config. The score uses **hysteresis** (separate rise and fall thresholds), so the state doesn't flicker.
- Hard rule: **eyes continuously closed for more than `eye_closure_trigger_s` (default 3.0 s) always raises at least a Warning.**
- Output a dataclass: `DrowsinessReading(score, level, eye_closed_s, perclos, yawn_rate, pitch, flags...)`.
- Also emit a separate `distraction` flag (looking away) that does not use the drowsiness braking path.

### 4.6 Driver response detection
- The driver can cancel an escalation by being alert again (eyes open and head up for `recovery_hold_s` seconds) **or** by pressing a "driver response" key (e.g. `SPACE`) to simulate touching the steering wheel.
- Recovery is only accepted after the hold time, to avoid flapping.

### 4.7 Safety and robustness rules
- **No face detected** does NOT count as "eyes closed" for more than a short grace period. After `no_face_timeout_s` raise a "DRIVER NOT VISIBLE" caution, not a drowsiness intervention.
- Poor lighting and occlusion must be reported as "LOW CONFIDENCE" in the UI.
- Include a **keyboard simulation mode**: pressing a key (e.g. `D`) injects simulated drowsiness, so the demo and tests never require anyone to actually fall asleep.
- Do NOT save webcam frames to disk. Process on-device only, and state this in the README (privacy).

---

## 5. Part B: Decision Logic and Alerts

### 5.1 State machine (`state_machine.py`)
```
ALERT -> CAUTION -> WARNING -> INTERVENTION -> SAFE_STOP -> RECOVERY -> ALERT
```
| State | Entry condition (configurable) | Behaviour |
|---|---|---|
| ALERT | score below caution threshold | Normal driving |
| CAUTION | score above caution threshold, or driver not visible | Soft visual notice, no sound |
| WARNING | eyes closed more than 3 s, or score above warning threshold | Visual alarm and buzzer, response timer starts |
| INTERVENTION | warning persists for `response_interval_s` without driver response | Hazard lights on, brake controller engaged |
| SAFE_STOP | vehicle speed reaches 0 (or target speed) | Vehicle held stopped, simulated emergency-call event logged |
| RECOVERY | driver confirmed alert | Stop braking, hand control back smoothly, return to ALERT |

- Transitions must be timestamped and written to the event log.
- Escalation must be **gradual**: the buzzer volume or frequency and the visual intensity grow with time spent in WARNING.
- Every transition is covered by unit tests.

### 5.2 Alerts (`alerts.py`)
- Visual: coloured banner (green/yellow/orange/red), flashing border, large text of the current state.
- Audio: generate a buzzer tone with NumPy and play it with `pygame.mixer` (no external audio file needed, but allow `buzzer.wav` override). Escalating pattern (e.g. slow beeps, then fast beeps, then continuous).
- Show a buzzer status indicator in the UI.

---

## 6. Part C: Vehicle Simulation and Braking

All vehicles are modelled **longitudinally** (1D along the road). Use a fixed time step (default `dt = 0.01 s`, with the simulation clock decoupled from the render FPS).

### 6.1 Vehicle model (`vehicle.py`)
State: position `x`, speed `v`, acceleration `a`. Per step:
```
a = clip(a_cmd, -a_brake_max(friction), a_accel_max)
v = max(0, v + a * dt)
x = x + v * dt
```
- `a_brake_max = mu * g` (default `mu` is 0.8 for dry, 0.5 for wet, 0.2 for icy, with `g = 9.81`), capped by a vehicle-specific limit.
- Fields: `brake_lights_on`, `hazards_on`, `brake_intensity` (0 to 1).
- Speeds are configured in km/h in the UI and converted to m/s internally.

### 6.2 Braking controllers (`controllers.py`)
Each controller returns a desired acceleration `a_cmd` (negative while braking) given the state:

1. **AbruptController:** step command to maximum braking (`-a_max`) at intervention time.
2. **GradualController (jerk-limited):** deceleration ramps up linearly with jerk limit `j` (m/s³), up to a comfortable cap, then holds it.
   ```
   a_cmd(t) = -min(a_cap, j * (t - t_brake_start))
   ```
3. **AdaptiveController (main novelty):** each step reads the following vehicle's gap, speed and TTC and picks the deceleration:
   - If the follower is far and slow, brake gently (comfort-focused, short ramp).
   - If TTC or gap drops below safety margins, increase deceleration as far as allowed, or hold steady rather than brake harder, whichever minimises predicted collision risk (the rule must be documented in code comments and in the report).
   - Must never exceed `a_brake_max`, and must still bring the vehicle to a stop.
   - Include a simple lookahead: predict the follower's trajectory for the next few seconds under the candidate decelerations and choose the one that keeps the minimum gap above a safety margin (a simple search over a few candidate deceleration levels is enough).
- All controllers share one interface: `step(ego_state, follower_state, dt) -> a_cmd`.

### 6.2.1 Early warning to the follower (V2V option, toggle in config)
When enabled, the follower is alerted at the moment of WARNING or INTERVENTION (hazards and brake lights light early). Its reaction delay is then reduced or starts earlier. Compare results with and without it.

### 6.3 Following vehicle (`follower.py`)
- Initial speed and gap are configurable.
- Driver model with a **reaction delay** (default 1.0 s, adjustable 0.5 to 2.0 s): the follower starts braking only after it perceives the lead's brake lights or deceleration plus the delay.
- Follower braking: moderate by default (a configurable fraction of `a_brake_max`) and may escalate to maximum when TTC is critical.
- Follower types (selectable): `attentive`, `distracted` (longer delay), `acc_follower` (a car-following controller such as an Intelligent Driver Model (IDM)).

### 6.4 Safety metrics (`safety_metrics.py`)
Implement and unit-test:
- **Gap:** `gap = x_lead - x_follow - vehicle_length`
- **Relative speed:** `v_rel = v_follow - v_lead` (positive means closing)
- **TTC:** `TTC = gap / v_rel` if `v_rel > 0`, otherwise infinity
- **Required deceleration** for the follower to avoid collision
- **RSS minimum safe distance** (Responsibility-Sensitive Safety), used as a published benchmark for "dangerously close":
  ```
  d_min = max(0, v_r*rho + 0.5*a_accel*rho^2 + (v_r + rho*a_accel)^2 / (2*b_min) - v_f^2 / (2*b_max))
  ```
  where `r` = rear (follower), `f` = front (ego), `rho` is the response time, and `a_accel`, `b_min`, `b_max` are the follower's max acceleration, minimum braking, and the lead's max braking.
- **Minimum gap** over the whole run, **collision flag**, collision time and **impact speed** (relative speed at contact).
- **Comfort metrics:** peak deceleration, peak jerk, total braking time, stopping distance.

### 6.5 Scenarios (`scenarios.py`)
Presets selectable from the UI (all values overridable):
| Name | Ego speed | Initial gap | Road | Follower |
|---|---|---|---|---|
| Highway, close follower | 120 km/h | 15 m | dry | attentive |
| Highway, safe gap | 120 km/h | 60 m | dry | attentive |
| City | 50 km/h | 10 m | dry | attentive |
| Wet road | 90 km/h | 30 m | wet | distracted |
| Icy road | 60 km/h | 40 m | icy | attentive |
| Distracted tailgater | 100 km/h | 12 m | dry | distracted |

### 6.6 Headless runner (`runner.py`)
- Run a full simulation without any UI or webcam, given a scenario, a controller and a drowsiness-trigger time. Returns a results object with time series and metrics. This is used by tests and by the batch experiments.

---

## 7. Part D: User Interface (Pygame)

Single window with these panels:

1. **Driver monitoring panel:** live webcam frame, optional landmark overlay (toggle), eye status (OPEN/CLOSED), EAR value, head pose (pitch/yaw/roll), eye-closure timer, PERCLOS, yawn count, FPS, and a "LOW CONFIDENCE / NO FACE" flag.
2. **Alert panel:** state-machine state, drowsiness score bar (0 to 100), warning level, buzzer status, countdown to intervention.
3. **Vehicle simulation view:** scrolling 2D road, ego vehicle (with brake lights and hazards), following vehicle, and a distance marker between them.
4. **Vehicle dashboard:** speedometer, braking intensity bar, brake-light indicator, following distance, TTC, and the collision-risk warning (green/yellow/red).
5. **Performance analysis panel:** live speed-vs-time plot for both vehicles, minimum gap so far, collision event info, and the controller comparison table.
6. **Controls:** Start, Pause, Reset, initial speed selector, scenario selector, braking-strategy selector (abrupt / gradual / adaptive), follower type, road condition, V2V alert toggle, landmark overlay toggle, source selector (webcam / video file), and the `D` key (simulate drowsiness), `SPACE` (driver response).

UI rules: the interface must stay responsive (at least 20 FPS in the UI with the camera thread running). Use a consistent colour scheme and readable fonts. Never block the main loop with heavy computation.

---

## 8. Part E: Experiments and Analysis

### 8.1 Logging (`logger.py`)
- Log every run to `data/runs/<timestamp>.csv` with time-series columns (time, ego speed, ego accel, brake intensity, follower speed, gap, TTC, state, drowsiness score) plus a `.json` metadata file (scenario, controller, parameters, event log, final metrics).

### 8.2 Batch experiments (`experiments.py`)
- Headless parameter sweeps over: initial speed, initial gap, follower reaction time, road friction, controller type, V2V on/off.
- Use Monte Carlo sampling for random parameter ranges (use seeded RNG so results are reproducible).
- Output: collision probability, mean/min gap, mean impact speed, and stopping distance for each controller.

### 8.3 Plots (`plots.py`)
Generate at least:
- Speed vs time for ego and follower (per controller).
- Gap vs time with the RSS minimum safe distance line.
- Deceleration and jerk vs time.
- **Heatmap of collision probability** (initial gap on one axis, initial speed on the other) for each controller.
- Bar chart comparing controllers (collision rate, min gap, stopping distance).
- Detector evaluation plots (see 8.4).

### 8.4 Detector evaluation
- Record or collect short labelled clips (eyes open, blinks, eyes closed more than 3 s, yawns, nodding, glasses, low light, tilted head).
- Compute: confusion matrix, precision, recall, F1, **detection latency** (time from event start to alarm), and **false alarms per hour** on normal-behaviour footage.
- Ablation: eyes only vs head pose only vs fused. Show that fusion helps (or honestly report if it does not).
- If a public dataset is used (NTHU-DDD, YawDD, UTA-RLDD), check and respect its licence. Do not redistribute it.

### 8.5 Report (`report.py`)
- Auto-generate `results/report.html` (or PDF) containing: scenario, parameters, graphs, metrics table, event log, detector evaluation, and a limitations section.

---

## 9. Configuration (`config/default.yaml`)

All tunable values live here. Example skeleton:
```yaml
camera:
  source: webcam          # webcam | video
  index: 0
  video_path: data/test_videos/sample.mp4
  width: 640
  height: 480

detection:
  ear_closed_ratio: 0.75
  ear_default_threshold: 0.21
  eye_closure_trigger_s: 3.0
  blink_max_s: 0.5
  perclos_window_s: 60
  mar_yawn_threshold: 0.6
  yawn_min_duration_s: 1.5
  pitch_down_deg: 20
  nod_duration_s: 2.0
  no_face_timeout_s: 2.0
  smoothing_frames: 5

fusion:
  weights: {eye_closure: 0.4, perclos: 0.3, head_nod: 0.2, yawn: 0.1}
  caution_score: 35
  warning_score: 60
  hysteresis: 8

alerts:
  response_interval_s: 5.0
  recovery_hold_s: 2.0

vehicle:
  initial_speed_kmh: 100
  length_m: 4.5
  accel_max: 3.0
  friction: dry           # dry | wet | icy
  g: 9.81

braking:
  strategy: adaptive      # abrupt | gradual | adaptive
  gradual_jerk: 2.0       # m/s^3
  gradual_cap: 4.0        # m/s^2
  v2v_alert: true

follower:
  type: attentive         # attentive | distracted | acc
  initial_gap_m: 30
  reaction_time_s: 1.0
  brake_fraction: 0.6

sim:
  dt: 0.01
  seed: 42
```
The code must validate the config, give clear errors for missing or invalid values, and fall back to defaults.

---

## 10. Testing Requirements

- **Unit tests (pytest):**
  - EAR/MAR/PERCLOS computed on synthetic landmark arrays with known results.
  - Fusion: a 3 s closure always yields at least WARNING. A 0.2 s blink never does.
  - State machine: every transition path, including recovery and no-face handling.
  - Vehicle: stopping distance matches `v^2 / (2a)` within tolerance for constant deceleration. Jerk-limited ramp matches analytical values.
  - Safety metrics: TTC, gap, RSS against hand-calculated values.
  - Controllers: never exceed `a_brake_max` and always reach a stop.
- **Integration test:** a headless run for each scenario completes without errors and produces metrics.
- **Reproducibility:** the same seed gives identical results.
- **Manual checklist** (Phase 9): webcam works, blinks do not trigger, a 3 s closure does, covering the camera leads to "no face" (not intervention), the `D` key works, and pause/reset work.

---

## 11. Coding Standards for the Agent

- Python with type hints and docstrings on all public functions and classes.
- Small, single-purpose modules, with no circular imports. Vision, decision, sim and UI communicate through plain dataclasses.
- No magic numbers. Every threshold lives in the config.
- Keep the simulation core free of Pygame and OpenCV imports so it can run headless.
- Handle errors gracefully (camera missing, no face, bad config). Never crash the UI.
- Log events with the `logging` module.
- Keep a short `README.md` with setup instructions (venv, `pip install -r requirements.txt`, how to run, keyboard shortcuts) and update it each phase.
- Make small commits with clear messages if git is used.
- Do not store or upload webcam images.

---

## 12. Implementation Roadmap (work phase by phase)

For each phase: implement, write or update tests, run them, run the app if relevant, summarise results, then stop and wait for confirmation.

### Phase 0: Project setup
- Create the repo structure from Section 3, `requirements.txt`, virtual environment, `config/default.yaml`, `config.py`, logging setup, and a `README.md` skeleton.
- **Done when:** `pip install -r requirements.txt` works and `python -m src.main --help` runs.

### Phase 1: Webcam and face landmarks
- Implement `video_source.py`, `camera.py` (threaded) and `face_tracker.py`.
- Create a small debug window (OpenCV or Pygame) that shows the live feed, landmarks, and FPS.
- **Done when:** the webcam feed shows landmarks at 20+ FPS, a video file also works, and a missing camera is handled cleanly.

### Phase 2: Eye closure detection
- Implement EAR, smoothing, calibration, the closure timer and the blink counter in `features.py` and `calibration.py`.
- Show eye state, EAR and closure time on the debug view.
- **Done when:** normal blinks are ignored, and a deliberate closure of more than 3 s raises a "closure exceeded" flag. Tests pass.

### Phase 3: PERCLOS, yawn, head pose and fusion
- Add PERCLOS, MAR/yawn detection, `solvePnP` head pose (pitch/yaw/roll), nod detection, and the fusion score with hysteresis (`fusion.py`).
- **Done when:** the score rises with sustained closure or nodding, falls with recovery, and does not flicker. The distraction flag works separately. Tests pass.

### Phase 4: State machine, alerts, buzzer
- Implement `state_machine.py` and `alerts.py` (visual banner and generated buzzer with escalation). Add the `D` (simulate drowsiness) and `SPACE` (driver response) keys.
- **Done when:** every state transition works and is logged, the buzzer escalates, and recovery works. Tests pass.

### Phase 5: Vehicle simulation and dashboard
- Implement `vehicle.py` and the Pygame road view, speedometer, brake lights, hazards and braking-intensity bar.
- **Done when:** a vehicle at a set speed brakes with a manually supplied deceleration and the stopping distance matches theory. Tests pass.

### Phase 6: Braking controllers
- Implement abrupt and gradual (jerk-limited) controllers and connect INTERVENTION to braking.
- **Done when:** a real or simulated drowsiness event leads to progressive braking until stop, then SAFE_STOP. Tests pass.

### Phase 7: Following vehicle and collision risk
- Implement `follower.py` (with reaction delay and three types), `safety_metrics.py` (gap, TTC, RSS, impact speed) and the risk indicator in the UI. Add scenario presets.
- **Done when:** minimum gap and collisions are recorded, risk levels display correctly, and scenarios load. Tests pass.

### Phase 8: Adaptive controller and V2V alert
- Implement `AdaptiveController` with the lookahead rule, and the V2V early-alert option.
- **Done when:** in at least one scenario the adaptive controller outperforms both abrupt and gradual braking, and a scenario exists where gradual braking is worse than abrupt (this must be reported honestly). Tests pass.

### Phase 9: Logging, experiments and analysis
- Implement `logger.py`, `runner.py` (headless), `experiments.py` (Monte Carlo sweeps), `plots.py` (all figures in 8.3) and `report.py`.
- **Done when:** one command generates all graphs and `results/report.html` from a seeded batch run.

### Phase 10: Detector evaluation
- Build the evaluation script for Section 8.4 (confusion matrix, precision, recall, F1, latency, false alarms per hour, ablation) using recorded clips.
- **Done when:** the metrics table and plots are produced for at least the "eyes open", "blink", "closed more than 3 s", "yawn" and "nod" cases.

### Phase 11: Integration, polish, demo
- Integrate everything into the full Pygame app with all panels and controls (Section 7). Add the pause/reset behaviour, error handling, and a demo mode (auto-runs a scripted scenario).
- Run the manual checklist from Section 10. Fix performance issues. Finish the README with screenshots, the keyboard shortcuts, and the limitations statement.
- **Done when:** the full demo runs smoothly end to end on a laptop with no crashes.

### Phase 12 (optional stretch)
- Iris-based gaze tracking.
- A small ML classifier (logistic regression or random forest) trained on landmark features, compared with the rule-based fusion.
- Multi-lane road with a shoulder pull-over manoeuvre.
- Replay mode with a timeline scrubber.

---

## 13. Definition of Done (whole project)

- Live webcam drowsiness detection with calibration, PERCLOS, yawn and head-pose signals fused into one score.
- A three-second closure triggers a warning, and ordinary blinks never do.
- Escalating visual and audio alerts with a configurable response window.
- Gradual, abrupt and adaptive braking on a simulated vehicle.
- A following vehicle with reaction delay, plus gap, TTC, RSS and collision metrics.
- Batch experiments, comparison graphs, the collision-probability heatmap, and an auto-generated report.
- Detector evaluation with confusion matrix, latency and false-alarm rate.
- Unit tests pass, and the app runs on a standard laptop with only a webcam.
- Limitations are clearly stated: simulation only, simplified models, camera-based detection can produce false positives and negatives, and results do not establish real-world safety.

---

## 14. Limitations (include verbatim in README and report)

This is a software simulation for academic demonstration. It does not control a real vehicle or its brakes. Camera-based
drowsiness detection can produce false positives and false negatives, particularly under poor lighting, facial occlusion
or unreliable head tracking. The braking controllers and the following-vehicle model are simplified mathematical
representations. Their results show behaviour within the simulation and do not establish real-world vehicle safety or
guarantee collision prevention. Gradual braking is not assumed to be safer in all conditions, since it can increase
stopping distance.

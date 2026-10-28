# Driver Drowsiness Detection and Adaptive Braking Simulation

A comprehensive scientific desktop simulation combining computer-vision driver monitoring with longitudinal vehicle dynamics, following-vehicle reaction modeling, and adaptive rear-end collision mitigation.

---

## Key Features

1. **Computer Vision & Driver Monitoring (`src/vision/`)**
   - Non-blocking threaded video capture (`camera.py`, `video_source.py`) supporting both live hardware webcam (index 0) and pre-recorded video files.
   - Robust facial landmark tracking (`face_tracker.py`) wrapping MediaPipe FaceLandmarker with XNNPACK CPU acceleration (478 landmarks).
   - High-fidelity biometric metrics (`features.py`):
     - **Eye Aspect Ratio (EAR)**: Dual-eye 6-point formulation with moving average smoothing.
     - **PERCLOS**: Sliding window (60s) calculation of eye closure percentage.
     - **Blink Dynamics**: Distinguishes normal brief blinks (<0.5s) from microsleep events.
     - **Mouth Aspect Ratio (MAR)**: Multi-point lip contour tracking for yawn frequency.
     - **Head Pose Estimation**: 3D-to-2D perspective-n-point (`cv2.solvePnP`) decomposing Euler pitch, yaw, and roll to detect head nodding vs. distraction.
   - Per-driver baseline calibration (`calibration.py`).

2. **Decision Logic & Safety State Machine (`src/decision/`)**
   - Multi-signal weighted fusion (`fusion.py`) with rise/fall hysteresis to prevent flickering.
   - Hard trigger rule: Continuous eye closure $\ge 3.0$ seconds automatically triggers Warning.
   - Safety escalation state machine (`state_machine.py`):
     `ALERT` $\to$ `CAUTION` $\to$ `WARNING` $\to$ `INTERVENTION` $\to$ `SAFE_STOP` $\to$ `RECOVERY` $\to$ `ALERT`
   - Real-time acoustic alert generator (`alerts.py`) utilizing synthesized audio tones via NumPy and Pygame mixer with escalating frequency and urgency.

3. **Longitudinal Vehicle Physics & Braking Controllers (`src/sim/`)**
   - Discrete fixed time-step ($\Delta t = 0.01\text{s}$) longitudinal physics model (`vehicle.py`) constrained by road surface friction ($\mu_{\text{dry}}=0.8, \mu_{\text{wet}}=0.5, \mu_{\text{icy}}=0.2$).
   - **Abrupt Controller**: Immediate step deceleration to physical friction limit.
   - **Gradual Controller**: Jerk-limited linear deceleration ramp up to a comfort ceiling.
   - **Adaptive Controller**: Predictive lookahead trajectory evaluator assessing candidate decelerations to minimize rear-end collision risk while safely stopping the vehicle.
   - **Follower Driver Model (`follower.py`)**: Models attentive, distracted, and Intelligent Driver Model (ACC / IDM) following vehicles with perception reaction delay and V2V early warning integration.
   - **Safety Metrics (`safety_metrics.py`)**: Computes gap, closing speed ($v_{\text{rel}}$), Time-to-Collision (TTC), RSS (Responsibility-Sensitive Safety) safe distance, and impact speed.

4. **Pygame Interactive Dashboard (`src/ui/`)**
   - 5-panel layout: Live Driver Monitoring, Safety Escalation Banner, Vehicle Telemetry Dashboard, 2D Animated Scrolling Road View, and Performance Strategy Comparison.

5. **Batch Experiments & Reporting (`src/analysis/`)**
   - Headless simulation runner (`runner.py`) for automated benchmarks.
   - Monte Carlo parameter sweeps (`experiments.py`) over speeds, gaps, follower reaction times, and road friction.
   - Matplotlib publication-ready figure generator (`plots.py`): speed profiles, dynamics, collision heatmaps, and controller comparison charts.
   - Detector evaluation suite (`evaluate_detector.py`): Confusion matrix, Precision, Recall, F1 score, detection latency, and modality ablation.
   - Auto-generated scientific HTML report (`report.py`) stored at `results/report.html`.

---

## Setup & Installation

### Prerequisites
- Python 3.10 to 3.13
- Laptop with webcam, display, and speakers

### 1. Virtual Environment & Dependencies
```bash
# Clone or open the project folder
cd sleep_detector

# Create virtual environment (Python 3.13)
py -3.13 -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## How to Run

### Interactive GUI Mode
```bash
# Launch interactive application (webcam default)
python -m src.main

# Launch with pre-recorded test video
python -m src.main --source video --video data/test_videos/sample.mp4

# Launch directly with a specific braking controller (abrupt, gradual, adaptive)
python -m src.main --strategy gradual
```

### Batch Experiments & Report Generation
Run Monte Carlo sweeps, detector evaluation, and generate HTML reports and graphs:
```bash
python -m src.main --experiments
```
Output artifacts are saved in [`results/report.html`](file:///e:/sleep_detector/results/report.html) and [`results/*.png`](file:///e:/sleep_detector/results/).

### Headless Verification Mode
```bash
python -m src.main --headless
```

---

## Interactive Keyboard Controls
| Key | Action |
|---|---|
| `D` | **Toggle Simulated Drowsiness** (tests 3s escalation & braking without sleeping) |
| `SPACE` | **Driver Response / Touch Wheel** (cancels warning / restores alert state) |
| `1` | Switch to **Abrupt Braking Controller** |
| `2` | Switch to **Gradual (Jerk-Limited) Controller** |
| `3` | Switch to **Adaptive (Gap-Aware Lookahead) Controller** |
| `L` | Toggle Facial Landmark Overlay |
| `P` | Pause / Resume simulation |
| `R` | Reset vehicles & simulation to scenario start |
| `ESC` / `Q` | Exit application |

---

## Testing
Execute full automated pytest test suite:
```bash
pytest
```

---

## Privacy Notice
Webcam frames are processed in-memory in real time. **No video frames or facial images are ever recorded, saved to disk, or transmitted over the network.**

---

## Limitations

> This is a software simulation for academic demonstration. It does not control a real vehicle or its brakes. Camera-based drowsiness detection can produce false positives and false negatives, particularly under poor lighting, facial occlusion or unreliable head tracking. The braking controllers and the following-vehicle model are simplified mathematical representations. Their results show behaviour within the simulation and do not establish real-world vehicle safety or guarantee collision prevention. Gradual braking is not assumed to be safer in all conditions, since it can increase stopping distance.

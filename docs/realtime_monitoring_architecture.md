# Real-Time Planarian Monitoring System — Architecture Document

**Created:** 2026-03-02
**Status:** Planning / Pre-implementation
**Hardware:** Raspberry Pi (already available), USB camera, custom light rig

---

## 1. Vision

A long-term (9+ month), real-time planarian monitoring system that:
- Continuously tracks each worm's position, orientation, speed, and behavior
- Extracts and stores behavioral metrics — NOT petabytes of raw video
- Controls hardware (lights on/off) in response to worm behavior (e.g., crossing a light beam)
- Runs 24/7 on a Raspberry Pi at the edge, with minimal human intervention

---

## 2. The Storage Problem (Why Extract, Don't Store)

### Raw video math
| Parameter           | Value             |
|---------------------|-------------------|
| Resolution          | 1920×1080         |
| Frame rate          | 30 fps            |
| H.264 bitrate      | ~5 Mbps           |
| Per hour            | ~2.25 GB          |
| Per day             | ~54 GB            |
| Per month           | ~1.6 TB           |
| Per 9 months        | ~14.6 TB          |
| Per worm × 9 months | ~14.6 TB          |

Even with aggressive compression, storing continuous video is impractical. The solution: **extract metrics in real-time and discard the frames**.

### Data reduction
| What we store                | Size per frame | Size per day | 9-month total |
|------------------------------|---------------|--------------|---------------|
| Raw video frame (H.264)     | ~21 KB        | ~54 GB       | 14.6 TB       |
| Extracted metrics row        | ~200 bytes    | ~500 MB      | ~135 GB       |
| **Reduction factor**         |               |              | **~100×**     |

With frame-skipping (every 3rd frame at 10 fps effective), we reduce further to ~170 MB/day of metrics, or **~45 GB for 9 months** — easily fits on a single SD card or USB drive.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Raspberry Pi                      │
│                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  Camera  │───▶│  Capture     │───▶│  Tracking │  │
│  │  (USB)   │    │  Loop        │    │  Engine   │  │
│  └──────────┘    │  (10 fps)    │    │           │  │
│                  └──────────────┘    └─────┬─────┘  │
│                                            │        │
│                    ┌───────────────────────┤        │
│                    ▼                       ▼        │
│  ┌──────────────────────┐   ┌────────────────────┐  │
│  │  Metrics DB          │   │  Hardware Control  │  │
│  │  (SQLite)            │   │  (GPIO → lights)   │  │
│  │  - position          │   │  - beam triggers   │  │
│  │  - speed             │   │  - scheduled on/off│  │
│  │  - orientation       │   │  - event responses │  │
│  │  - area / shape      │   └────────────────────┘  │
│  │  - behavior flags    │                           │
│  └──────────┬───────────┘                           │
│             │                                       │
│             ▼                                       │
│  ┌──────────────────────┐   ┌────────────────────┐  │
│  │  Rolling Video Buffer│   │  Event Clip Saver  │  │
│  │  (last 24–48 hrs)    │   │  (interesting      │  │
│  │  auto-overwrite      │   │   moments only)    │  │
│  └──────────────────────┘   └────────────────────┘  │
│                                                     │
│             ▼ (nightly sync)                        │
│  ┌──────────────────────┐                           │
│  │  Network Sync        │                           │
│  │  → NAS / lab server  │                           │
│  │  → cloud backup      │                           │
│  └──────────────────────┘                           │
└─────────────────────────────────────────────────────┘
```

---

## 4. Component Details

### 4.1 Camera Capture Loop

**Goal:** Grab frames at a consistent rate, apply the grid-baseline subtraction, feed to tracker.

```python
# Pseudocode — real-time capture loop
import cv2
import time

cap = cv2.VideoCapture(0)  # USB camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

TARGET_FPS = 10  # Effective tracking rate
frame_interval = 1.0 / TARGET_FPS

while running:
    t0 = time.monotonic()
    ok, frame = cap.read()
    if not ok:
        handle_camera_error()
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Track
    result = detect_worm(gray, grid_baseline, dish_mask, last_centroid, ...)

    # Store metrics
    db.insert_metric(timestamp, result)

    # Hardware control check
    hardware_controller.check_triggers(result)

    # Optional: write to rolling video buffer
    if rolling_buffer_enabled:
        video_writer.write(frame)

    # Rate limiting
    elapsed = time.monotonic() - t0
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)
```

**Key design choices:**
- **10 fps effective rate** — sufficient for planarian speed (~0.5–2 mm/s). No need for 30 fps.
- **Direct camera access** via `cv2.VideoCapture(0)` — no intermediate video files.
- **Rate-limited loop** — ensures consistent timing without overloading the Pi.

### 4.2 Tracking Engine (Adapted from open_dish_tracker.py)

The existing tracker needs these adaptations for real-time use:

| Offline (current)                        | Real-time (target)                       |
|------------------------------------------|------------------------------------------|
| Reads from .mkv video files              | Reads from live camera feed              |
| `build_grid_baseline()` from file frames | Builds baseline during startup/calibration |
| Processes all frames then writes CSV     | Writes each metric row immediately        |
| Overlay written to video file            | Optional live preview window              |
| Manual seed via GUI popup                | Auto-seed after startup stabilization     |

**Startup sequence:**
1. Power on → camera warm-up (5 seconds)
2. Capture 20 "empty dish" frames → build grid baseline
3. Wait for worm detection (auto-seed) or manual confirmation
4. Enter continuous tracking loop

**Adaptive baseline for long-term drift:**
Over 9 months, lighting conditions, dish cleanliness, camera position, and grid paper will change. The baseline must adapt.

```python
class AdaptiveBaseline:
    """Slowly adapts the grid baseline to account for long-term drift.

    Strategy: When the worm is detected with high confidence, the pixels
    OUTSIDE the worm contour are "safe" background. Blend those pixels
    into the baseline with a very slow learning rate.
    """

    def __init__(self, initial_baseline, learning_rate=0.0001):
        self.baseline = initial_baseline.copy()
        self.learning_rate = learning_rate

    def update(self, gray_frame, worm_mask, confidence):
        """Update baseline using non-worm pixels when confidence is high."""
        if confidence < 0.8:
            return  # Don't update when uncertain

        # Only update pixels NOT covered by the worm
        safe_mask = ~worm_mask
        alpha = self.learning_rate
        self.baseline[safe_mask] = (
            (1 - alpha) * self.baseline[safe_mask] +
            alpha * gray_frame[safe_mask]
        )
```

**Learning rate reasoning:**
- At `alpha = 0.0001` and 10 fps, the effective half-life is ~19 minutes
- This is fast enough to track lighting changes (sunrise/sunset) but slow enough to not absorb the worm into the baseline
- Can be tuned: slower (0.00001) for very stable environments, faster (0.001) if lighting changes rapidly

### 4.3 Metrics Database (SQLite)

SQLite is ideal for edge deployment: zero-config, single file, works well on Pi, handles concurrent reads.

```sql
-- Core tracking table — one row per frame
CREATE TABLE tracking (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,          -- Unix timestamp (ms precision)
    session_id  TEXT NOT NULL,           -- e.g., "worm_A_2026-03-01"
    frame_idx   INTEGER NOT NULL,

    -- Position & movement
    x_px        REAL,                   -- Centroid x (pixels)
    y_px        REAL,                   -- Centroid y (pixels)
    x_mm        REAL,                   -- Centroid x (mm, calibrated)
    y_mm        REAL,                   -- Centroid y (mm, calibrated)
    speed_px_s  REAL,                   -- Instantaneous speed (px/s)
    speed_mm_s  REAL,                   -- Instantaneous speed (mm/s)

    -- Orientation
    body_angle  REAL,                   -- Body axis angle (degrees)
    head_angle  REAL,                   -- Head direction (degrees)

    -- Morphology
    area_px     REAL,                   -- Blob area (pixels)
    contour_len REAL,                   -- Contour perimeter
    aspect_ratio REAL,                  -- Fitted ellipse aspect ratio

    -- Detection quality
    confidence  REAL,                   -- Detection confidence [0–1]
    is_lost     INTEGER DEFAULT 0,      -- 1 if detection failed this frame

    -- Derived behavior (computed in post-processing or real-time)
    is_moving       INTEGER,            -- Speed above threshold
    is_turning      INTEGER,            -- Angular velocity above threshold
    in_light_zone   INTEGER,            -- Currently in light beam area

    UNIQUE(session_id, frame_idx)
);

-- Indices for common queries
CREATE INDEX idx_tracking_session_time ON tracking(session_id, timestamp);
CREATE INDEX idx_tracking_lost ON tracking(session_id, is_lost);

-- Event log — discrete behavioral events
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,
    session_id  TEXT NOT NULL,
    event_type  TEXT NOT NULL,           -- 'light_crossing', 'reversal',
                                        -- 'long_pause', 'head_poke', etc.
    x_px        REAL,
    y_px        REAL,
    metadata    TEXT                     -- JSON blob for event-specific data
);

CREATE INDEX idx_events_session_type ON events(session_id, event_type);

-- Hardware actions log
CREATE TABLE hardware_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,
    action      TEXT NOT NULL,           -- 'light_on', 'light_off',
                                        -- 'stimulus_pulse', etc.
    trigger     TEXT,                    -- What caused the action
    metadata    TEXT                     -- JSON details
);

-- Calibration snapshots — periodic re-calibrations
CREATE TABLE calibrations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL NOT NULL,
    session_id      TEXT NOT NULL,
    dish_center_x   REAL,
    dish_center_y   REAL,
    dish_radius_px  REAL,
    mm_per_px       REAL,
    grid_spacing_px REAL,
    baseline_hash   TEXT                -- Hash of baseline image for tracking drift
);
```

**Storage estimates at 10 fps:**
- ~200 bytes per row × 864,000 rows/day = ~165 MB/day
- 9 months ≈ 270 days × 165 MB = **~44 GB** total
- SQLite handles this comfortably (tested to multi-TB)

### 4.4 Hardware Control (GPIO)

**Goal:** Turn lights on/off in response to worm behavior or on a schedule.

```python
import RPi.GPIO as GPIO

class HardwareController:
    """Controls lights and stimuli via Raspberry Pi GPIO pins."""

    # Pin assignments (BCM numbering)
    LIGHT_PIN = 17          # Main dish illumination
    STIMULUS_PIN = 27       # Light beam for crossing experiments
    INDICATOR_PIN = 22      # Status LED

    def __init__(self, db):
        self.db = db
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.LIGHT_PIN, GPIO.OUT)
        GPIO.setup(self.STIMULUS_PIN, GPIO.OUT)
        GPIO.setup(self.INDICATOR_PIN, GPIO.OUT)

        # Light beam zone definition (pixels)
        self.beam_zone = None  # Set during calibration
        self.beam_active = False

    def define_beam_zone(self, x_center, y_center, width, height):
        """Define the rectangular zone where the light beam hits the dish."""
        self.beam_zone = (
            x_center - width // 2, y_center - height // 2,
            x_center + width // 2, y_center + height // 2
        )

    def check_triggers(self, tracking_result):
        """Called every frame — checks if hardware actions are needed."""
        if tracking_result is None or tracking_result['is_lost']:
            return

        x, y = tracking_result['x_px'], tracking_result['y_px']

        # Light beam crossing detection
        if self.beam_zone:
            in_zone = (self.beam_zone[0] <= x <= self.beam_zone[2] and
                       self.beam_zone[1] <= y <= self.beam_zone[3])

            if in_zone and not self.beam_active:
                # Worm just entered the beam zone
                self.beam_active = True
                self._trigger_light(on=True, reason='beam_entry')
                self.db.log_event('light_crossing', x, y,
                                  {'direction': 'enter'})

            elif not in_zone and self.beam_active:
                # Worm left the beam zone
                self.beam_active = False
                self._trigger_light(on=False, reason='beam_exit')
                self.db.log_event('light_crossing', x, y,
                                  {'direction': 'exit'})

    def _trigger_light(self, on, reason):
        """Activate or deactivate the stimulus light."""
        GPIO.output(self.STIMULUS_PIN, GPIO.HIGH if on else GPIO.LOW)
        self.db.log_hardware('light_on' if on else 'light_off', reason)

    def set_schedule(self, schedule):
        """Set a light/dark cycle schedule.

        schedule: list of (hour_on, hour_off) tuples
        Example: [(6, 18)] = lights on at 6am, off at 6pm
        """
        self.schedule = schedule

    def check_schedule(self, current_time):
        """Check if scheduled light changes are needed."""
        hour = current_time.hour
        for on_hour, off_hour in self.schedule:
            should_be_on = on_hour <= hour < off_hour
            current_state = GPIO.input(self.LIGHT_PIN)
            if should_be_on and not current_state:
                GPIO.output(self.LIGHT_PIN, GPIO.HIGH)
                self.db.log_hardware('light_on', 'schedule')
            elif not should_be_on and current_state:
                GPIO.output(self.LIGHT_PIN, GPIO.LOW)
                self.db.log_hardware('light_off', 'schedule')

    def cleanup(self):
        GPIO.cleanup()
```

### 4.5 Tiered Storage Strategy

```
Tier 1: Metrics DB (SQLite)
├── Keep: FOREVER
├── Size: ~165 MB/day (~44 GB / 9 months)
├── Contains: All tracking data, events, hardware logs
└── Backup: Nightly rsync to NAS/cloud

Tier 2: Rolling Video Buffer
├── Keep: Last 24–48 hours (auto-overwrite oldest)
├── Size: ~54 GB fixed (circular buffer)
├── Contains: Full-resolution continuous video
├── Purpose: Debug tracking issues, replay recent events
└── Implementation: 1-hour segment files, delete oldest when full

Tier 3: Event Clips
├── Keep: FOREVER (or user-curated)
├── Size: ~1–5 GB / month (depends on event frequency)
├── Contains: 10-second clips around interesting events
├── Triggered by: Light crossings, reversals, unusual behavior
└── Backup: Nightly rsync to NAS/cloud

Tier 4: Periodic Snapshots
├── Keep: FOREVER
├── Size: ~50 MB / month
├── Contains: One JPEG every 15 minutes (timestamped)
├── Purpose: Visual audit trail, timelapse generation
└── Backup: Nightly rsync to NAS/cloud
```

### 4.6 Event Clip Saver

When something interesting happens, save a short video clip around that moment.

```python
from collections import deque

class EventClipSaver:
    """Saves short video clips around interesting events.

    Maintains a rolling buffer of recent frames. When triggered,
    writes the buffer + N future frames to a clip file.
    """

    def __init__(self, output_dir, buffer_seconds=5, post_seconds=5,
                 fps=10, resolution=(1920, 1080)):
        self.buffer = deque(maxlen=buffer_seconds * fps)
        self.output_dir = output_dir
        self.post_seconds = post_seconds
        self.fps = fps
        self.resolution = resolution
        self.recording = False
        self.post_frames_remaining = 0
        self.writer = None

    def feed_frame(self, frame):
        """Feed every frame — buffers and optionally records."""
        self.buffer.append(frame.copy())

        if self.recording:
            self.writer.write(frame)
            self.post_frames_remaining -= 1
            if self.post_frames_remaining <= 0:
                self._stop_recording()

    def trigger(self, event_type, timestamp):
        """Start recording an event clip."""
        if self.recording:
            # Extend current recording
            self.post_frames_remaining = self.post_seconds * self.fps
            return

        filename = f"{event_type}_{timestamp:.0f}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            filepath, fourcc, self.fps, self.resolution)

        # Write buffered (pre-event) frames
        for buffered_frame in self.buffer:
            self.writer.write(buffered_frame)

        self.recording = True
        self.post_frames_remaining = self.post_seconds * self.fps

    def _stop_recording(self):
        self.recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
```

---

## 5. Raspberry Pi Specifics

### 5.1 Hardware Requirements

| Component            | Minimum                   | Recommended               |
|----------------------|---------------------------|---------------------------|
| Pi Model             | Raspberry Pi 4 (4GB)      | Raspberry Pi 5 (8GB)      |
| Camera               | USB webcam (1080p)        | Pi Camera Module 3        |
| Storage              | 128 GB microSD            | 256 GB microSD + USB SSD  |
| Power                | Official Pi power supply  | UPS HAT for power outages |
| GPIO accessories     | Relay module for lights   | Relay + LED driver board  |
| Cooling              | Passive heatsink          | Active fan (for 24/7 use) |
| Network              | WiFi                      | Ethernet (more reliable)  |

### 5.2 Performance Budget

At 1080p and 10 fps on a Raspberry Pi 4:

| Operation                    | Time per frame | Notes                     |
|------------------------------|---------------|----------------------------|
| Camera capture               | ~10 ms        | USB camera latency         |
| Grayscale + normalize        | ~3 ms         | OpenCV, numpy              |
| Grid baseline subtraction    | ~5 ms         | Element-wise float ops     |
| Thresholding + contours      | ~4 ms         | OpenCV binary ops          |
| Candidate scoring + selection| ~2 ms         | Small number of contours   |
| Orientation computation      | ~2 ms         | Ellipse fitting            |
| DB write                     | ~1 ms         | SQLite WAL mode            |
| Hardware GPIO check          | ~0.1 ms       | Negligible                 |
| **Total**                    | **~27 ms**    | **Fits in 100 ms budget**  |

This leaves ~73 ms of headroom per frame. If needed, we can:
- Drop to 640×480 for even faster processing
- Process ROI-only instead of full frame (already partially done)
- Use Pi Camera's hardware-accelerated capture

### 5.3 Reliability for 24/7 Operation

**Watchdog:** Use systemd to auto-restart the tracker if it crashes.

```ini
# /etc/systemd/system/worm-tracker.service
[Unit]
Description=Planarian Worm Tracker
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/planarian-tracker
ExecStart=/home/pi/planarian-tracker/venv/bin/python tracker_realtime.py
Restart=always
RestartSec=10
WatchdogSec=60

[Install]
WantedBy=multi-user.target
```

**Graceful recovery:**
- On startup, check DB for last known position → resume tracking from there
- If camera disconnects, retry every 5 seconds with exponential backoff
- If disk fills up, purge oldest rolling video segments first
- Log all errors to a separate log file with rotation

**Health monitoring:**
```python
def health_check(self):
    """Periodic health check — run every 60 seconds."""
    checks = {
        'camera_ok': self.camera.isOpened(),
        'disk_free_gb': shutil.disk_usage('/').free / 1e9,
        'db_size_mb': os.path.getsize(self.db_path) / 1e6,
        'detection_rate_1h': self.get_detection_rate(hours=1),
        'cpu_temp': self.get_cpu_temp(),
        'uptime_hours': (time.time() - self.start_time) / 3600,
    }

    # Alerts
    if checks['disk_free_gb'] < 5:
        self.purge_oldest_video_segments()
    if checks['cpu_temp'] > 80:
        self.log_warning('CPU temperature high')
    if checks['detection_rate_1h'] < 0.5:
        self.log_warning('Low detection rate — check camera/dish')

    self.db.log_health(checks)
    return checks
```

---

## 6. Light Beam Experiment Design

The user's specific goal: turn lights on/off when the worm crosses a light beam.

### 6.1 Physical Setup

```
         ┌──────────────────────┐
         │      Camera          │
         │      (above)         │
         └──────────┬───────────┘
                    │ views down
                    ▼
    ┌───────────────────────────────┐
    │          Petri Dish           │
    │                               │
    │    LED ─ ─ ─ ─ ─ ─ ─▶ Sensor  │  ← Light beam across dish
    │    (GPIO)     │     (optional)│
    │               │               │
    │          worm crosses         │
    │           beam here           │
    │                               │
    └───────────────────────────────┘
```

**Two approaches for beam detection:**

1. **Vision-based (recommended to start):** Define a virtual beam zone in pixel coordinates. When the worm's centroid enters the zone, trigger the GPIO. No extra hardware needed beyond the camera.

2. **Physical beam + photodiode:** Use an actual LED + photodiode pair across the dish. The photodiode triggers an interrupt on a GPIO input pin. More precise but requires additional hardware alignment.

### 6.2 Experimental Protocol

```python
# Example: Classical conditioning with light
class LightBeamExperiment:
    """Turn on overhead light when worm crosses the beam zone.

    Protocol:
    - Beam zone: vertical line at dish center
    - When worm crosses left→right: light ON for 30 seconds
    - When worm crosses right→left: nothing (control direction)
    - Log all crossings with direction, latency, speed
    """

    def __init__(self, hw_controller, db):
        self.hw = hw_controller
        self.db = db
        self.last_side = None  # 'left' or 'right'
        self.beam_x = None     # Set during calibration

    def check(self, x_px, y_px, timestamp):
        current_side = 'left' if x_px < self.beam_x else 'right'

        if self.last_side and current_side != self.last_side:
            direction = f"{self.last_side}_to_{current_side}"
            self.db.log_event('beam_crossing', x_px, y_px, {
                'direction': direction,
                'timestamp': timestamp
            })

            if direction == 'left_to_right':
                self.hw.pulse_light(duration=30.0)  # 30-second light burst

        self.last_side = current_side
```

---

## 7. Network & Multi-Worm Scaling

For monitoring multiple worms simultaneously:

```
┌─────────┐  ┌─────────┐  ┌─────────┐
│  Pi #1  │  │  Pi #2  │  │  Pi #3  │   ← One Pi per dish
│ Worm A  │  │ Worm B  │  │ Worm C  │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┼────────────┘
                  │ WiFi/Ethernet
                  ▼
        ┌─────────────────┐
        │   Lab Server    │
        │   (NAS/PC)      │
        │                 │
        │  - Aggregated DB│
        │  - Dashboard    │
        │  - Backup       │
        │  - Analysis     │
        └─────────────────┘
```

**Sync protocol:**
- Each Pi runs independently — no shared state needed during tracking
- Nightly: rsync metrics DB + event clips to lab server
- Lab server merges per-worm DBs into an aggregated analysis DB
- Dashboard (Grafana or custom web app) queries the aggregated DB

**Why one Pi per dish (not one Pi for N cameras):**
- Isolation: one Pi crashing doesn't affect others
- GPIO: each Pi controls its own dish's lights
- Simplicity: identical setup replicated N times
- Cost: Pi Zero 2W (~$15) is sufficient for single-dish tracking

---

## 8. Migration Path: Offline → Real-Time

### Phase 1: Current (DONE)
- Offline batch processing of recorded .mkv files
- Manual seed via GUI popup
- CSV output + overlay video for validation
- Grid baseline subtraction achieving ~97% detection

### Phase 2: Near-term improvements
- [ ] Fix dish detection (Otsu grouping dish + paper)
- [ ] Auto-seeding (detect worm appearance automatically)
- [ ] Refactor `detect_worm()` into a `WormTracker` class with state
- [ ] Extract the tracking loop from `process_session()` into a reusable `track_frame()` method

### Phase 3: Real-time prototype
- [ ] Create `tracker_realtime.py` — live camera version
- [ ] Implement `AdaptiveBaseline` class
- [ ] Add SQLite database layer
- [ ] Add systemd service for auto-start
- [ ] Test 24-hour continuous run

### Phase 4: Hardware integration
- [ ] GPIO light control module
- [ ] Define beam zone calibration procedure
- [ ] Implement `LightBeamExperiment` protocol
- [ ] Test stimulus-response loop latency (<100 ms target)

### Phase 5: Multi-worm deployment
- [ ] Duplicate Pi setup for N worms
- [ ] Implement nightly sync to lab server
- [ ] Build dashboard for multi-worm monitoring
- [ ] Set up alerting (email/SMS for issues)

### Phase 6: Long-term reliability
- [ ] Adaptive baseline tuning over weeks
- [ ] Automatic re-calibration (dish detection drift)
- [ ] Disk space management automation
- [ ] UPS integration for power outage recovery
- [ ] Data export pipelines for analysis (R, Python, etc.)

---

## 9. Key Design Decisions to Make Later

These don't need to be decided now, but should be considered when implementation begins:

1. **Camera choice:** USB webcam vs. Pi Camera Module. Pi Camera has lower latency and hardware encoding, but USB is more flexible.

2. **Frame rate vs. accuracy tradeoff:** 10 fps is likely sufficient, but some behaviors (head pokes, rapid reversals) might benefit from 30 fps analysis with 10 fps storage.

3. **Single-process vs. multi-process:** Could separate capture, tracking, and storage into different processes connected via queues. More complex but prevents slow DB writes from causing frame drops.

4. **Baseline refresh strategy:** How often to fully rebuild the grid baseline? Options: (a) never — just use adaptive, (b) daily during a scheduled "worm removal" period, (c) weekly manual recalibration.

5. **Light beam implementation:** Vision-based virtual beam vs. physical LED + photodiode pair. Start virtual, upgrade to physical if sub-frame precision is needed.

6. **Data format for long-term analysis:** SQLite is great for collection, but researchers may want Parquet/HDF5 for large-scale analysis in Python/R. Build an export pipeline.

---

## 10. Bill of Materials (Estimated)

| Item                           | Qty | Unit Cost | Total    |
|--------------------------------|-----|-----------|----------|
| Raspberry Pi 4/5 (4GB)         | 1   | $55       | $55      |
| Pi Camera Module 3 or USB cam  | 1   | $25–35    | $35      |
| 256 GB microSD (high endurance)| 1   | $30       | $30      |
| USB SSD 500 GB (rolling video) | 1   | $40       | $40      |
| Relay module (2-channel)       | 1   | $8        | $8       |
| LED + photodiode pair          | 1   | $5        | $5       |
| Pi power supply (official)     | 1   | $12       | $12      |
| UPS HAT (optional)             | 1   | $25       | $25      |
| Case + fan                     | 1   | $15       | $15      |
| Jumper wires, breadboard       | 1   | $10       | $10      |
| **Per-dish total**             |     |           | **~$235**|

For N worms: ~$235 × N (can use Pi Zero 2W at $15 to reduce to ~$180/dish).

---

## Appendix A: Useful Commands

```bash
# Start tracker service
sudo systemctl start worm-tracker
sudo systemctl status worm-tracker

# View live logs
journalctl -u worm-tracker -f

# Check disk usage
df -h /home/pi/data/

# Query recent tracking data
sqlite3 /home/pi/data/tracking.db \
  "SELECT datetime(timestamp, 'unixepoch'), speed_mm_s FROM tracking
   ORDER BY timestamp DESC LIMIT 10;"

# Export day's data to CSV
sqlite3 -header -csv /home/pi/data/tracking.db \
  "SELECT * FROM tracking
   WHERE timestamp > strftime('%s', 'now', '-1 day');" > today.csv

# Sync to lab server
rsync -avz /home/pi/data/ lab-server:/data/worms/worm_A/
```

## Appendix B: Current Tracker Performance Reference

From the offline batch processing (2026-03-02):

| Metric                | Value                    |
|-----------------------|--------------------------|
| Detection rate        | ~96.8% (61 videos)       |
| Detection method      | Grid baseline subtraction|
| Baseline frames       | 20 pre-worm frames       |
| Threshold             | 0.05 (baseline - gray)   |
| Search radius         | 120 px (grows when lost) |
| Max search radius     | 360 px (3× base)         |
| Frame skip            | 3 (every 3rd frame)      |
| Effective fps         | ~10 fps                  |
| Processing speed      | ~25 fps on MacBook       |

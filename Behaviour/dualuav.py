"""
Behaviour summary:
- Drone1 (leader) is manual-driven.
- Drone2 (follower) is API-driven.
- Detection thread captures images from leader camera and runs YOLOv5.
- On person detection above CONF_THRESH, a detection is stored and an alert printed.
- CLI commands:
    go         - Resume normal following
    stop       - Stop following (hover)
    status     - Print positions + detection status
    investigate- Send follower to last detected position (pauses normal following)
    dismiss    - Dismiss last detection
    return     - Stop investigating and resume following (if previously following)
    land       - Land follower and leader (briefly enables API control for leader)
    exit       - Stop script (doesn't auto-land leader)
"""

import time
import math
import threading
import warnings
import csv
import os
from typing import Optional
import numpy as np
import cv2
import torch
import airsim

# ------------ CONFIG -------------
LEADER = "Drone1"
FOLLOWER = "Drone2"
LEADER_CAMERA = "0"
FOLLOWER_CAMERA = "0"
CONF_THRESH = 0.65                     # Confidence threshold for detection reporting
DETECT_INTERVAL = 0.8                  # Seconds between leader detection frames
FORWARD_ESTIMATE = 10.0                # Rough forward projection for world target estimation
INVESTIGATE_ORBIT_RADIUS = 6.0         # Distance from target for the follower investigate waypoint
INVESTIGATE_CONFIRM_SAMPLES = 6
INVESTIGATE_CONFIRM_INTERVAL = 0.6
SAFE_ALT_RELATIVE = -2.0               # Safe relative altitude offset (NED; negative = up)
FOLLOW_DISTANCE = 8.0                  # Default following distance behind leader
LATERAL_OFFSET = 3.0                   # Lateral offset from leader while following
MAX_SPEED = 5.0                        # Maximum translation speed for moveToPosition
UPDATE_RATE = 8                        # Control update loop rate (Hz)
ALERT_SUPPRESS_S = 5.0                 # Suppress repeated alert prints for this many seconds
DISPLAY_WINDOW_NAME = "Leader (L) --- Follower (R)"

# Descent tuning:
DESCENT_STEP_M = 1.0                   # Large per-step descent in metres (positive moves down in NED)
MAX_DESCENT_STEPS = 8                  # Maximum number of descent steps to attempt
MIN_ALT_ABOVE_LEADER = 1.25            # Minimum allowed altitude above leader to avoid collision (NED)
ARRIVAL_TOLERANCE_M = 0.15             # Position tolerance considered 'arrived' (metres)
ARRIVAL_TIMEOUT_S = 12.0               # Timeout for moveToPosition arrival (seconds)

# Pause-and-confirm tuning (when follower detects)
PAUSE_ON_CONFIRM_SAMPLES = 5           # Number of extra samples to collect while hovering
PAUSE_ON_CONFIRM_INTERVAL = 0.5        # Interval between extra confirmation samples (seconds)
PAUSE_ON_CONFIRM_HOLD_S = 3.0          # Hold time after initial confirmation (seconds)

# Logging
LOG_CSV = "investigation_log.csv"      # CSV file to append investigation summaries
LOG_LOCK = threading.Lock()
# ---------------------------------

# Suppress specific torch future warnings that are noisy but harmless
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")


class DualUAV:
    """Main class that controls a follower UAV which assists a manually-piloted leader UAV."""

    def __init__(self):
        # Connect to AirSim and prepare initial state
        print("[init] connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # State flags
        self.is_following = False
        self.investigating = False
        self.disable_follower_loop = False   # When True follower_loop will not issue follow commands
        self.should_exit = False

        # Model and detection state
        self.model = None
        self.model_lock = threading.Lock()   # Lock around model inference to avoid races
        self.last_detection = None           # Dictionary storing the last leader detection
        self.last_detection_time = 0.0
        self.last_alert_print_time = 0.0
        self.last_alert_signature = None

        # Frames and display overlays for the GUI window
        self.leader_frame = None
        self.follower_frame = None
        self.leader_detection_for_display = None
        self.follower_detection_for_display = None
        self.follower_max_conf_during_investigation = 0.0

        # Threading locks
        self.client_lock = threading.Lock()  # Protects access to AirSim client calls
        self.display_lock = threading.Lock() # Protects frame and overlay variables

        # Ensure CSV header exists before first use
        self._ensure_log_header()

    # ---------------- Logging helpers ----------------
    def _ensure_log_header(self):
        """Ensure the CSV log file exists and has a header row."""
        with LOG_LOCK:
            if not os.path.exists(LOG_CSV):
                with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp_utc",
                        "investigation_id",
                        "leader_conf_before",
                        "follower_max_conf",
                        "confirmed",
                        "investigation_duration_s",
                        "descent_steps",
                        "leader_x", "leader_y", "leader_z",
                        "follower_x", "follower_y", "follower_z",
                        "leader_bbox",
                        "follower_bbox",
                        "note"
                    ])

    def log_investigation(self, inv_id, leader_conf, follower_max_conf, confirmed, duration_s,
                         descent_steps, leader_pos, follower_pos, leader_bbox, follower_bbox, note=""):
        """Append one investigation summary row to the CSV file immediately."""
        row = [
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            inv_id,
            f"{leader_conf:.3f}",
            f"{follower_max_conf:.3f}",
            str(bool(confirmed)),
            f"{duration_s:.3f}",
            int(descent_steps),
            f"{leader_pos.x_val:.3f}", f"{leader_pos.y_val:.3f}", f"{leader_pos.z_val:.3f}",
            f"{follower_pos.x_val:.3f}", f"{follower_pos.y_val:.3f}", f"{follower_pos.z_val:.3f}",
            (":".join(map(lambda v: f"{int(v)}", leader_bbox)) if leader_bbox else ""),
            (":".join(map(lambda v: f"{int(v)}", follower_bbox)) if follower_bbox else ""),
            note
        ]
        with LOG_LOCK:
            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        print(f"[log] Investigation logged to {LOG_CSV} (id={inv_id})")

    # ---------------- Utilities ----------------
    def load_model(self):
        """Load a YOLOv5s model from torch.hub for lightweight onboard detection."""
        print("[model] Loading YOLOv5 (yolov5s) via torch.hub (may download weights first run)...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = CONF_THRESH
        print("[model] Loaded. Classes:", self.model.names)

    def quaternion_to_yaw(self, q: airsim.Quaternionr) -> float:
        """Convert an AirSim quaternion to a yaw angle (radians)."""
        w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_state(self, vehicle_name: str):
        """Fetch the MultirotorState for the named vehicle in a thread-safe manner."""
        with self.client_lock:
            return self.client.getMultirotorState(vehicle_name=vehicle_name)

    def sim_get_image_bgr(self, camera_name: str, vehicle_name: str) -> Optional[np.ndarray]:
        """
        Retrieve a Scene image from AirSim and decode it to an OpenCV BGR ndarray.
        Returns None if no image is available.
        """
        with self.client_lock:
            raw = self.client.simGetImage(camera_name, airsim.ImageType.Scene, vehicle_name=vehicle_name)
        if not raw:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    # ---------------- Follower setup & motion ----------------
    def setup_follower(self):
        """Enable API control for the follower, arm it, and take off to a safe altitude."""
        print("[setup] Enabling API control and arming follower:", FOLLOWER)
        with self.client_lock:
            self.client.enableApiControl(True, FOLLOWER)
            self.client.armDisarm(True, FOLLOWER)
            # Wait for takeoff to complete; join() will block until AirSim reports completion
            self.client.takeoffAsync(vehicle_name=FOLLOWER).join()
        time.sleep(0.5)
        # Attempt to ensure manual control of the leader is preserved
        try:
            self.client.enableApiControl(False, LEADER)
        except Exception:
            pass

    def move_follower_to(self, pos: airsim.Vector3r, wait: bool = False, timeout: float = ARRIVAL_TIMEOUT_S):
        """
        Command the follower to move to the given world position.
        If wait is True, block until movement completes (or timeout).
        Returns True on 'success' (or at least that the command was issued), False on error/timeout.
        """
        try:
            with self.client_lock:
                fut = self.client.moveToPositionAsync(pos.x_val, pos.y_val, pos.z_val, MAX_SPEED, vehicle_name=FOLLOWER)
            if not wait or fut is None:
                return True
            try:
                fut.join()
            except Exception:
                # Fallback polling if join raised an exception
                start = time.time()
                while True:
                    st = self.get_state(FOLLOWER)
                    cur = st.kinematics_estimated.position
                    dx = cur.x_val - pos.x_val
                    dy = cur.y_val - pos.y_val
                    dz = cur.z_val - pos.z_val
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist <= ARRIVAL_TOLERANCE_M:
                        return True
                    if time.time() - start > timeout:
                        print(f"[move] Arrival timeout after {timeout}s (dist={dist:.2f}m)")
                        return False
                    time.sleep(0.15)
            # Confirm position after join
            st = self.get_state(FOLLOWER)
            cur = st.kinematics_estimated.position
            dx = cur.x_val - pos.x_val
            dy = cur.y_val - pos.y_val
            dz = cur.z_val - pos.z_val
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= max(ARRIVAL_TOLERANCE_M, 0.05):
                return True
            else:
                print(f"[move] Warning: after join still dist={dist:.2f}m")
                return True
        except Exception as e:
            print("[move] Error:", e)
            return False

    def force_descend_chunk(self, chunk_m: float, speed_mps: float):
        """
        Force a downward movement of approximately chunk_m by using a vertical velocity command.
        NED convention: positive vz moves downwards. Duration is chunk_m / speed_mps.
        """
        if chunk_m <= 0 or speed_mps <= 0:
            return False
        duration = max(0.6, abs(chunk_m) / speed_mps + 0.3)  # Add a small margin
        try:
            with self.client_lock:
                fut = self.client.moveByVelocityAsync(0, 0, float(speed_mps), duration, vehicle_name=FOLLOWER)
                fut.join()
            return True
        except Exception as e:
            print("[force_descend] Error:", e)
            return False

    def hover_follower(self):
        """Command the follower to hold its current altitude and hover in place."""
        try:
            with self.client_lock:
                z = self.client.getMultirotorState(vehicle_name=FOLLOWER).kinematics_estimated.position.z_val
                self.client.moveByVelocityZAsync(0, 0, z, 1.0, vehicle_name=FOLLOWER)
        except Exception:
            pass

    # ---------------- Detector (leader) ----------------
    def detector_loop(self):
        """Thread loop that continuously runs detection on the leader camera to locate candidate targets."""
        print("[detector] Detector thread started")
        while not self.should_exit:
            try:
                img = self.sim_get_image_bgr(LEADER_CAMERA, LEADER)
                if img is None:
                    time.sleep(DETECT_INTERVAL)
                    continue

                # Update display frame for leader
                with self.display_lock:
                    self.leader_frame = img.copy()

                # Run model inference (guarded by the model lock)
                with self.model_lock:
                    results = self.model(img)
                preds = results.xyxy[0].cpu().numpy()

                # Find the best 'person' detection above threshold
                best = None
                for *box, conf, cls in preds:
                    class_id = int(cls)
                    name = self.model.names[class_id].lower()
                    if name in ("person", "people") and conf >= CONF_THRESH:
                        x1, y1, x2, y2 = map(float, box)
                        if best is None or conf > best["conf"]:
                            best = {"bbox": (x1, y1, x2, y2), "conf": float(conf)}

                if best is not None:
                    # Estimate a rough world position in front of the leader for the 'target' (simple forward projection)
                    st = self.get_state(LEADER)
                    leader_pos = st.kinematics_estimated.position
                    leader_orient = st.kinematics_estimated.orientation
                    yaw = self.quaternion_to_yaw(leader_orient)
                    fx = math.cos(yaw); fy = math.sin(yaw)
                    world_x = leader_pos.x_val + fx * FORWARD_ESTIMATE
                    world_y = leader_pos.y_val + fy * FORWARD_ESTIMATE
                    target = airsim.Vector3r(world_x, world_y, leader_pos.z_val)

                    # Build an alert signature and suppress repeating print spam
                    signature = (round(best["conf"], 3), round(world_x, 2), round(world_y, 2))
                    now = time.time()
                    if signature != self.last_alert_signature or (now - self.last_alert_print_time) > ALERT_SUPPRESS_S:
                        self.last_alert_signature = signature
                        self.last_alert_print_time = now
                        print(f"[ALERT] Leader detected person conf={best['conf']:.2f} at approx world x={world_x:.2f},y={world_y:.2f}")
                        print("         Type 'investigate' to send follower to inspect, or 'dismiss' to ignore.")

                    # Store last detection for later investigation
                    self.last_detection = {"class_name": "person", "conf": best["conf"], "bbox": best["bbox"],
                                           "target_pos": target, "confirmed": False}
                    self.last_detection_time = now

                    # Update leader overlay for the display
                    with self.display_lock:
                        self.leader_detection_for_display = {"bbox": best["bbox"], "conf": best["conf"]}
                else:
                    with self.display_lock:
                        self.leader_detection_for_display = None

                time.sleep(DETECT_INTERVAL)
            except Exception as e:
                print("[detector] Error:", repr(e))
                time.sleep(0.5)
        print("[detector] Exiting")

    # -------------- Follower loop ----------------
    def follower_loop(self):
        """
        Background thread that implements the simple following behaviour:
        When is_following is True, follower keeps a fixed offset behind the leader.
        If disable_follower_loop is True, the loop yields and does not issue follow commands.
        """
        print("[follower_loop] Started")
        while not self.should_exit:
            try:
                if self.disable_follower_loop:
                    time.sleep(0.1)
                    continue

                if self.is_following:
                    st = self.get_state(LEADER)
                    leader_pos = st.kinematics_estimated.position
                    leader_orient = st.kinematics_estimated.orientation
                    yaw = self.quaternion_to_yaw(leader_orient)
                    # Compute follow offset in world frame using leader yaw
                    dx = -FOLLOW_DISTANCE * math.cos(yaw) + LATERAL_OFFSET * math.sin(yaw)
                    dy = -FOLLOW_DISTANCE * math.sin(yaw) - LATERAL_OFFSET * math.cos(yaw)
                    dz = leader_pos.z_val - 2.0
                    target = airsim.Vector3r(leader_pos.x_val + dx, leader_pos.y_val + dy, dz)
                    # Issue asynchronous move; do not wait here to keep loop responsive
                    self.move_follower_to(target, wait=False)
                else:
                    # Hover in-place when not following
                    self.hover_follower()
            except Exception as e:
                print("[follower_loop] Error:", repr(e))
            time.sleep(1.0 / UPDATE_RATE)
        print("[follower_loop] Exiting")

    # ------------- Follower camera updater -------------
    def follower_camera_updater(self):
        """Keep sampling the follower camera for the GUI display (no model inference here)."""
        while not self.should_exit:
            try:
                img = self.sim_get_image_bgr(FOLLOWER_CAMERA, FOLLOWER)
                if img is not None:
                    with self.display_lock:
                        self.follower_frame = img.copy()
                time.sleep(0.6)
            except Exception as e:
                print("[camera_updater] Error:", repr(e))
                time.sleep(1.0)

    # ----------------- Investigate behaviour (move -> rotate -> descend) -----------------
    def rotate_follower_to_face_leader(self):
        """Rotate the follower in-place so that its nose faces the leader's position."""
        try:
            fstate = self.get_state(FOLLOWER)
            follower_pos = fstate.kinematics_estimated.position
            lstate = self.get_state(LEADER)
            leader_pos = lstate.kinematics_estimated.position
            vx = leader_pos.x_val - follower_pos.x_val
            vy = leader_pos.y_val - follower_pos.y_val
            yaw_rad = math.atan2(vy, vx)
            yaw_deg = math.degrees(yaw_rad)
            # Use yaw-mode non-rate command to set absolute yaw in degrees
            with self.client_lock:
                self.client.moveByVelocityZAsync(0, 0, follower_pos.z_val, 1.0,
                                                 yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg),
                                                 vehicle_name=FOLLOWER).join()
            time.sleep(0.25)
        except Exception as e:
            print("[rotate] Error:", e)

    def investigate_once(self):
        """
        Move to the opposite-side waypoint relative to the leader's estimated target,
        rotate to face the leader, then descend stepwise trying to confirm the detection.
        On first follower detection, pause and collect extra confirmation samples, then log results.
        """
        if not self.last_detection or not self.last_detection.get("target_pos"):
            print("[investigate] No recent detection to investigate.")
            return

        # Start an investigation run and capture initial metrics for logging
        inv_start_time = time.time()
        inv_id = int(inv_start_time)  # Simple unique id
        leader_conf_before = float(self.last_detection.get("conf", 0.0))
        leader_bbox = tuple(map(int, self.last_detection.get("bbox", (0, 0, 0, 0)))) if self.last_detection.get("bbox") else None

        # Compute the investigate waypoint on the opposite side of the estimated target
        tgt = self.last_detection["target_pos"]
        st = self.get_state(LEADER)
        leader_pos = st.kinematics_estimated.position

        vx = leader_pos.x_val - tgt.x_val
        vy = leader_pos.y_val - tgt.y_val
        mag = math.hypot(vx, vy)
        if mag < 0.001:
            yaw = self.quaternion_to_yaw(st.kinematics_estimated.orientation)
            vx = math.cos(yaw); vy = math.sin(yaw)
            mag = 1.0
        ux = vx / mag; uy = vy / mag

        wx = tgt.x_val - ux * INVESTIGATE_ORBIT_RADIUS
        wy = tgt.y_val - uy * INVESTIGATE_ORBIT_RADIUS
        initial_wz = leader_pos.z_val + SAFE_ALT_RELATIVE
        waypoint = airsim.Vector3r(float(wx), float(wy), float(initial_wz))

        print(f"[investigate] Beginning investigate: waypoint x={wx:.2f}, y={wy:.2f}, z={initial_wz:.2f}")

        # Claim exclusive control: pause follower_loop while investigating
        self.disable_follower_loop = True
        self.investigating = True
        self.follower_max_conf_during_investigation = 0.0
        descent_steps_done = 0
        with self.display_lock:
            self.follower_detection_for_display = None

        # 1) Travel to waypoint and wait for arrival
        arrived = self.move_follower_to(waypoint, wait=True, timeout=ARRIVAL_TIMEOUT_S)
        if not arrived:
            print("[investigate] Warning: follower did not reach waypoint reliably; proceeding anyway.")
        else:
            print("[investigate] Follower reached waypoint")

        time.sleep(0.6)
        self.hover_follower()
        time.sleep(0.3)

        # 2) Rotate to face the leader to improve viewpoint for inspection
        print("[investigate] Rotating to face leader")
        self.rotate_follower_to_face_leader()
        time.sleep(0.25)

        # 3) Descent loop: sample at current altitude, then descend in steps until confirmation or limit
        confirmed = False
        for step in range(MAX_DESCENT_STEPS + 1):
            # Attempt detection at current altitude
            img = self.sim_get_image_bgr(FOLLOWER_CAMERA, FOLLOWER)
            local_best = None
            if img is not None:
                with self.model_lock:
                    res = self.model(img)
                preds = res.xyxy[0].cpu().numpy()
                for *box, conf, cls in preds:
                    name = self.model.names[int(cls)].lower()
                    if name in ("person", "people") and conf >= CONF_THRESH:
                        x1, y1, x2, y2 = map(float, box)
                        if local_best is None or conf > local_best["conf"]:
                            local_best = {"bbox": (x1, y1, x2, y2), "conf": float(conf)}

            if local_best is not None:
                # FIRST confirmation event: update overlays, update max conf, then pause and re-sample
                confirmed = True
                with self.display_lock:
                    self.follower_detection_for_display = {"bbox": local_best["bbox"], "conf": local_best["conf"]}
                if local_best["conf"] > self.follower_max_conf_during_investigation:
                    self.follower_max_conf_during_investigation = local_best["conf"]
                print(f"[investigate] FIRST detection by follower at step {step+1} conf={local_best['conf']:.2f} - Pausing to confirm")

                # Pause and collect additional confirmation samples while hovering
                hold_samples = []
                # Hover to stabilise the view
                self.hover_follower()
                for s in range(PAUSE_ON_CONFIRM_SAMPLES):
                    time.sleep(PAUSE_ON_CONFIRM_INTERVAL)
                    img2 = self.sim_get_image_bgr(FOLLOWER_CAMERA, FOLLOWER)
                    if img2 is None:
                        continue
                    with self.model_lock:
                        res2 = self.model(img2)
                    preds2 = res2.xyxy[0].cpu().numpy()
                    best2 = None
                    for *b2, conf2, cls2 in preds2:
                        name2 = self.model.names[int(cls2)].lower()
                        if name2 in ("person", "people") and conf2 >= CONF_THRESH:
                            x21, y21, x22, y22 = map(float, b2)
                            if best2 is None or conf2 > best2["conf"]:
                                best2 = {"bbox": (x21, y21, x22, y22), "conf": float(conf2)}
                    if best2 is not None:
                        hold_samples.append(best2["conf"])
                        with self.display_lock:
                            self.follower_detection_for_display = {"bbox": best2["bbox"], "conf": best2["conf"]}
                        if best2["conf"] > self.follower_max_conf_during_investigation:
                            self.follower_max_conf_during_investigation = best2["conf"]

                # Optional extra hold time for visual confirmation
                time.sleep(PAUSE_ON_CONFIRM_HOLD_S)

                # Collect follower and leader positions for logging
                fpos_after = self.get_state(FOLLOWER).kinematics_estimated.position
                lpos_after = self.get_state(LEADER).kinematics_estimated.position
                follower_bbox = tuple(map(int, self.follower_detection_for_display["bbox"])) if self.follower_detection_for_display else None

                # Finish: compute metrics and log the investigation summary
                inv_duration = time.time() - inv_start_time
                leader_pos_at_start = leader_pos
                follower_pos_at_end = fpos_after
                self.log_investigation(
                    inv_id,
                    leader_conf_before,
                    self.follower_max_conf_during_investigation,
                    True,
                    inv_duration,
                    descent_steps_done,
                    leader_pos_at_start,
                    follower_pos_at_end,
                    leader_bbox,
                    follower_bbox,
                    note=f"hold_samples={len(hold_samples)} avg_hold_conf={(sum(hold_samples)/len(hold_samples)) if hold_samples else 0:.3f}"
                )

                # Break out and resume follow behaviour (we already hovered)
                break

            # If not confirmed, prepare to descend one step
            if step >= MAX_DESCENT_STEPS:
                print("[investigate] Reached maximum descent rounds without confirmation")
                break

            # Compute next z from current follower position (NED convention)
            fstate = self.get_state(FOLLOWER)
            follower_pos = fstate.kinematics_estimated.position
            leader_pos = self.get_state(LEADER).kinematics_estimated.position
            next_z = follower_pos.z_val + DESCENT_STEP_M
            min_allowed_z = leader_pos.z_val + MIN_ALT_ABOVE_LEADER
            if next_z >= min_allowed_z:
                print("[investigate] Cannot descend further without breaching safety limit")
                break

            # Command descent and wait for it to complete
            descent_target = airsim.Vector3r(follower_pos.x_val, follower_pos.y_val, next_z)
            print(f"[investigate] Descending to z={next_z:.2f} (step {step+1})")
            moved = self.move_follower_to(descent_target, wait=True, timeout=8.0)
            if not moved:
                print("[investigate] Descent move did not complete within timeout (continuing attempts)")
            descent_steps_done += 1
            time.sleep(0.35)

        # If not confirmed in loop, perform final confirmation attempts and log NOT CONFIRMED if still negative
        if not confirmed:
            for i in range(INVESTIGATE_CONFIRM_SAMPLES):
                img = self.sim_get_image_bgr(FOLLOWER_CAMERA, FOLLOWER)
                if img is None:
                    time.sleep(INVESTIGATE_CONFIRM_INTERVAL)
                    continue
                with self.model_lock:
                    res = self.model(img)
                preds = res.xyxy[0].cpu().numpy()
                local_best2 = None
                for *box, conf, cls in preds:
                    name = self.model.names[int(cls)].lower()
                    if name in ("person", "people") and conf >= CONF_THRESH:
                        x1, y1, x2, y2 = map(float, box)
                        if local_best2 is None or conf > local_best2["conf"]:
                            local_best2 = {"bbox": (x1, y1, x2, y2), "conf": float(conf)}
                if local_best2 is not None:
                    confirmed = True
                    with self.display_lock:
                        self.follower_detection_for_display = {"bbox": local_best2["bbox"], "conf": local_best2["conf"]}
                    if local_best2["conf"] > self.follower_max_conf_during_investigation:
                        self.follower_max_conf_during_investigation = local_best2["conf"]
                    print(f"[investigate] Late confirmation detected conf={local_best2['conf']:.2f}")
                    break
                time.sleep(INVESTIGATE_CONFIRM_INTERVAL)

        # Metrics & release control (if not logged earlier)
        if not confirmed:
            inv_duration = time.time() - inv_start_time
            fpos_final = self.get_state(FOLLOWER).kinematics_estimated.position
            lpos_final = self.get_state(LEADER).kinematics_estimated.position
            follower_bbox = tuple(map(int, self.follower_detection_for_display["bbox"])) if self.follower_detection_for_display else None
            self.log_investigation(
                inv_id,
                leader_conf_before,
                self.follower_max_conf_during_investigation,
                False,
                inv_duration,
                descent_steps_done,
                lpos_final,
                fpos_final,
                leader_bbox,
                follower_bbox,
                note="not_confirmed"
            )
            print(f"[investigate RESULT] leader_conf={leader_conf_before:.2f}, follower_max_conf={self.follower_max_conf_during_investigation:.2f} => NOT CONFIRMED")

        # Release exclusive control and resume simple following behaviour
        time.sleep(0.4)
        self.disable_follower_loop = False
        self.investigating = False
        self.is_following = True
        print("[investigate] Finished, resuming follow behaviour")

    # ---------------- Display thread (with keyboard controls) ----------------
    def display_loop(self):
        """
        Combined leader/follower view with overlays and keyboard controls.
        Keys (when OpenCV window is focused):
            q - Quit entire programme (sets should_exit)
            g - Go (resume following)
            s - Stop (hover)
            i - Investigate (start investigate_once in background)
            d - Dismiss detection
            r - Return / resume follow (stop investigate)
            l - Land follower (and attempt leader landing briefly)
            t - Print status
        """
        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("[display] Window started. Focus this window and press keys: g/go, s/stop, i/investigate, d/dismiss, r/return, l/land, t/status, q/quit")
        delay = 30  # Milliseconds between frames for cv2.waitKey

        while not self.should_exit:
            left = None; right = None
            try:
                with self.display_lock:
                    if self.leader_frame is not None:
                        left = self.leader_frame.copy()
                    if self.follower_frame is not None:
                        right = self.follower_frame.copy()

                    # Leader overlay (draw bounding box and confidence)
                    if left is not None and self.leader_detection_for_display:
                        try:
                            x1, y1, x2, y2 = map(int, self.leader_detection_for_display["bbox"])
                            conf = self.leader_detection_for_display["conf"]
                            cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(left, f"leader: {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        except Exception:
                            pass

                    # Follower overlay (draw bounding box and confidence)
                    if right is not None and self.follower_detection_for_display:
                        try:
                            x1, y1, x2, y2 = map(int, self.follower_detection_for_display["bbox"])
                            conf = self.follower_detection_for_display["conf"]
                            cv2.rectangle(right, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(right, f"follower: {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        except Exception:
                            pass

                    # Metric text to display above the frames
                    leader_conf = float(self.last_detection["conf"]) if self.last_detection else 0.0
                    follower_conf = float(self.follower_max_conf_during_investigation)
                    metric_text = f"leader_conf={leader_conf:.2f}  follower_max_conf={follower_conf:.2f}  confirmed={self.last_detection.get('confirmed', False) if self.last_detection else False}"

                # Build the side-by-side canvas for display
                if left is None and right is None:
                    canvas = np.zeros((480, 960, 3), dtype=np.uint8)
                    cv2.putText(canvas, "Waiting for frames...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                else:
                    H = 360
                    def fit(img):
                        """Resize an image to the fixed height H preserving aspect ratio."""
                        if img is None:
                            return np.zeros((H, int(H*4/3), 3), dtype=np.uint8)
                        h, w = img.shape[:2]
                        new_w = int(w * (H / h))
                        return cv2.resize(img, (new_w, H))
                    leftr = fit(left)
                    rightr = fit(right)
                    canvas = np.hstack([leftr, rightr])
                    cv2.putText(canvas, metric_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show the canvas in the named window
                cv2.imshow(DISPLAY_WINDOW_NAME, canvas)

                # Keyboard handling (window must be focused for keypresses to register)
                key = cv2.waitKey(delay) & 0xFF
                if key != 0xFF:
                    try:
                        kch = chr(key).lower()
                    except Exception:
                        kch = ''
                    if kch == 'q':
                        print("[display] 'q' pressed - exiting")
                        self.should_exit = True
                        break
                    elif kch == 'g':
                        self.is_following = True
                        self.investigating = False
                        print("[kbd] Go/resume following")
                    elif kch == 's':
                        self.is_following = False
                        print("[kbd] Stop/hover")
                    elif kch == 'i':
                        # Trigger investigate only if we have a recent detection stored
                        if self.last_detection and self.last_detection.get("bbox"):
                            print("[kbd] Investigate triggered")
                            self.is_following = False
                            self.investigating = True
                            t = threading.Thread(target=self.investigate_once, daemon=True)
                            t.start()
                        else:
                            print("[kbd] No detection to investigate")
                    elif kch == 'd':
                        # Dismiss the current detection and clear overlays
                        self.last_detection = None
                        with self.display_lock:
                            self.leader_detection_for_display = None
                            self.follower_detection_for_display = None
                        print("[kbd] Detection dismissed")
                    elif kch == 'r':
                        if self.investigating:
                            self.investigating = False
                            self.is_following = True
                            print("[kbd] Return/resume follow")
                        else:
                            print("[kbd] Not currently investigating")
                    elif kch == 'l':
                        # Land follower and attempt to land leader via API (briefly enable API control)
                        print("[kbd] Land follower (attempt leader land as well)")
                        try:
                            with self.client_lock:
                                self.client.landAsync(vehicle_name=FOLLOWER).join()
                                self.client.armDisarm(False, FOLLOWER)
                        except Exception as e:
                            print("[kbd] Follower land error:", e)
                        # try:
                        #     with self.client_lock:
                        #         print("[kbd] Attempting to land leader via API (temporary API control).")
                        #         self.client.enableApiControl(True, LEADER)
                        #         self.client.landAsync(vehicle_name=LEADER).join()
                        #         self.client.armDisarm(False, LEADER)
                        #         self.client.enableApiControl(False, LEADER)
                        # except Exception as e:
                        #     print("[kbd] Leader land attempt error:", e)
                    elif kch == 't':
                        # Print a concise status summary to the console
                        try:
                            l = self.get_state(LEADER).kinematics_estimated.position
                            f = self.get_state(FOLLOWER).kinematics_estimated.position
                            print(f"[status] Leader: x={l.x_val:.2f}, y={l.y_val:.2f}, z={l.z_val:.2f}")
                            print(f"[status] Follower: x={f.x_val:.2f}, y={f.y_val:.2f}, z={f.z_val:.2f}")
                            print(f"[status] following={self.is_following}, investigating={self.investigating}")
                            if self.last_detection:
                                tgt = self.last_detection.get("target_pos")
                                conf = self.last_detection.get("conf", 0.0)
                                confd = self.last_detection.get("confirmed", False)
                                if tgt:
                                    print(f"  Last detection: conf={conf:.2f}, confirmed={confd}, target ~ x={tgt.x_val:.2f}, y={tgt.y_val:.2f}, z={tgt.z_val:.2f}")
                                else:
                                    print(f"  Last detection: conf={conf:.2f}, confirmed={confd}, (no target_pos)")
                            else:
                                print("  No last_detection")
                        except Exception as e:
                            print("[status] Error:", e)
                    # Unhandled keys are ignored
            except Exception as e:
                # Print occasional display loop exceptions to aid debugging without spamming
                print("[display] Error:", repr(e))
                time.sleep(0.05)

        # Attempt to destroy the OpenCV window cleanly on exit
        try:
            cv2.destroyWindow(DISPLAY_WINDOW_NAME)
        except Exception:
            pass
        print("[display] Exiting")

    # ---------------- CLI ----------------
    def command_loop(self):
        """Simple console command loop retained for compatibility with original CLI commands."""
        print("[command] Commands: go, stop, status, investigate, dismiss, land, exit")
        while not self.should_exit:
            try:
                cmd = input("> ").strip().lower()
            except EOFError:
                cmd = "exit"
            if cmd == "go":
                self.is_following = True
                self.investigating = False
                print("[command] Following enabled")
            elif cmd == "stop":
                self.is_following = False
                print("[command] Following disabled (hovering)")
            elif cmd == "status":
                try:
                    l = self.get_state(LEADER).kinematics_estimated.position
                    f = self.get_state(FOLLOWER).kinematics_estimated.position
                    print(f"[status] Leader: x={l.x_val:.2f}, y={l.y_val:.2f}, z={l.z_val:.2f}")
                    print(f"[status] Follower: x={f.x_val:.2f}, y={f.y_val:.2f}, z={f.z_val:.2f}")
                    print(f"[status] following={self.is_following}, investigating={self.investigating}")
                    if self.last_detection:
                        tgt = self.last_detection.get("target_pos")
                        conf = self.last_detection.get("conf", 0.0)
                        confd = self.last_detection.get("confirmed", False)
                        if tgt:
                            print(f"  Last detection: conf={conf:.2f}, confirmed={confd}, target ~ x={tgt.x_val:.2f}, y={tgt.y_val:.2f}, z={tgt.z_val:.2f}")
                        else:
                            print(f"  Last detection: conf={conf:.2f}, confirmed={confd}, (no target_pos)")
                    else:
                        print("  No last_detection")
                except Exception as e:
                    print("[status] Error:", e)
            elif cmd == "investigate":
                if not self.last_detection:
                    print("[command] No detection to investigate")
                else:
                    t = threading.Thread(target=self.investigate_once, daemon=True)
                    t.start()
            elif cmd == "dismiss":
                self.last_detection = None
                with self.display_lock:
                    self.leader_detection_for_display = None
                    self.follower_detection_for_display = None
                print("[command] Last detection dismissed")
            elif cmd == "land":
                print("[command] Landing follower (and attempting leader land via API briefly).")
                try:
                    with self.client_lock:
                        self.client.landAsync(vehicle_name=FOLLOWER).join()
                        self.client.armDisarm(False, FOLLOWER)
                except Exception as e:
                    print("[land] Follower land error:", e)
                try:
                    print("[land] Attempting to land leader via API control (may override manual input briefly).")
                    with self.client_lock:
                        self.client.enableApiControl(True, LEADER)
                        self.client.landAsync(vehicle_name=LEADER).join()
                        self.client.armDisarm(False, LEADER)
                        self.client.enableApiControl(False, LEADER)
                except Exception as e:
                    print("[land] Leader land attempt error (you may land leader manually):", e)
            elif cmd == "exit":
                print("[command] Exit requested.")
                self.should_exit = True
                break
            else:
                print("[command] Unknown:", cmd)
        print("[command] Loop exiting")

    # ---------------- Run ----------------
    def run(self):
        """Start threads and enter the main loop until exit is requested."""
        try:
            self.setup_follower()
            self.load_model()

            # Start background threads for detection, following, camera update, display and CLI
            det_t = threading.Thread(target=self.detector_loop, daemon=True)
            fol_t = threading.Thread(target=self.follower_loop, daemon=True)
            cam_t = threading.Thread(target=self.follower_camera_updater, daemon=True)
            disp_t = threading.Thread(target=self.display_loop, daemon=True)
            cmd_t = threading.Thread(target=self.command_loop, daemon=True)
            det_t.start(); fol_t.start(); cam_t.start(); disp_t.start(); cmd_t.start()

            # Default behaviour is to follow the leader; user can override via keys or CLI
            self.is_following = True

            while not self.should_exit:
                time.sleep(0.2)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: shutting down")
            self.should_exit = True
        finally:
            # Attempt a safe shutdown: land follower and disarm
            try:
                print("[shutdown] Landing follower and disarming...")
                with self.client_lock:
                    self.client.landAsync(vehicle_name=FOLLOWER).join()
                    self.client.armDisarm(False, FOLLOWER)
            except Exception:
                pass
            print("[shutdown] Done. Exiting.")


if __name__ == "__main__":
    app = DualUAV()
    app.run()

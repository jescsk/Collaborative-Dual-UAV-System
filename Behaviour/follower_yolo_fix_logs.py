"""
Behaviour summary:
- Drone1 (leader) is manual-driven.
- Drone2 (follower) is API-driven.
- Detection thread captures images from leader camera and runs YOLOv5.
- On person detection above CONF_THRESH, a detection is stored and an alert printed.
- CLI commands:
    go         - resume normal following
    stop       - stop following (hover)
    status     - print positions + detection status
    investigate- send follower to last detected position (pauses normal following)
    dismiss    - dismiss last detection
    return     - stop investigating and resume following (if previously following)
    land       - land follower and leader (briefly enables API control for leader)
    exit       - stop script (doesn't auto-land leader)
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
CONF_THRESH = 0.65
DETECT_INTERVAL = 0.8
FORWARD_ESTIMATE = 10.0
INVESTIGATE_ORBIT_RADIUS = 6.0
INVESTIGATE_CONFIRM_SAMPLES = 6
INVESTIGATE_CONFIRM_INTERVAL = 0.6
SAFE_ALT_RELATIVE = -2.0
FOLLOW_DISTANCE = 8.0
LATERAL_OFFSET = 3.0
MAX_SPEED = 5.0
UPDATE_RATE = 8
ALERT_SUPPRESS_S = 5.0
DISPLAY_WINDOW_NAME = "Leader (L) --- Follower (R)"

# Descent tuning:
DESCENT_STEP_M = 4.0            # Drone altitude lowering rate for investigation
MAX_DESCENT_STEPS = 8
MIN_ALT_ABOVE_LEADER = 1.25
ARRIVAL_TOLERANCE_M = 0.15
ARRIVAL_TIMEOUT_S = 12.0

# Pause-and-confirm tuning (when follower detects)
PAUSE_ON_CONFIRM_SAMPLES = 5     # Number of extra samples to collect while hovering
PAUSE_ON_CONFIRM_INTERVAL = 0.5  # Seconds between the extra samples
PAUSE_ON_CONFIRM_HOLD_S = 3.0    # How long to hold (hover) after initial confirmation before resuming

# Logging
LOG_CSV = "investigation_log.csv"
LOG_LOCK = threading.Lock()
# ---------------------------------

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

class FollowerWithImprovedInvestigate:
    def __init__(self):
        print("[init] connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # state flags
        self.is_following = False
        self.investigating = False
        self.disable_follower_loop = False
        self.should_exit = False

        # model + detection state
        self.model = None
        self.model_lock = threading.Lock()
        self.last_detection = None
        self.last_detection_time = 0.0
        self.last_alert_print_time = 0.0
        self.last_alert_signature = None

        # display frames and overlays
        self.leader_frame = None
        self.follower_frame = None
        self.leader_detection_for_display = None
        self.follower_detection_for_display = None
        self.follower_max_conf_during_investigation = 0.0

        # locks
        self.client_lock = threading.Lock()
        self.display_lock = threading.Lock()

        # ensure CSV header exists
        self._ensure_log_header()

    # ---------------- logging ----------------
    def _ensure_log_header(self):
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
                        "leader_x","leader_y","leader_z",
                        "follower_x","follower_y","follower_z",
                        "leader_bbox",  # as x1:y1:x2:y2 or empty
                        "follower_bbox",
                        "note"
                    ])

    def log_investigation(self, inv_id, leader_conf, follower_max_conf, confirmed, duration_s,
                         descent_steps, leader_pos, follower_pos, leader_bbox, follower_bbox, note=""):
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
        print(f"[log] investigation logged to {LOG_CSV} (id={inv_id})")

    # ---------------- utils ----------------
    def load_model(self):
        print("[model] loading YOLOv5 (yolov5s) via torch.hub (may download weights first run)...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = CONF_THRESH
        print("[model] loaded. classes:", self.model.names)

    def quaternion_to_yaw(self, q: airsim.Quaternionr) -> float:
        w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_state(self, vehicle_name: str):
        with self.client_lock:
            return self.client.getMultirotorState(vehicle_name=vehicle_name)

    def sim_get_image_bgr(self, camera_name: str, vehicle_name: str) -> Optional[np.ndarray]:
        with self.client_lock:
            raw = self.client.simGetImage(camera_name, airsim.ImageType.Scene, vehicle_name=vehicle_name)
        if not raw:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    # ---------------- follower setup & motion ----------------
    def setup_follower(self):
        print("[setup] enabling API control and arming follower:", FOLLOWER)
        with self.client_lock:
            self.client.enableApiControl(True, FOLLOWER)
            self.client.armDisarm(True, FOLLOWER)
            self.client.takeoffAsync(vehicle_name=FOLLOWER).join()
        time.sleep(0.5)
        try:
            self.client.enableApiControl(False, LEADER)
        except Exception:
            pass

    def move_follower_to(self, pos: airsim.Vector3r, wait: bool = False, timeout: float = ARRIVAL_TIMEOUT_S):
        """Move follower to pos. If wait True, wait for the AirSim future to finish (.join())."""
        try:
            with self.client_lock:
                fut = self.client.moveToPositionAsync(pos.x_val, pos.y_val, pos.z_val, MAX_SPEED, vehicle_name=FOLLOWER)
            if not wait or fut is None:
                return True
            try:
                fut.join()
            except Exception:
                start = time.time()
                while True:
                    st = self.get_state(FOLLOWER)
                    cur = st.kinematics_estimated.position
                    dx = cur.x_val - pos.x_val
                    dy = cur.y_val - pos.y_val
                    dz = cur.z_val - pos.z_val
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if dist <= ARRIVAL_TOLERANCE_M:
                        return True
                    if time.time() - start > timeout:
                        print(f"[move] arrival timeout after {timeout}s (dist={dist:.2f}m)")
                        return False
                    time.sleep(0.15)
            st = self.get_state(FOLLOWER)
            cur = st.kinematics_estimated.position
            dx = cur.x_val - pos.x_val
            dy = cur.y_val - pos.y_val
            dz = cur.z_val - pos.z_val
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist <= max(ARRIVAL_TOLERANCE_M, 0.05):
                return True
            else:
                print(f"[move] warning: after join still dist={dist:.2f}m")
                return True
        except Exception as e:
            print("[move] error:", e)
            return False

    def force_descend_chunk(self, chunk_m: float, speed_mps: float):
        if chunk_m <= 0 or speed_mps <= 0:
            return False
        duration = max(0.6, abs(chunk_m) / speed_mps + 0.3)
        try:
            with self.client_lock:
                fut = self.client.moveByVelocityAsync(0, 0, float(speed_mps), duration, vehicle_name=FOLLOWER)
                fut.join()
            return True
        except Exception as e:
            print("[force_descend] error:", e)
            return False

    def hover_follower(self):
        try:
            with self.client_lock:
                z = self.client.getMultirotorState(vehicle_name=FOLLOWER).kinematics_estimated.position.z_val
                self.client.moveByVelocityZAsync(0, 0, z, 1.0, vehicle_name=FOLLOWER)
        except Exception:
            pass

    # ---------------- detector (leader) ----------------
    def detector_loop(self):
        print("[detector] detector thread started")
        while not self.should_exit:
            try:
                img = self.sim_get_image_bgr(LEADER_CAMERA, LEADER)
                if img is None:
                    time.sleep(DETECT_INTERVAL)
                    continue

                with self.display_lock:
                    self.leader_frame = img.copy()

                with self.model_lock:
                    results = self.model(img)
                preds = results.xyxy[0].cpu().numpy()

                best = None
                for *box, conf, cls in preds:
                    class_id = int(cls)
                    name = self.model.names[class_id].lower()
                    if name in ("person", "people") and conf >= CONF_THRESH:
                        x1, y1, x2, y2 = map(float, box)
                        if best is None or conf > best["conf"]:
                            best = {"bbox": (x1, y1, x2, y2), "conf": float(conf)}

                if best is not None:
                    st = self.get_state(LEADER)
                    leader_pos = st.kinematics_estimated.position
                    leader_orient = st.kinematics_estimated.orientation
                    yaw = self.quaternion_to_yaw(leader_orient)
                    fx = math.cos(yaw); fy = math.sin(yaw)
                    world_x = leader_pos.x_val + fx * FORWARD_ESTIMATE
                    world_y = leader_pos.y_val + fy * FORWARD_ESTIMATE
                    target = airsim.Vector3r(world_x, world_y, leader_pos.z_val)

                    signature = (round(best["conf"],3), round(world_x,2), round(world_y,2))
                    now = time.time()
                    if signature != self.last_alert_signature or (now - self.last_alert_print_time) > ALERT_SUPPRESS_S:
                        self.last_alert_signature = signature
                        self.last_alert_print_time = now
                        print(f"[ALERT] Leader detected person conf={best['conf']:.2f} at approx world x={world_x:.2f},y={world_y:.2f}")
                        print("         Type 'investigate' to send follower to inspect, or 'dismiss' to ignore.")

                    self.last_detection = {"class_name": "person", "conf": best["conf"], "bbox": best["bbox"],
                                           "target_pos": target, "confirmed": False}
                    self.last_detection_time = now

                    with self.display_lock:
                        self.leader_detection_for_display = {"bbox": best["bbox"], "conf": best["conf"]}
                else:
                    with self.display_lock:
                        self.leader_detection_for_display = None

                time.sleep(DETECT_INTERVAL)
            except Exception as e:
                print("[detector] error:", repr(e))
                time.sleep(0.5)
        print("[detector] exiting")

    # -------------- follower loop ----------------
    def follower_loop(self):
        print("[follower_loop] started")
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
                    dx = -FOLLOW_DISTANCE * math.cos(yaw) + LATERAL_OFFSET * math.sin(yaw)
                    dy = -FOLLOW_DISTANCE * math.sin(yaw) - LATERAL_OFFSET * math.cos(yaw)
                    dz = leader_pos.z_val - 2.0
                    target = airsim.Vector3r(leader_pos.x_val + dx, leader_pos.y_val + dy, dz)
                    self.move_follower_to(target, wait=False)
                else:
                    self.hover_follower()
            except Exception as e:
                print("[follower_loop] error:", repr(e))
            time.sleep(1.0 / UPDATE_RATE)
        print("[follower_loop] exiting")

    # ------------- follower camera updater -------------
    def follower_camera_updater(self):
        while not self.should_exit:
            try:
                img = self.sim_get_image_bgr(FOLLOWER_CAMERA, FOLLOWER)
                if img is not None:
                    with self.display_lock:
                        self.follower_frame = img.copy()
                time.sleep(0.6)
            except Exception as e:
                print("[camera_updater] error:", repr(e))
                time.sleep(1.0)

    # ----------------- investigate (move -> rotate -> descend) -----------------
    def rotate_follower_to_face_leader(self):
        try:
            fstate = self.get_state(FOLLOWER)
            follower_pos = fstate.kinematics_estimated.position
            lstate = self.get_state(LEADER)
            leader_pos = lstate.kinematics_estimated.position
            vx = leader_pos.x_val - follower_pos.x_val
            vy = leader_pos.y_val - follower_pos.y_val
            yaw_rad = math.atan2(vy, vx)
            yaw_deg = math.degrees(yaw_rad)
            with self.client_lock:
                self.client.moveByVelocityZAsync(0, 0, follower_pos.z_val, 1.0,
                                                 yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg),
                                                 vehicle_name=FOLLOWER).join()
            time.sleep(0.25)
        except Exception as e:
            print("[rotate] error:", e)

    def investigate_once(self):
        """
        Move to opposite side, rotate to face leader, then descend stepwise.
        On first detection by follower, hold (pause) for extra samples and log metrics.
        """
        if not self.last_detection or not self.last_detection.get("target_pos"):
            print("[investigate] no recent detection to investigate.")
            return

        inv_start_time = time.time()
        inv_id = int(inv_start_time)  # simple id for logging
        leader_conf_before = float(self.last_detection.get("conf", 0.0))
        leader_bbox = tuple(map(int, self.last_detection.get("bbox", (0,0,0,0)))) if self.last_detection.get("bbox") else None

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

        print(f"[investigate] beginning investigate: waypoint x={wx:.2f}, y={wy:.2f}, z={initial_wz:.2f}")

        # claim exclusive control
        self.disable_follower_loop = True
        self.investigating = True
        self.follower_max_conf_during_investigation = 0.0
        descent_steps_done = 0
        with self.display_lock:
            self.follower_detection_for_display = None

        # 1) travel to waypoint and wait using fut.join()
        arrived = self.move_follower_to(waypoint, wait=True, timeout=ARRIVAL_TIMEOUT_S)
        if not arrived:
            print("[investigate] WARNING: follower did not reach waypoint reliably; proceeding anyway.")
        else:
            print("[investigate] follower reached waypoint")

        time.sleep(0.6)
        self.hover_follower()
        time.sleep(0.3)

        # 2) rotate to face leader
        print("[investigate] rotating to face leader")
        self.rotate_follower_to_face_leader()
        time.sleep(0.25)

        # 3) descent loop
        confirmed = False
        for step in range(MAX_DESCENT_STEPS + 1):
            # attempt detection at current altitude
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
                # FIRST confirmation event: update display, update max conf, then pause and collect extra samples
                confirmed = True
                with self.display_lock:
                    self.follower_detection_for_display = {"bbox": local_best["bbox"], "conf": local_best["conf"]}
                if local_best["conf"] > self.follower_max_conf_during_investigation:
                    self.follower_max_conf_during_investigation = local_best["conf"]
                print(f"[investigate] FIRST detection by follower at step {step+1} conf={local_best['conf']:.2f} - PAUSING to confirm")

                # Pause & collect additional confirmation samples while hovering
                hold_samples = []
                hold_start = time.time()
                # hover in place
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

                # optional extra hold time for visual confirmation
                time.sleep(PAUSE_ON_CONFIRM_HOLD_S)

                # collect follower pos & leader pos for logging
                fpos_after = self.get_state(FOLLOWER).kinematics_estimated.position
                lpos_after = self.get_state(LEADER).kinematics_estimated.position
                follower_bbox = tuple(map(int, self.follower_detection_for_display["bbox"])) if self.follower_detection_for_display else None

                # finish: compute metrics and log
                inv_duration = time.time() - inv_start_time
                leader_pos_at_start = leader_pos
                follower_pos_at_end = fpos_after
                # log the investigation result
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

                # break out and resume follow behavior (we already hovered)
                break

            # if not confirmed, prepare to descend one step
            if step >= MAX_DESCENT_STEPS:
                print("[investigate] reached maximum descent rounds without confirmation")
                break

            # compute next z from current follower position (NED convention used consistently)
            fstate = self.get_state(FOLLOWER)
            follower_pos = fstate.kinematics_estimated.position
            leader_pos = self.get_state(LEADER).kinematics_estimated.position
            next_z = follower_pos.z_val + DESCENT_STEP_M
            min_allowed_z = leader_pos.z_val + MIN_ALT_ABOVE_LEADER
            # if next_z >= min_allowed_z:
            #     print("[investigate] cannot descend further without breaching safety limit")
            #     break

            # command descent and wait for it to complete
            descent_target = airsim.Vector3r(follower_pos.x_val, follower_pos.y_val, next_z)
            print(f"[investigate] descending to z={next_z:.2f} (step {step+1})")
            moved = self.move_follower_to(descent_target, wait=True, timeout=8.0)
            if not moved:
                print("[investigate] descent move did not complete within timeout (continuing attempts)")
            descent_steps_done += 1
            time.sleep(0.35)

        # if we exited loop without confirmed True, run final confirm samples and log NOT CONFIRMED
        if not confirmed:
            # final confirm attempts
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
                    print(f"[investigate] late confirmation detected conf={local_best2['conf']:.2f}")
                    break
                time.sleep(INVESTIGATE_CONFIRM_INTERVAL)

        # metrics & release control (if not already logged)
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

        # release exclusive control and resume following
        time.sleep(0.4)
        self.disable_follower_loop = False
        self.investigating = False
        self.is_following = True
        print("[investigate] finished, resuming follow behavior")

    # ---------------- display thread ----------------
    def display_loop(self):
        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("[display] window started. Press 'q' in window or type 'exit' to quit.")
        while not self.should_exit:
            left = None; right = None
            with self.display_lock:
                if self.leader_frame is not None:
                    left = self.leader_frame.copy()
                if self.follower_frame is not None:
                    right = self.follower_frame.copy()

                if left is not None and self.leader_detection_for_display:
                    x1,y1,x2,y2 = map(int, self.leader_detection_for_display["bbox"])
                    conf = self.leader_detection_for_display["conf"]
                    cv2.rectangle(left, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(left, f"leader: {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                if right is not None and self.follower_detection_for_display:
                    x1,y1,x2,y2 = map(int, self.follower_detection_for_display["bbox"])
                    conf = self.follower_detection_for_display["conf"]
                    cv2.rectangle(right, (x1,y1), (x2,y2), (255,0,0), 2)
                    cv2.putText(right, f"follower: {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                leader_conf = float(self.last_detection["conf"]) if self.last_detection else 0.0
                follower_conf = float(self.follower_max_conf_during_investigation)
                metric_text = f"leader_conf={leader_conf:.2f}  follower_max_conf={follower_conf:.2f}  confirmed={self.last_detection.get('confirmed', False) if self.last_detection else False}"

            if left is None and right is None:
                canvas = np.zeros((480, 960, 3), dtype=np.uint8)
                cv2.putText(canvas, "Waiting for frames...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
            else:
                H = 360
                def fit(img):
                    if img is None:
                        return np.zeros((H, int(H*4/3), 3), dtype=np.uint8)
                    h,w = img.shape[:2]
                    new_w = int(w * (H / h))
                    return cv2.resize(img, (new_w, H))
                leftr = fit(left)
                rightr = fit(right)
                canvas = np.hstack([leftr, rightr])
                cv2.putText(canvas, metric_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow(DISPLAY_WINDOW_NAME, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                self.should_exit = True
                break

        cv2.destroyWindow(DISPLAY_WINDOW_NAME)
        print("[display] exiting")

    # ---------------- CLI ----------------
    def command_loop(self):
        print("[command] commands: go, stop, status, investigate, dismiss, land, exit")
        while not self.should_exit:
            try:
                cmd = input("> ").strip().lower()
            except EOFError:
                cmd = "exit"
            if cmd == "go":
                self.is_following = True
                self.investigating = False
                print("[command] following enabled")
            elif cmd == "stop":
                self.is_following = False
                print("[command] following disabled (hovering)")
            elif cmd == "status":
                try:
                    l = self.get_state(LEADER).kinematics_estimated.position
                    f = self.get_state(FOLLOWER).kinematics_estimated.position
                    print(f"[status] leader: x={l.x_val:.2f}, y={l.y_val:.2f}, z={l.z_val:.2f}")
                    print(f"[status] follower: x={f.x_val:.2f}, y={f.y_val:.2f}, z={f.z_val:.2f}")
                    print(f"[status] following={self.is_following}, investigating={self.investigating}")
                    if self.last_detection:
                        tgt = self.last_detection.get("target_pos")
                        conf = self.last_detection.get("conf", 0.0)
                        confd = self.last_detection.get("confirmed", False)
                        if tgt:
                            print(f"  last_detection: conf={conf:.2f}, confirmed={confd}, target ~ x={tgt.x_val:.2f}, y={tgt.y_val:.2f}, z={tgt.z_val:.2f}")
                        else:
                            print(f"  last_detection: conf={conf:.2f}, confirmed={confd}, (no target_pos)")
                    else:
                        print("  no last_detection")
                except Exception as e:
                    print("[status] error:", e)
            elif cmd == "investigate":
                if not self.last_detection:
                    print("[command] no detection to investigate")
                else:
                    t = threading.Thread(target=self.investigate_once, daemon=True)
                    t.start()
            elif cmd == "dismiss":
                self.last_detection = None
                with self.display_lock:
                    self.leader_detection_for_display = None
                    self.follower_detection_for_display = None
                print("[command] last detection dismissed")
            elif cmd == "land":
                print("[command] landing follower (and attempting leader land via API briefly).")
                try:
                    with self.client_lock:
                        self.client.landAsync(vehicle_name=FOLLOWER).join()
                        self.client.armDisarm(False, FOLLOWER)
                except Exception as e:
                    print("[land] follower land error:", e)
                try:
                    print("[land] attempting to land leader via API control (will override manual input briefly).")
                    self.client.enableApiControl(True, LEADER)
                    self.client.landAsync(vehicle_name=LEADER).join()
                    self.client.armDisarm(False, LEADER)
                    self.client.enableApiControl(False, LEADER)
                except Exception as e:
                    print("[land] leader land attempt error (you may land leader manually):", e)
            elif cmd == "exit":
                print("[command] exit requested.")
                self.should_exit = True
                break
            else:
                print("[command] unknown:", cmd)
        print("[command] loop exiting")

    # ---------------- run ----------------
    def run(self):
        try:
            self.setup_follower()
            self.load_model()

            det_t = threading.Thread(target=self.detector_loop, daemon=True)
            fol_t = threading.Thread(target=self.follower_loop, daemon=True)
            cam_t = threading.Thread(target=self.follower_camera_updater, daemon=True)
            disp_t = threading.Thread(target=self.display_loop, daemon=True)
            cmd_t = threading.Thread(target=self.command_loop, daemon=True)
            det_t.start(); fol_t.start(); cam_t.start(); disp_t.start(); cmd_t.start()

            self.is_following = True

            while not self.should_exit:
                time.sleep(0.2)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: shutting down")
            self.should_exit = True
        finally:
            try:
                print("[shutdown] landing follower and disarming...")
                with self.client_lock:
                    self.client.landAsync(vehicle_name=FOLLOWER).join()
                    self.client.armDisarm(False, FOLLOWER)
            except Exception:
                pass
            print("[shutdown] done. Exiting.")


if __name__ == "__main__":
    app = FollowerWithImprovedInvestigate()
    app.run()

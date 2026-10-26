import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from uuid import uuid4

import cv2
import torch
from ultralytics import YOLO

# ============================================================
# PATH / ENV
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "wearpack.db")
PENDING_DIR = os.path.join(BASE_DIR, "spool", "pending")
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
LIVE_FRAME_PATH = os.path.join(RUNTIME_DIR, "latest.jpg")
LIVE_META_PATH = os.path.join(RUNTIME_DIR, "detections.json")

os.makedirs(PENDING_DIR, exist_ok=True)
os.makedirs(RUNTIME_DIR, exist_ok=True)

RTSP_URL = os.getenv("RTSP_URL")
if not RTSP_URL:
    raise RuntimeError("RTSP_URL belum diset")

CAMERA_NAME = os.getenv("CAMERA_NAME", "CAM01")

# ============================================================
# MODEL / PERFORMANCE
# ============================================================
MODEL_NAME = os.getenv("PERSON_MODEL", "yolo11n.pt")
PERSON_CONFIDENCE = float(os.getenv("PERSON_CONFIDENCE", "0.35"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "512"))
TARGET_AI_FPS = float(os.getenv("TARGET_AI_FPS", "10"))

# ============================================================
# LIVE PREVIEW PUBLISHING
# ============================================================
LIVE_PREVIEW_FPS = float(os.getenv("LIVE_PREVIEW_FPS", "6"))
LIVE_PREVIEW_WIDTH = int(os.getenv("LIVE_PREVIEW_WIDTH", "960"))
LIVE_JPEG_QUALITY = int(os.getenv("LIVE_JPEG_QUALITY", "75"))

# ============================================================
# CAPTURE POLICY
# ============================================================

FRAME_SELECTION_WINDOW = float(
    os.getenv("FRAME_SELECTION_WINDOW", "0.4")
)

MAX_CAPTURES_PER_TRACK = int(
    os.getenv("MAX_CAPTURES_PER_TRACK", "12")
)

SHARPNESS_THRESHOLD = float(
    os.getenv("SHARPNESS_THRESHOLD", "50")
)

MIN_PERSON_HEIGHT = int(
    os.getenv("MIN_PERSON_HEIGHT", "250")
)

MIN_PERSON_WIDTH = int(
    os.getenv("MIN_PERSON_WIDTH", "80")
)

TRACK_TIMEOUT = float(
    os.getenv("TRACK_TIMEOUT", "5")
)

# Capture zone: center person harus berada di area ini.
ZONE_X1 = float(os.getenv("CAPTURE_ZONE_X1", "0.12"))
ZONE_Y1 = float(os.getenv("CAPTURE_ZONE_Y1", "0.08"))
ZONE_X2 = float(os.getenv("CAPTURE_ZONE_X2", "0.88"))
ZONE_Y2 = float(os.getenv("CAPTURE_ZONE_Y2", "0.92"))

SESSION_ID = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Camera: {CAMERA_NAME}")
print(f"[INFO] Session: {SESSION_ID}")
print("[INFO] Loading YOLO...")
model = YOLO(MODEL_NAME)

# ============================================================
# DATABASE
# ============================================================
def insert_capture(timestamp, track_id, filename, sharpness):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO captures (
            timestamp,
            camera,
            session_id,
            track_id,
            filename,
            sharpness,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """,
        (
            timestamp,
            CAMERA_NAME,
            SESSION_ID,
            int(track_id),
            filename,
            float(sharpness),
        ),
    )
    capture_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()
    return capture_id

def get_track_result(track_id):
    conn = sqlite3.connect(
        DB_PATH,
        timeout=5
    )

    conn.row_factory = sqlite3.Row

    row = conn.execute(
        """
        SELECT
            status,
            ocr_text,
            confidence
        FROM track_results
        WHERE
            camera=?
            AND session_id=?
            AND track_id=?
        """,
        (
            CAMERA_NAME,
            SESSION_ID,
            int(track_id)
        )
    ).fetchone()

    conn.close()

    return dict(row) if row else None


def has_open_capture(track_id):
    conn = sqlite3.connect(
        DB_PATH,
        timeout=5
    )

    row = conn.execute(
        """
        SELECT 1
        FROM captures
        WHERE
            camera=?
            AND session_id=?
            AND track_id=?
            AND status IN (
                'pending',
                'processing'
            )
        LIMIT 1
        """,
        (
            CAMERA_NAME,
            SESSION_ID,
            int(track_id)
        )
    ).fetchone()

    conn.close()

    return row is not None


# ============================================================
# IMAGE HELPERS
# ============================================================
def sharpness_score(image):
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_frame_quality(
    frame,
    box,
    torso_sharpness
):
    frame_h, frame_w = frame.shape[:2]

    x1, y1, x2, y2 = box

    person_h = y2 - y1

    if person_h <= 0:
        return -1.0

    # posisi tengah orang
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    frame_cx = frame_w / 2
    frame_cy = frame_h / 2

    # makin dekat tengah frame, makin bagus
    dx = abs(cx - frame_cx) / frame_cx
    dy = abs(cy - frame_cy) / frame_cy

    center_score = max(
        0.0,
        1.0 - ((dx + dy) / 2)
    )

    # makin besar orang di frame, makin bagus
    height_ratio = person_h / frame_h

    size_score = min(
        height_ratio / 0.7,
        1.0
    )

    # sharpness tetap faktor utama
    quality_score = (
        torso_sharpness
        + center_score * 150
        + size_score * 100
    )

    return float(quality_score)


def extract_person_crop(frame, box, pad_x=0.06, pad_y=0.03):
    x1, y1, x2, y2 = [int(v) for v in box]
    frame_h, frame_w = frame.shape[:2]
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    px = int(width * pad_x)
    py = int(height * pad_y)

    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(frame_w, x2 + px)
    cy2 = min(frame_h, y2 + py)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return crop


def get_torso_for_quality(person_crop):
    if person_crop is None or person_crop.size == 0:
        return None
    h, w = person_crop.shape[:2]
    x1 = int(w * 0.18)
    x2 = int(w * 0.82)
    y1 = int(h * 0.08)
    y2 = int(h * 0.56)
    torso = person_crop[y1:y2, x1:x2]
    if torso.size == 0:
        return None
    return torso


def is_in_capture_zone(frame_shape, box):
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    zx1 = frame_w * ZONE_X1
    zy1 = frame_h * ZONE_Y1
    zx2 = frame_w * ZONE_X2
    zy2 = frame_h * ZONE_Y2

    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def save_capture(track_id, crop, sharpness):
    timestamp = datetime.now()
    timestamp_db = timestamp.isoformat(timespec="milliseconds")
    timestamp_file = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    filename = (
        f"{CAMERA_NAME}_{SESSION_ID}_track{int(track_id)}_"
        f"{timestamp_file}_{uuid4().hex[:6]}.jpg"
    )

    final_path = os.path.join(PENDING_DIR, filename)
    temp_path = final_path + ".tmp.jpg"
    ok = cv2.imwrite(
        temp_path,
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    if not ok:
        print("[ERROR] Gagal menyimpan capture")
        return None

    os.replace(temp_path, final_path)
    capture_id = insert_capture(timestamp_db, int(track_id), filename, float(sharpness))
    print(
        f"[CAPTURE] ID={capture_id} | Track={int(track_id)} | "
        f"Sharp={float(sharpness):.1f} | {filename}"
    )
    return capture_id


# ============================================================
# ATOMIC LIVE PUBLISH
# ============================================================
def atomic_write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def publish_live_frame(frame, detections):
    if frame is None or frame.size == 0:
        return

    src_h, src_w = frame.shape[:2]
    if LIVE_PREVIEW_WIDTH > 0 and src_w > LIVE_PREVIEW_WIDTH:
        scale = LIVE_PREVIEW_WIDTH / float(src_w)
        preview = cv2.resize(
            frame,
            (LIVE_PREVIEW_WIDTH, int(src_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = frame.copy()
        scale = 1.0

    preview_h, preview_w = preview.shape[:2]

    preview_detections = []
    for d in detections:
        preview_detections.append(
            {
                "track_id": int(d["track_id"]),
                "confidence": float(d["confidence"]),
                "x1": int(d["x1"] * scale),
                "y1": int(d["y1"] * scale),
                "x2": int(d["x2"] * scale),
                "y2": int(d["y2"] * scale),
                "in_capture_zone": bool(d["in_capture_zone"]),
            }
        )

    tmp_frame = LIVE_FRAME_PATH + ".tmp.jpg"
    ok = cv2.imwrite(
        tmp_frame,
        preview,
        [cv2.IMWRITE_JPEG_QUALITY, LIVE_JPEG_QUALITY],
    )
    if ok:
        os.replace(tmp_frame, LIVE_FRAME_PATH)

    payload = {
        "camera": CAMERA_NAME,
        "session_id": SESSION_ID,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "preview_width": int(preview_w),
        "preview_height": int(preview_h),
        "capture_zone": {
            "x1": int(preview_w * ZONE_X1),
            "y1": int(preview_h * ZONE_Y1),
            "x2": int(preview_w * ZONE_X2),
            "y2": int(preview_h * ZONE_Y2),
        },
        "detections": preview_detections,
    }
    atomic_write_json(LIVE_META_PATH, payload)


# ============================================================
# CAMERA READER
# ============================================================
class CameraReader:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def connect(self):
        print("[CAMERA] Connecting...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cap.isOpened():
            print("[CAMERA] Connected.")
            return True
        print("[CAMERA] Connection failed.")
        return False

    def start(self):
        self.connect()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(2)
                self.connect()
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("[CAMERA] Stream lost. Reconnecting...")
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(2)
                continue

            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stop_event.set()
        if self.cap:
            self.cap.release()


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(DB_PATH):
        raise RuntimeError("Database belum ada. Jalankan: python init_db.py")

    camera = CameraReader(RTSP_URL)
    camera.start()

    track_states = {}
    ai_interval = 1.0 / max(TARGET_AI_FPS, 0.1)
    live_interval = 1.0 / max(LIVE_PREVIEW_FPS, 0.1)
    last_live_publish = 0.0

    print("[INFO] Capture service running. Ctrl+C untuk stop.")

    try:
        while True:
            loop_start = time.time()
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            now = time.time()

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=PERSON_CONFIDENCE,
                imgsz=YOLO_IMAGE_SIZE,
                device=DEVICE,
                verbose=False,
            )

            result = results[0]
            detections = []

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                for box, track_id_raw, conf_raw in zip(boxes, track_ids, confs):
                    track_id = int(track_id_raw)
                    confidence = float(conf_raw)
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    person_width = x2 - x1
                    person_height = y2 - y1
                    in_zone = is_in_capture_zone(frame.shape, (x1, y1, x2, y2))

                    detections.append(
                        {
                            "track_id": track_id,
                            "confidence": confidence,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "in_capture_zone": in_zone,
                        }
                    )

                    state = track_states.setdefault(
                        track_id,
                        {
                            "last_seen": now,
                            "window_start": now,
                            "best_score": -1.0,
                            "best_crop": None,
                            "best_sharpness": 0.0,
                            "capture_count": 0,
                        },
                    )
                    state["last_seen"] = now

                    if state["capture_count"] >= MAX_CAPTURES_PER_TRACK:
                        continue

                    # Hindari partial detection / objek terlalu kecil.
                    if person_height < MIN_PERSON_HEIGHT or person_width < MIN_PERSON_WIDTH:
                        state["window_start"] = now
                        state["best_score"] = -1.0
                        state["best_crop"] = None
                        state["best_sharpness"] = 0.0
                        continue

                    if not in_zone:
                        state["window_start"] = now
                        state["best_score"] = -1.0
                        state["best_crop"] = None
                        state["best_sharpness"] = 0.0
                        continue

                    person_crop = extract_person_crop(frame, (x1, y1, x2, y2))
                    if person_crop is None:
                        continue

                    quality_crop = get_torso_for_quality(person_crop)
                    if quality_crop is None:
                        continue

                    torso_sharpness = sharpness_score(
                        quality_crop
                    )

                    quality_score = calculate_frame_quality(
                        frame,
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        torso_sharpness
                    )

                    if quality_score > state["best_score"]:

                        state["best_score"] = quality_score

                        state["best_sharpness"] = (
                            torso_sharpness
                        )

                        state["best_crop"] = (
                            person_crop.copy()
                        )

                    window_elapsed = (
                        now - state["window_start"]
                    )

                    if (
                        window_elapsed
                        >= FRAME_SELECTION_WINDOW
                    ):

                        # =========================================
                        # CEK HASIL OCR TERAKHIR TRACK INI
                        # =========================================

                        track_result = get_track_result(
                            track_id
                        )

                        status = (
                            track_result.get("status")
                            if track_result
                            else None
                        )


                        # =========================================
                        # SUDAH RECOGNIZED → STOP
                        # =========================================

                        if status == "recognized":

                            state["window_start"] = now
                            state["best_score"] = -1.0
                            state["best_crop"] = None
                            state["best_sharpness"] = 0.0

                            continue


                        # =========================================
                        # OCR SEBELUMNYA MASIH BEKERJA
                        # =========================================

                        if has_open_capture(
                            track_id
                        ):

                            state["window_start"] = now
                            state["best_score"] = -1.0
                            state["best_crop"] = None
                            state["best_sharpness"] = 0.0

                            continue


                        # =========================================
                        # BEST FRAME WINDOW INI
                        # =========================================

                        best_crop = (
                            state["best_crop"]
                        )

                        best_sharpness = (
                            state["best_sharpness"]
                        )


                        if (
                            best_crop is not None
                            and
                            best_sharpness
                            >= SHARPNESS_THRESHOLD
                        ):

                            capture_id = save_capture(
                                track_id,
                                best_crop,
                                best_sharpness
                            )

                            if capture_id is not None:

                                state[
                                    "capture_count"
                                ] += 1

                                print(
                                    f"[FRAME SELECT] "
                                    f"Track={track_id} | "
                                    f"attempt="
                                    f"{state['capture_count']}/"
                                    f"{MAX_CAPTURES_PER_TRACK} | "
                                    f"quality="
                                    f"{state['best_score']:.1f} | "
                                    f"sharp="
                                    f"{best_sharpness:.1f} | "
                                    f"previous="
                                    f"{status or 'initial'}"
                                )


                        # =========================================
                        # RESET UNTUK WINDOW BERIKUTNYA
                        # =========================================

                        state["window_start"] = now
                        state["best_score"] = -1.0
                        state["best_crop"] = None
                        state["best_sharpness"] = 0.0
            if now - last_live_publish >= live_interval:
                publish_live_frame(frame, detections)
                last_live_publish = now

            # Cleanup track state.
            expired = [
                track_id
                for track_id, state in track_states.items()
                if now - state["last_seen"] > TRACK_TIMEOUT
            ]
            for track_id in expired:
                del track_states[track_id]

            elapsed = time.time() - loop_start
            sleep_time = ai_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Stop capture service...")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()

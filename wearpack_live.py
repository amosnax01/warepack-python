import os

# Batasi thread CPU agar EasyOCR tidak mengambil semua resource Mac
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import re
import csv
import time
import queue
import threading
from collections import defaultdict, deque, Counter
from datetime import datetime

import cv2
import easyocr
import torch
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

RTSP_URL = os.getenv("RTSP_URL")

if not RTSP_URL:
    raise RuntimeError(
        "RTSP_URL belum diset. Contoh:\n"
        "export RTSP_URL='rtsp://USER:PASS@192.168.3.2:8554'"
    )


CAMERA_NAME = os.getenv(
    "CAMERA_NAME",
    "CAM01"
)


# ============================================================
# YOLO
# ============================================================

MODEL_NAME = "yolo11n.pt"

PERSON_CONFIDENCE = 0.35

# Lebih ringan daripada 640
YOLO_IMAGE_SIZE = 416

# AI tidak perlu mengikuti seluruh 24 FPS kamera
TARGET_AI_FPS = 10


# ============================================================
# OCR
# ============================================================

# Setiap berapa detik kita coba recognition per orang
OCR_INTERVAL = 0.25

# Simpan history recognition
OCR_HISTORY_SIZE = 5

# Minimal berapa kali kode sama ditemukan
MIN_VOTES = 2

# Minimal rasio voting
MIN_VOTE_RATIO = 0.60

# Confidence minimum EasyOCR
OCR_MIN_CONFIDENCE = 0.65


# ============================================================
# IMAGE QUALITY
# ============================================================

# Hanya OCR crop yang cukup tajam
SHARPNESS_THRESHOLD = 80.0

# Jangan OCR orang yang terlalu jauh/kecil
MIN_PERSON_HEIGHT = 280


# ============================================================
# TRACKING
# ============================================================

# Track dianggap hilang setelah sekian detik
TRACK_TIMEOUT = 5.0

# Mencegah event ganda jika tracker putus sebentar
EVENT_COOLDOWN_SECONDS = 10.0


# ============================================================
# OUTPUT
# ============================================================

LOG_FILE = "wearpack_events.csv"


# ============================================================
# VALID WEARPACK CODE
# ============================================================

VALID_CODES = {
    f"C{i}"
    for i in range(1, 41)
}


# ============================================================
# MAPPING CODE -> NAMA
# ============================================================

PERSON_NAMES = {
    "C1": "DAROTO",

    # nanti tinggal tambah:
    # "C2": "BUDI",
    # "C3": "ANDI",
    # "C4": "SUSANTO",
}


# ============================================================
# CPU / GPU CONFIG
# ============================================================

try:
    torch.set_num_threads(2)
except Exception:
    pass


try:
    cv2.setNumThreads(1)
except Exception:
    pass


if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


print(
    f"[INFO] YOLO device: {DEVICE}"
)


# ============================================================
# LOAD YOLO
# ============================================================

print(
    "[INFO] Loading YOLO..."
)

model = YOLO(
    MODEL_NAME
)


# ============================================================
# LOAD EASYOCR
# ============================================================

# Jangan gpu=True di Mac kita.
# Sebelumnya konflik dengan YOLO MPS / Metal.
print(
    "[INFO] Loading EasyOCR (CPU)..."
)

ocr_reader = easyocr.Reader(
    ["en"],
    gpu=False,
    verbose=False
)


# ============================================================
# SHARED STATE
# ============================================================

latest_ai_frame = None

frame_lock = threading.Lock()

state_lock = threading.Lock()

stop_event = threading.Event()


# OCR queue
ocr_queue = queue.Queue(
    maxsize=8
)

ocr_pending = set()


# History hasil recognition
ocr_history = defaultdict(
    lambda: deque(
        maxlen=OCR_HISTORY_SIZE
    )
)


# Waktu OCR terakhir
last_ocr_time = defaultdict(
    float
)


# Track yang sudah diketahui
stable_code = {}


# Kapan track terakhir terlihat
last_seen = {}


# Mencegah event code yang sama berulang cepat
last_logged_by_code = {}


# ============================================================
# CSV
# ============================================================

def init_log():

    if os.path.exists(LOG_FILE):
        return

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "camera",
            "track_id",
            "wearpack_code",
            "person_name"
        ])


# ============================================================
# LOG EVENT
# ============================================================

def log_event(
    track_id,
    code
):

    now = time.time()

    # ----------------------------------------
    # EVENT COOLDOWN
    # ----------------------------------------

    with state_lock:

        last_log = last_logged_by_code.get(
            code,
            0.0
        )

        if (
            now - last_log
            < EVENT_COOLDOWN_SECONDS
        ):
            return

        last_logged_by_code[code] = now


    person_name = PERSON_NAMES.get(
        code,
        "UNKNOWN"
    )

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )


    # ----------------------------------------
    # SIMPAN CSV
    # ----------------------------------------

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            timestamp,
            CAMERA_NAME,
            track_id,
            code,
            person_name
        ])


    # ----------------------------------------
    # TERMINAL OUTPUT
    # ----------------------------------------

    print(
        f"[EVENT] "
        f"{timestamp} | "
        f"{CAMERA_NAME} | "
        f"Track {track_id} | "
        f"{code} | "
        f"{person_name}"
    )


# ============================================================
# NORMALIZE OCR
# ============================================================

def normalize_code(
    text
):

    text = text.upper()

    # Kesalahan umum OCR
    text = text.replace(
        "|",
        "1"
    )

    text = text.replace(
        "I",
        "1"
    )

    text = text.replace(
        "L",
        "1"
    )

    # Hilangkan whitespace
    text = re.sub(
        r"\s+",
        "",
        text
    )

    # HARUS ada huruf C
    # lalu 1-2 digit

    match = re.search(
        r"C(\d{1,2})",
        text
    )

    if not match:
        return None


    number = int(
        match.group(1)
    )

    code = f"C{number}"


    if code in VALID_CODES:
        return code


    return None


# ============================================================
# SHARPNESS
# ============================================================

def sharpness_score(
    image
):

    if (
        image is None
        or image.size == 0
    ):
        return 0.0


    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


    return float(
        score
    )


# ============================================================
# EASYOCR
# ============================================================

def recognize_wearpack(
    image
):

    if (
        image is None
        or image.size == 0
    ):
        return None


    height, width = image.shape[:2]


    # ========================================================
    # RESIZE
    # ========================================================

    # Kalau terlalu kecil, perbesar
    if width < 320:

        scale = 320 / width

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )


    # Kalau terlalu besar, kecilkan
    elif width > 640:

        scale = 640 / width

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )


    # ========================================================
    # OCR
    # ========================================================
    ocr_start = time.perf_counter()
    results = ocr_reader.readtext(

        image,

        detail=1,

        paragraph=False,

        allowlist="C0123456789",

        decoder="greedy",

        text_threshold=0.5,

        low_text=0.3,

        link_threshold=0.4,

        canvas_size=640,

        mag_ratio=1.0
    )
    ocr_ms = (
        time.perf_counter() - ocr_start
    ) * 1000

    print(f"[PERF] OCR: {ocr_ms:.0f} ms")


    if not results:
        return None


    detected = []


    # ========================================================
    # FILTER RESULT
    # ========================================================

    for (
        bbox,
        text,
        confidence
    ) in results:

        confidence = float(
            confidence
        )


        # Reject OCR confidence rendah
        if (
            confidence
            < OCR_MIN_CONFIDENCE
        ):
            continue


        x_left = min(
            point[0]
            for point in bbox
        )

        y_top = min(
            point[1]
            for point in bbox
        )


        detected.append({

            "x": x_left,

            "y": y_top,

            "text":
                text.upper().strip(),

            "confidence":
                confidence
        })


    if not detected:
        return None


    # ========================================================
    # SORT
    # ========================================================

    # Jika OCR membaca:
    #
    # C
    # 1
    #
    # terpisah, kita gabungkan

    detected.sort(

        key=lambda item: (

            round(
                item["y"] / 50
            ),

            item["x"]
        )
    )


    combined = "".join(

        item["text"]

        for item in detected
    )


    code = normalize_code(
        combined
    )


    # ========================================================
    # DEBUG
    # ========================================================

    debug = [

        (
            item["text"],
            round(
                item["confidence"],
                2
            )
        )

        for item in detected
    ]


    print(
        f"[OCR] "
        f"{debug} | "
        f"combined={combined!r} | "
        f"result={code}"
    )


    return code


# ============================================================
# PERSON -> WEARPACK ROI
# ============================================================

def extract_wearpack_roi(
    frame,
    box
):

    x1, y1, x2, y2 = box


    width = x2 - x1

    height = y2 - y1


    if (
        width <= 0
        or height <= 0
    ):
        return None


    # ========================================================
    # TORSO ROI
    # ========================================================

    # Horizontal:
    # ambil 70% bagian tengah

    rx1 = int(
        x1
        + width * 0.15
    )

    rx2 = int(
        x1
        + width * 0.85
    )


    # Vertical:
    # dada / punggung atas

    ry1 = int(
        y1
        + height * 0.10
    )

    ry2 = int(
        y1
        + height * 0.48
    )


    frame_h, frame_w = (
        frame.shape[:2]
    )


    rx1 = max(
        0,
        min(
            frame_w,
            rx1
        )
    )

    rx2 = max(
        0,
        min(
            frame_w,
            rx2
        )
    )

    ry1 = max(
        0,
        min(
            frame_h,
            ry1
        )
    )

    ry2 = max(
        0,
        min(
            frame_h,
            ry2
        )
    )


    if (
        rx2 <= rx1
        or ry2 <= ry1
    ):
        return None


    crop = frame[
        ry1:ry2,
        rx1:rx2
    ]


    if crop.size == 0:
        return None


    return crop


# ============================================================
# MULTI FRAME VOTING
# ============================================================

def get_stable_code_from_history(
    history
):

    if (
        len(history)
        < MIN_VOTES
    ):
        return None


    counts = Counter(
        history
    )


    code, count = (
        counts.most_common(1)[0]
    )


    vote_ratio = (
        count
        / len(history)
    )


    if (
        count >= MIN_VOTES
        and
        vote_ratio
        >= MIN_VOTE_RATIO
    ):
        return code


    return None


# ============================================================
# CLEANUP OLD TRACK
# ============================================================

def cleanup_tracks(
    now
):

    with state_lock:

        expired = [

            track_id

            for (
                track_id,
                seen_time
            ) in last_seen.items()

            if (
                now
                - seen_time
                > TRACK_TIMEOUT
            )
        ]


        for track_id in expired:

            last_seen.pop(
                track_id,
                None
            )

            last_ocr_time.pop(
                track_id,
                None
            )

            ocr_history.pop(
                track_id,
                None
            )

            stable_code.pop(
                track_id,
                None
            )

            ocr_pending.discard(
                track_id
            )


    return expired


# ============================================================
# RTSP CONNECT
# ============================================================

def open_camera():

    print(
        "[INFO] Connecting RTSP..."
    )


    # Paksa TCP
    os.environ[
        "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    ] = "rtsp_transport;tcp"


    cap = cv2.VideoCapture(

        RTSP_URL,

        cv2.CAP_FFMPEG
    )


    # Hindari buffer besar / delay
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )


    return cap


# ============================================================
# OCR WORKER
# ============================================================

def ocr_worker():

    while not stop_event.is_set():

        try:

            item = ocr_queue.get(
                timeout=0.5
            )

        except queue.Empty:

            continue


        # Shutdown signal
        if item is None:

            ocr_queue.task_done()

            break


        (
            track_id,
            crop,
            crop_sharpness
        ) = item


        event_to_log = None


        try:

            # =================================================
            # RECOGNITION
            # =================================================

            code = recognize_wearpack(
                crop
            )


            with state_lock:

                # Track mungkin sudah hilang
                # ketika OCR selesai
                if (
                    track_id
                    not in last_seen
                ):
                    continue


                # =================================================
                # VOTING
                # =================================================

                if (
                    code
                    and
                    track_id
                    not in stable_code
                ):

                    ocr_history[
                        track_id
                    ].append(
                        code
                    )


                    candidate = (
                        get_stable_code_from_history(
                            ocr_history[
                                track_id
                            ]
                        )
                    )


                    if candidate:

                        # LOCK recognition
                        stable_code[
                            track_id
                        ] = candidate


                        event_to_log = (
                            track_id,
                            candidate
                        )


            # =================================================
            # EVENT
            # =================================================

            if event_to_log:

                log_event(
                    *event_to_log
                )


        except Exception as e:

            print(
                f"[OCR ERROR] "
                f"Track {track_id}: "
                f"{e}"
            )


        finally:

            with state_lock:

                ocr_pending.discard(
                    track_id
                )


            ocr_queue.task_done()


# ============================================================
# AI WORKER
# ============================================================

def ai_worker():

    # ========================================================
    # BEST FRAME
    # ========================================================

    # track_id:
    # (
    #     sharpness,
    #     crop,
    #     timestamp
    # )

    best_crops = {}


    interval = (
        1.0
        / TARGET_AI_FPS
    )


    # Untuk statistik
    ai_counter = 0

    ai_timer = time.time()


    while not stop_event.is_set():

        started = time.time()


        # ====================================================
        # AMBIL FRAME TERBARU
        # ====================================================

        with frame_lock:

            if latest_ai_frame is None:

                frame = None

            else:

                frame = (
                    latest_ai_frame.copy()
                )


        if frame is None:

            time.sleep(
                0.01
            )

            continue


        now = time.time()


        # ====================================================
        # YOLO + BYTETRACK
        # ====================================================

        try:
            yolo_start = time.perf_counter()
            results = model.track(

                frame,

                persist=True,

                tracker="bytetrack.yaml",

                classes=[0],

                conf=PERSON_CONFIDENCE,

                imgsz=YOLO_IMAGE_SIZE,

                device=DEVICE,

                verbose=False
            )
            yolo_ms = (
                time.perf_counter() - yolo_start
            ) * 1000

            print(f"[PERF] YOLO: {yolo_ms:.0f} ms")


        except Exception as e:

            print(
                f"[AI ERROR] {e}"
            )

            time.sleep(
                0.2
            )

            continue


        result = results[0]


        # ====================================================
        # PERSON DETECTED
        # ====================================================

        if (

            result.boxes
            is not None

            and

            result.boxes.id
            is not None
        ):


            boxes = (

                result.boxes.xyxy

                .cpu()

                .numpy()

                .astype(int)
            )


            track_ids = (

                result.boxes.id

                .cpu()

                .numpy()

                .astype(int)
            )


            # =================================================
            # EACH PERSON
            # =================================================

            for (
                box,
                track_id
            ) in zip(
                boxes,
                track_ids
            ):


                x1, y1, x2, y2 = (
                    box.tolist()
                )


                person_height = (
                    y2 - y1
                )


                # =============================================
                # TRACK ACTIVE
                # =============================================

                with state_lock:

                    last_seen[
                        track_id
                    ] = now


                    already_stable = (

                        track_id
                        in stable_code
                    )


                # Sudah diketahui siapa:
                # tidak perlu OCR lagi
                if already_stable:

                    best_crops.pop(
                        track_id,
                        None
                    )

                    continue


                # =============================================
                # ORANG TERLALU JAUH
                # =============================================

                if (
                    person_height
                    < MIN_PERSON_HEIGHT
                ):

                    continue


                # =============================================
                # CROP TORSO
                # =============================================

                crop = (
                    extract_wearpack_roi(

                        frame,

                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )
                )


                if crop is None:

                    continue


                # =============================================
                # SHARPNESS
                # =============================================

                score = sharpness_score(
                    crop
                )


                # =============================================
                # SIMPAN FRAME PALING TAJAM
                # =============================================

                old_best = (
                    best_crops.get(
                        track_id
                    )
                )


                if (
                    old_best is None

                    or

                    score
                    > old_best[0]
                ):

                    best_crops[
                        track_id
                    ] = (

                        score,

                        crop.copy(),

                        now
                    )


                # =============================================
                # CEK APAKAH SUDAH WAKTUNYA OCR
                # =============================================

                with state_lock:

                    due = (

                        now
                        - last_ocr_time[
                            track_id
                        ]

                        >= OCR_INTERVAL
                    )


                    pending = (

                        track_id
                        in ocr_pending
                    )


                    stable = (

                        track_id
                        in stable_code
                    )


                if (
                    not due
                    or pending
                    or stable
                ):

                    continue


                # =============================================
                # AMBIL BEST FRAME
                # =============================================

                best = (
                    best_crops.get(
                        track_id
                    )
                )


                if best is None:

                    continue


                (
                    best_score,
                    best_crop,
                    _
                ) = best


                # Reset window
                best_crops.pop(
                    track_id,
                    None
                )


                with state_lock:

                    last_ocr_time[
                        track_id
                    ] = now


                # =============================================
                # FRAME BLUR -> JANGAN OCR
                # =============================================

                if (
                    best_score
                    < SHARPNESS_THRESHOLD
                ):

                    continue


                # =============================================
                # SEND TO OCR THREAD
                # =============================================

                try:

                    ocr_queue.put_nowait(

                        (
                            track_id,
                            best_crop,
                            best_score
                        )
                    )


                    with state_lock:

                        ocr_pending.add(
                            track_id
                        )


                except queue.Full:

                    # Lebih baik drop crop daripada
                    # membuat sistem delay
                    pass


        # ====================================================
        # CLEAN OLD TRACKS
        # ====================================================

        expired = cleanup_tracks(
            now
        )


        for track_id in expired:

            best_crops.pop(
                track_id,
                None
            )


        # ====================================================
        # CLEAN OLD BEST CROP
        # ====================================================

        for track_id in list(
            best_crops.keys()
        ):

            _, _, timestamp = (
                best_crops[
                    track_id
                ]
            )


            if (
                now - timestamp
                > TRACK_TIMEOUT
            ):

                best_crops.pop(
                    track_id,
                    None
                )


        # ====================================================
        # AI STATISTICS
        # ====================================================

        ai_counter += 1


        if (
            now - ai_timer
            >= 5.0
        ):

            actual_ai_fps = (

                ai_counter

                / (
                    now
                    - ai_timer
                )
            )


            print(

                f"[STATS] "
                f"AI FPS="
                f"{actual_ai_fps:.1f} | "
                f"OCR queue="
                f"{ocr_queue.qsize()}"
            )


            ai_counter = 0

            ai_timer = now


        # ====================================================
        # LIMIT AI FPS
        # ====================================================

        elapsed = (
            time.time()
            - started
        )


        remaining = (
            interval
            - elapsed
        )


        if remaining > 0:

            time.sleep(
                remaining
            )


# ============================================================
# MAIN
# HEADLESS / DATA ONLY
# ============================================================

def main():

    global latest_ai_frame


    init_log()


    # ========================================================
    # START OCR THREAD
    # ========================================================

    ocr_thread = threading.Thread(

        target=ocr_worker,

        name="ocr_worker",

        daemon=True
    )


    ocr_thread.start()


    # ========================================================
    # START AI THREAD
    # ========================================================

    ai_thread = threading.Thread(

        target=ai_worker,

        name="ai_worker",

        daemon=True
    )


    ai_thread.start()


    # ========================================================
    # CONNECT CAMERA
    # ========================================================

    cap = open_camera()


    if not cap.isOpened():

        print(
            "[ERROR] RTSP tidak bisa dibuka"
        )

        stop_event.set()

        return


    print(
        "[INFO] CCTV connected."
    )

    print(
        "[INFO] Headless mode aktif."
    )

    print(
        "[INFO] Tidak ada window video."
    )

    print(
        "[INFO] Tekan Ctrl+C untuk berhenti."
    )

    print(
        f"[INFO] CSV output: "
        f"{LOG_FILE}"
    )


    # ========================================================
    # CAPTURE FPS STAT
    # ========================================================

    capture_counter = 0

    capture_timer = time.time()


    # ========================================================
    # MAIN CAPTURE LOOP
    # ========================================================

    try:

        while not stop_event.is_set():


            ret, frame = (
                cap.read()
            )


            # =================================================
            # RTSP PUTUS
            # =================================================

            if not ret:

                print(
                    "[WARN] Stream putus. "
                    "Reconnecting..."
                )


                cap.release()


                time.sleep(
                    2
                )


                cap = open_camera()


                continue


            # =================================================
            # UPDATE FRAME TERBARU
            # =================================================

            # Tidak ada queue frame.
            #
            # AI selalu mengambil frame PALING BARU.
            #
            # Jadi sistem tidak menumpuk delay.

            with frame_lock:

                latest_ai_frame = frame


            # =================================================
            # CAPTURE FPS
            # =================================================

            capture_counter += 1

            now = time.time()


            if (
                now
                - capture_timer
                >= 5.0
            ):

                capture_fps = (

                    capture_counter

                    / (
                        now
                        - capture_timer
                    )
                )


                print(

                    f"[STATS] "
                    f"Capture FPS="
                    f"{capture_fps:.1f}"
                )


                capture_counter = 0

                capture_timer = now


    # ========================================================
    # CTRL+C
    # ========================================================

    except KeyboardInterrupt:

        print(
            "\n[INFO] Shutdown requested..."
        )


    # ========================================================
    # CLEAN SHUTDOWN
    # ========================================================

    finally:

        stop_event.set()


        # Bangunkan OCR worker
        try:

            ocr_queue.put_nowait(
                None
            )

        except queue.Full:

            pass


        cap.release()


        ai_thread.join(
            timeout=2.0
        )


        ocr_thread.join(
            timeout=2.0
        )


        print(
            "[INFO] Stopped."
        )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
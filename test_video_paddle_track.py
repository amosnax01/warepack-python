import json
import os
import re
import sqlite3
import time
from datetime import datetime
from uuid import uuid4

import cv2
import numpy as np

from paddleocr import TextDetection, TextRecognition

from vision.video_source import VideoSource
from vision.yolox_detector import YOLOXDetector
from vision.bytetrack_tracker import ByteTracker


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "videos/B12.mp4"

# Hanya untuk scoring benchmark.
# TIDAK PERNAH dipakai mengubah hasil OCR.
EXPECTED = None

CAMERA_NAME = "VIDEO_B12"
SESSION_ID = "VIDEO_B12_RAW_OCR"

MODEL_PATH = "models/yolox_s.onnx"

PERSON_CONFIDENCE = 0.20

# OCR per Track ID setiap 100 ms video time
OCR_INTERVAL_SECONDS = 0.10

RUNTIME_DIR = "runtime"

DB_PATH = os.path.join(
    "data",
    "wearpack.db",
)

PROCESSED_DIR = os.path.join(
    "spool",
    "processed",
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True,
)

OCR_MATCH_MIN_CONFIDENCE = 0.65

LIVE_FRAME_PATH = os.path.join(
    RUNTIME_DIR,
    "latest.jpg",
)

LIVE_META_PATH = os.path.join(
    RUNTIME_DIR,
    "detections.json",
)

OUTPUT_PATH = os.path.join(
    RUNTIME_DIR,
    "B12_paddle_live.mp4",
)

REALTIME_PLAYBACK = True
SAVE_OUTPUT_VIDEO = True

JPEG_QUALITY = 82

os.makedirs(
    RUNTIME_DIR,
    exist_ok=True,
)


# ============================================================
# TEXT CLEANUP ONLY
# ============================================================

def normalize_text(text):

    if not text:
        return None

    value = re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper(),
    )

    return value or None


# ============================================================
# PADDLE RESULT
# ============================================================

def result_to_dict(result):

    if isinstance(result, dict):
        return result

    try:

        value = result.json

        if callable(value):
            value = value()

        if isinstance(value, str):
            return json.loads(value)

        if isinstance(value, dict):
            return value

    except Exception:
        pass

    try:
        return dict(result)

    except Exception:
        return {}


def parse_detections(results):

    output = []

    for result in results:

        data = result_to_dict(
            result
        )

        if (
            "res" in data
            and isinstance(
                data["res"],
                dict,
            )
        ):
            data = data["res"]

        polys = data.get(
            "dt_polys",
            []
        )

        scores = data.get(
            "dt_scores",
            []
        )

        # Paddle bisa return numpy array
        if polys is None:
            polys = []

        if scores is None:
            scores = []

        try:
            poly_count = len(polys)
        except TypeError:
            poly_count = 0

        for index in range(
            poly_count
        ):

            poly = np.asarray(
                polys[index],
                dtype=np.float32,
            )

            score = 0.0

            try:
                if index < len(scores):
                    score = float(
                        scores[index]
                    )
            except Exception:
                score = 0.0

            output.append(
                {
                    "poly": poly,
                    "score": score,
                }
            )

    return output

def parse_recognition(result):

    data = result_to_dict(
        result
    )

    if (
        "res" in data
        and isinstance(
            data["res"],
            dict,
        )
    ):
        data = data["res"]

    text = normalize_text(
        data.get(
            "rec_text",
            "",
        )
    )

    try:

        score = float(
            data.get(
                "rec_score",
                0.0,
            )
            or 0.0
        )

    except Exception:
        score = 0.0

    return text, score


# ============================================================
# POLYGON CROP
# ============================================================

def order_points(points):

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    rect = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    total = points.sum(
        axis=1
    )

    rect[0] = points[
        np.argmin(total)
    ]

    rect[2] = points[
        np.argmax(total)
    ]

    diff = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    rect[1] = points[
        np.argmin(diff)
    ]

    rect[3] = points[
        np.argmax(diff)
    ]

    return rect


def crop_polygon(image, poly):

    poly = np.asarray(
        poly,
        dtype=np.float32,
    )

    if len(poly) != 4:

        x, y, w, h = (
            cv2.boundingRect(
                poly.astype(
                    np.int32
                )
            )
        )

        crop = image[
            y:y + h,
            x:x + w,
        ]

        if crop.size == 0:
            return None

        return crop

    rect = order_points(
        poly
    )

    tl, tr, br, bl = rect

    width = max(
        int(
            np.linalg.norm(
                tr - tl
            )
        ),
        int(
            np.linalg.norm(
                br - bl
            )
        ),
    )

    height = max(
        int(
            np.linalg.norm(
                bl - tl
            )
        ),
        int(
            np.linalg.norm(
                br - tr
            )
        ),
    )

    if width < 4 or height < 4:
        return None

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [
                width - 1,
                height - 1,
            ],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = (
        cv2.getPerspectiveTransform(
            rect,
            destination,
        )
    )

    crop = cv2.warpPerspective(
        image,
        matrix,
        (
            width,
            height,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=(
            cv2.BORDER_REPLICATE
        ),
    )

    if crop.size == 0:
        return None

    if (
        crop.shape[0]
        >
        crop.shape[1] * 1.4
    ):

        crop = cv2.rotate(
            crop,
            cv2.ROTATE_90_CLOCKWISE,
        )

    return crop


# ============================================================
# PERSON → OCR SEARCH ROI
# ============================================================

def extract_search_roi(
    frame,
    box,
):

    frame_h, frame_w = (
        frame.shape[:2]
    )

    x1, y1, x2, y2 = [
        int(value)
        for value in box
    ]

    person_w = x2 - x1
    person_h = y2 - y1

    if (
        person_w <= 0
        or person_h <= 0
    ):
        return None, None

    # sedikit padding
    pad_x = int(
        person_w * 0.03
    )

    pad_y = int(
        person_h * 0.03
    )

    px1 = max(
        0,
        x1 - pad_x,
    )

    py1 = max(
        0,
        y1 - pad_y,
    )

    px2 = min(
        frame_w,
        x2 + pad_x,
    )

    py2 = min(
        frame_h,
        y2 + pad_y,
    )

    person = frame[
        py1:py2,
        px1:px2,
    ]

    if person.size == 0:
        return None, None

    h, w = person.shape[:2]

    # ROI sama seperti modular benchmark
    sx1 = int(w * 0.10)
    sx2 = int(w * 0.90)

    sy1 = int(h * 0.12)
    sy2 = int(h * 0.52)

    roi = person[
        sy1:sy2,
        sx1:sx2,
    ]

    if roi.size == 0:
        return None, None

    absolute_roi_box = (
        px1 + sx1,
        py1 + sy1,
        px1 + sx2,
        py1 + sy2,
    )

    return (
        roi,
        absolute_roi_box,
    )


# ============================================================
# OCR
# ============================================================

def run_ocr(
    text_detector,
    text_recognizer,
    roi,
):

    total_started = (
        time.perf_counter()
    )

    # ------------------------
    # DETECTOR
    # ------------------------

    det_started = (
        time.perf_counter()
    )

    det_results = list(
        text_detector.predict(
            input=roi
        )
    )

    det_ms = (
        time.perf_counter()
        - det_started
    ) * 1000.0

    text_boxes = (
        parse_detections(
            det_results
        )
    )

    candidates = []

    rec_ms = 0.0

    # ------------------------
    # RECOGNIZER
    # ------------------------

    for item in text_boxes:

        crop = crop_polygon(
            roi,
            item["poly"],
        )

        if crop is None:
            continue

        crop_h, crop_w = (
            crop.shape[:2]
        )

        if (
            crop_w < 15
            or crop_h < 8
        ):
            continue

        rec_started = (
            time.perf_counter()
        )

        rec_results = list(
            text_recognizer.predict(
                input=crop,
                batch_size=1,
            )
        )

        rec_ms += (
            time.perf_counter()
            - rec_started
        ) * 1000.0

        for rec_result in rec_results:

            text, confidence = (
                parse_recognition(
                    rec_result
                )
            )

            if not text:
                continue

            candidates.append(
                {
                    "text": text,
                    "confidence": float(
                        confidence
                    ),
                    "det_score": float(
                        item["score"]
                    ),
                }
            )

    # IMPORTANT:
    # pilih berdasarkan confidence OCR.
    # BUKAN berdasarkan EXPECTED B12.
    candidates.sort(
        key=lambda item: (
            item["confidence"]
        ),
        reverse=True,
    )

    if candidates:

        best = candidates[0]

        text = best["text"]

        confidence = float(
            best["confidence"]
        )

    else:

        text = None
        confidence = 0.0

    total_ms = (
        time.perf_counter()
        - total_started
    ) * 1000.0

    return {
        "text": text,
        "confidence": confidence,
        "boxes": len(
            text_boxes
        ),
        "candidates": candidates,
        "det_ms": det_ms,
        "rec_ms": rec_ms,
        "total_ms": total_ms,
    }


# ============================================================
# LIVE FE
# ============================================================

def atomic_write_json(
    path,
    payload,
):

    temp_path = (
        path + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            separators=(",", ":"),
        )

    os.replace(
        temp_path,
        path,
    )


def publish_live(
    frame,
    frame_index,
    video_time,
    items,
):

    # latest.jpg
    temp_frame = (
        LIVE_FRAME_PATH
        + ".tmp.jpg"
    )

    ok = cv2.imwrite(
        temp_frame,
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    if ok:

        os.replace(
            temp_frame,
            LIVE_FRAME_PATH,
        )

    # detections.json
    height, width = (
        frame.shape[:2]
    )

    payload = {
        "camera": CAMERA_NAME,

        "session_id": (
            SESSION_ID
        ),

        "timestamp": (
            datetime.now()
            .isoformat(
                timespec="milliseconds"
            )
        ),

        "source": "video",

        "video_path": (
            VIDEO_PATH
        ),

        "frame_index": int(
            frame_index
        ),

        "video_time": float(
            video_time
        ),

        "preview_width": int(
            width
        ),

        "preview_height": int(
            height
        ),

        "capture_zone": {
            "x1": 0,
            "y1": 0,
            "x2": int(width),
            "y2": int(height),
        },

        "detections": items,
    }

    atomic_write_json(
        LIVE_META_PATH,
        payload,
    )


# ============================================================
# DASHBOARD CARD
# ============================================================

def lookup_master(code):

    if not code:
        return None

    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    conn.row_factory = sqlite3.Row

    row = conn.execute(
        """
        SELECT
            code,
            person_name,
            department
        FROM code_master
        WHERE
            code=?
            AND active=1
        """,
        (
            code.upper(),
        ),
    ).fetchone()

    conn.close()

    return (
        dict(row)
        if row
        else None
    )


def extract_person_crop(
    frame,
    box,
):

    frame_h, frame_w = (
        frame.shape[:2]
    )

    x1, y1, x2, y2 = [
        int(value)
        for value in box
    ]

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None

    pad_x = int(w * 0.03)
    pad_y = int(h * 0.03)

    x1 = max(
        0,
        x1 - pad_x,
    )

    y1 = max(
        0,
        y1 - pad_y,
    )

    x2 = min(
        frame_w,
        x2 + pad_x,
    )

    y2 = min(
        frame_h,
        y2 + pad_y,
    )

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return crop


def save_ocr_card(
    frame,
    track,
    ocr_result,
    video_time,
):

    text = ocr_result.get(
        "text"
    )

    if not text:
        return None

    confidence = float(
        ocr_result.get(
            "confidence",
            0.0,
        )
    )

    track_id = int(
        track["track_id"]
    )

    person_crop = (
        extract_person_crop(
            frame,
            track["box"],
        )
    )

    if person_crop is None:
        return None

    # --------------------------------------------------------
    # Exact master lookup ONLY.
    # Tidak mengubah OCR.
    # --------------------------------------------------------

    master = lookup_master(
        text
    )

    if (
        master is not None
        and confidence
        >= OCR_MATCH_MIN_CONFIDENCE
    ):
        status = "recognized"

        matched_code = (
            master["code"]
        )

        person_name = (
            master["person_name"]
        )

    else:

        status = "unrecognized"

        matched_code = None
        person_name = None

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    now = datetime.now()

    timestamp_db = (
        now.isoformat(
            timespec="milliseconds"
        )
    )

    timestamp_file = (
        now.strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    filename = (
        f"{CAMERA_NAME}_"
        f"{SESSION_ID}_"
        f"track{track_id}_"
        f"{timestamp_file}_"
        f"{uuid4().hex[:6]}.jpg"
    )

    final_path = os.path.join(
        PROCESSED_DIR,
        filename,
    )

    ok = cv2.imwrite(
        final_path,
        person_crop,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92,
        ],
    )

    if not ok:

        print(
            "[CARD ERROR] "
            "gagal save image"
        )

        return None

    # --------------------------------------------------------
    # SHARPNESS
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        person_crop,
        cv2.COLOR_BGR2GRAY,
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    # --------------------------------------------------------
    # INSERT CAPTURE
    # --------------------------------------------------------

    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

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
            status,
            ocr_raw_text,
            ocr_text,
            ocr_confidence,
            matched_code,
            person_name
        )
        VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
        """,
        (
            timestamp_db,
            CAMERA_NAME,
            SESSION_ID,
            track_id,
            filename,
            sharpness,
            status,

            # RAW OCR tetap sama.
            text,
            text,

            confidence,
            matched_code,
            person_name,
        ),
    )

    capture_id = int(
        cursor.lastrowid
    )

    conn.commit()
    conn.close()

    print(
        f"[CARD] "
        f"ID={capture_id} | "
        f"t={video_time:.2f}s | "
        f"Track={track_id} | "
        f"OCR={text} | "
        f"status={status} | "
        f"conf={confidence:.3f}"
    )

    return capture_id

# ============================================================
# MAIN
# ============================================================

def main():

    source = VideoSource(
        VIDEO_PATH
    )

    # ------------------------
    # YOLOX
    # ------------------------

    person_detector = (
        YOLOXDetector(
            model_path=MODEL_PATH,
            conf_threshold=(
                PERSON_CONFIDENCE
            ),
        )
    )

    # ------------------------
    # BYTETRACK
    # ------------------------

    tracker = ByteTracker(
        frame_rate=int(
            round(
                source.fps
            )
        )
    )

    # ------------------------
    # PADDLE
    # ------------------------

    print(
        "[INFO] Loading Paddle "
        "modular OCR..."
    )

    text_detector = (
        TextDetection(
            model_name=(
                "PP-OCRv5_mobile_det"
            ),
            device="cpu",
        )
    )

    text_recognizer = (
        TextRecognition(
            model_name=(
                "en_PP-OCRv5_mobile_rec"
            ),
            device="cpu",
        )
    )

    print(
        "[INFO] Ready."
    )

    print(
        "[INFO] FE source:"
    )

    print(
        "       runtime/latest.jpg"
    )

    print(
        "       runtime/detections.json"
    )

    print()

    # latest raw OCR per Track ID
    track_state = {}

    stats = {
        "attempts": 0,
        "text": 0,
        "no_ocr": 0,
    }

    writer = None

    wall_started = (
        time.perf_counter()
    )

    first_video_time = None

    try:

        while True:

            (
                ok,
                frame,
                frame_index,
                video_time,
            ) = source.read()

            if not ok:
                break

            if first_video_time is None:
                first_video_time = (
                    video_time
                )

            # ================================================
            # PERSON DETECT + TRACK
            # ================================================

            detections = (
                person_detector.detect(
                    frame
                )
            )

            tracks = tracker.update(
                detections
            )

            display = frame.copy()

            live_items = []

            # ================================================
            # EACH TRACK
            # ================================================

            for track in tracks:

                track_id = int(
                    track["track_id"]
                )

                state = (
                    track_state.setdefault(
                        track_id,
                        {
                            "last_ocr_video_time":
                                -999.0,

                            "text":
                                None,

                            "confidence":
                                0.0,

                            "boxes":
                                0,

                            "total_ms":
                                0.0,

                            "last_card_text": None,
                        },
                    )
                )

                roi, roi_box = (
                    extract_search_roi(
                        frame,
                        track["box"],
                    )
                )

                # ============================================
                # OCR EACH ~100ms VIDEO TIME
                # ============================================

                should_ocr = (
                    roi is not None
                    and
                    video_time
                    - state[
                        "last_ocr_video_time"
                    ]
                    >=
                    OCR_INTERVAL_SECONDS
                )

                if should_ocr:

                    result = run_ocr(
                        text_detector,
                        text_recognizer,
                        roi,
                    )

                    state[
                        "last_ocr_video_time"
                    ] = video_time

                    # RAW RESULT
                    state["text"] = (
                        result["text"]
                    )

                    state[
                        "confidence"
                    ] = (
                        result[
                            "confidence"
                        ]
                    )

                    state["boxes"] = (
                        result["boxes"]
                    )

                    state[
                        "total_ms"
                    ] = (
                        result[
                            "total_ms"
                        ]
                    )

                    # ============================================
                    # DASHBOARD CARD
                    # ============================================

                    if (
                        result["text"] is not None
                        and
                        result["text"]
                        != state["last_card_text"]
                    ):

                        capture_id = save_ocr_card(
                            frame,
                            track,
                            result,
                            video_time,
                        )

                        if capture_id is not None:

                            state[
                                "last_card_text"
                            ] = result["text"]

                    # ----------------------------------------
                    # benchmark stats only
                    # ----------------------------------------

                    stats["attempts"] += 1

                    if result["text"] is None:
                        stats["no_ocr"] += 1
                    else:
                        stats["text"] += 1

                    # ----------------------------------------
                    # RAW LOG
                    # ----------------------------------------

                    print(
                        f"[OCR] "
                        f"t="
                        f"{video_time:5.2f}s | "
                        f"frame="
                        f"{frame_index:4d} | "
                        f"track="
                        f"{track_id} | "
                        f"text="
                        f"{result['text']!r} | "
                        f"conf="
                        f"{result['confidence']:.3f} | "
                        f"boxes="
                        f"{result['boxes']} | "
                        f"det="
                        f"{result['det_ms']:.1f}ms | "
                        f"rec="
                        f"{result['rec_ms']:.1f}ms | "
                        f"total="
                        f"{result['total_ms']:.1f}ms"
                    )

                # ============================================
                # DRAW PERSON
                # ============================================

                x1, y1, x2, y2 = [
                    int(value)
                    for value
                    in track["box"]
                ]

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # OCR search ROI
                if roi_box is not None:

                    (
                        rx1,
                        ry1,
                        rx2,
                        ry2,
                    ) = roi_box

                    cv2.rectangle(
                        display,
                        (
                            rx1,
                            ry1,
                        ),
                        (
                            rx2,
                            ry2,
                        ),
                        (
                            255,
                            255,
                            0,
                        ),
                        1,
                    )

                # ============================================
                # RAW OCR LABEL
                # ============================================

                if state["text"]:

                    label = (
                        f"Track {track_id} | "
                        f"OCR: "
                        f"{state['text']} "
                        f"{state['confidence']:.2f}"
                    )

                else:

                    label = (
                        f"Track {track_id} | "
                        f"NO_OCR"
                    )

                cv2.putText(
                    display,
                    label,
                    (
                        x1,
                        max(
                            30,
                            y1 - 10,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                # ============================================
                # FE METADATA
                # ============================================

                live_items.append(
                    {
                        "track_id":
                            track_id,

                        "confidence":
                            float(
                                track[
                                    "confidence"
                                ]
                            ),

                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,

                        "in_capture_zone":
                            True,

                        "ocr_status": (
                            "text"
                            if state["text"]
                            else "no_ocr"
                        ),

                        "ocr_text":
                            state["text"],

                        "ocr_confidence":
                            float(
                                state[
                                    "confidence"
                                ]
                            ),

                        "ocr_boxes":
                            int(
                                state[
                                    "boxes"
                                ]
                            ),

                        "ocr_total_ms":
                            float(
                                state[
                                    "total_ms"
                                ]
                            ),
                    }
                )

            # ================================================
            # VIDEO INFO
            # ================================================

            cv2.putText(
                display,
                (
                    f"VIDEO B12 | "
                    f"{video_time:.2f}s | "
                    f"frame "
                    f"{frame_index}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # ================================================
            # PUBLISH TO CURRENT FE
            # ================================================

            publish_live(
                display,
                frame_index,
                video_time,
                live_items,
            )

            # ================================================
            # SAVE TEST VIDEO
            # ================================================

            if SAVE_OUTPUT_VIDEO:

                if writer is None:

                    h, w = (
                        display.shape[:2]
                    )

                    writer = (
                        cv2.VideoWriter(
                            OUTPUT_PATH,

                            cv2.VideoWriter_fourcc(
                                *"mp4v"
                            ),

                            source.fps,

                            (
                                w,
                                h,
                            ),
                        )
                    )

                writer.write(
                    display
                )

            # ================================================
            # REALTIME PLAYBACK
            # ================================================

            if REALTIME_PLAYBACK:

                relative_time = (
                    video_time
                    - first_video_time
                )

                target_wall = (
                    wall_started
                    + relative_time
                )

                sleep_time = (
                    target_wall
                    - time.perf_counter()
                )

                # kalau AI lebih lambat,
                # jangan tambah delay lagi
                if sleep_time > 0:

                    time.sleep(
                        sleep_time
                    )

    finally:

        source.release()

        if writer is not None:
            writer.release()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "RAW VIDEO OCR RESULT"
    )
    print("=" * 80)

    print(
        f"Attempts : "
        f"{stats['attempts']}"
    )

    print(
    f"Attempts : {stats['attempts']}"
    )

    print(
        f"Text     : {stats['text']}"
    )

    print(
        f"No OCR   : {stats['no_ocr']}"
    )

    if stats["attempts"]:

        text_rate = (
            stats["text"]
            / stats["attempts"]
            * 100.0
        )

        print(
            f"Text rate: {text_rate:.2f}%"
        )

    print()
    print(
        f"Saved video: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
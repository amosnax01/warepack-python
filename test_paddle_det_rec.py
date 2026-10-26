import os
import re
import glob
import time
import json

import cv2
import numpy as np

from paddleocr import TextDetection
from paddleocr import TextRecognition


# ============================================================
# CONFIG
# ============================================================

CAPTURE_DIR = "runtime/capture_test/B12/track_0"
EXPECTED = "B12"

DEBUG_DIR = "runtime/paddle_det_rec/B12"

os.makedirs(DEBUG_DIR, exist_ok=True)


# ============================================================
# TEXT CLEANUP ONLY
# ============================================================

def normalize_text(text):
    """
    Cleanup only.

    Tidak ada:
    I -> 1
    8 -> B
    B124 -> B12
    substring correction
    """

    if not text:
        return None

    value = re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper(),
    )

    return value or None


# ============================================================
# PERSON -> OCR SEARCH ROI
# ============================================================

def extract_search_roi(image):
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]

    # Area torso yang cukup luas.
    # Text detector yang akan mencari B12 di dalam sini.
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)

    y1 = int(h * 0.12)
    y2 = int(h * 0.52)

    roi = image[
        y1:y2,
        x1:x2,
    ]

    if roi.size == 0:
        return None

    return roi


# ============================================================
# RESULT PARSER
# ============================================================

def result_to_dict(result):

    if isinstance(result, dict):
        return result

    try:
        data = result.json

        if callable(data):
            data = data()

        if isinstance(data, str):
            return json.loads(data)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    try:
        return dict(result)
    except Exception:
        return {}


def extract_detection(result):

    data = result_to_dict(result)

    if (
        "res" in data
        and isinstance(data["res"], dict)
    ):
        data = data["res"]

    polys = data.get(
        "dt_polys",
        [],
    )

    scores = data.get(
        "dt_scores",
        [],
    )

    if polys is None:
        polys = []

    if scores is None:
        scores = []

    outputs = []

    for index, poly in enumerate(polys):

        poly = np.asarray(
            poly,
            dtype=np.float32,
        )

        score = 0.0

        if index < len(scores):
            try:
                score = float(scores[index])
            except Exception:
                pass

        outputs.append(
            {
                "poly": poly,
                "score": score,
            }
        )

    return outputs


def extract_recognition(result):

    data = result_to_dict(result)

    if (
        "res" in data
        and isinstance(data["res"], dict)
    ):
        data = data["res"]

    text = normalize_text(
        data.get(
            "rec_text",
            "",
        )
    )

    try:
        confidence = float(
            data.get(
                "rec_score",
                0.0,
            )
            or 0.0
        )
    except Exception:
        confidence = 0.0

    return text, confidence


# ============================================================
# POLYGON -> STRAIGHT TEXT CROP
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

    s = points.sum(axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def crop_text_polygon(image, poly):

    if image is None or image.size == 0:
        return None

    poly = np.asarray(
        poly,
        dtype=np.float32,
    )

    # Normal Paddle detector: quadrilateral.
    if len(poly) != 4:

        x, y, w, h = cv2.boundingRect(
            poly.astype(np.int32)
        )

        crop = image[
            y:y + h,
            x:x + w,
        ]

        return (
            crop
            if crop.size
            else None
        )

    rect = order_points(poly)

    tl, tr, br, bl = rect

    width_top = np.linalg.norm(
        tr - tl
    )

    width_bottom = np.linalg.norm(
        br - bl
    )

    height_left = np.linalg.norm(
        bl - tl
    )

    height_right = np.linalg.norm(
        br - tr
    )

    width = max(
        int(width_top),
        int(width_bottom),
    )

    height = max(
        int(height_left),
        int(height_right),
    )

    if width < 4 or height < 4:
        return None

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination,
    )

    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if crop.size == 0:
        return None

    # Kalau detector menghasilkan vertical crop,
    # putar agar recognizer menerima horizontal text.
    if crop.shape[0] > crop.shape[1] * 1.4:

        crop = cv2.rotate(
            crop,
            cv2.ROTATE_90_CLOCKWISE,
        )

    return crop


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("[INFO] Loading Paddle text detector...")

detector = TextDetection(
    model_name="PP-OCRv5_mobile_det",
    device="cpu",

    # Lebih sensitif terhadap tulisan samar
    thresh=0.20,
    box_thresh=0.35,

    # Jangan terlalu banyak background masuk ke crop
    unclip_ratio=1.5,
)

print("[INFO] Loading Paddle text recognizer...")

recognizer = TextRecognition(
    model_name="en_PP-OCRv5_mobile_rec",
    device="cpu",
)

print("[INFO] Models ready.")
print()


# ============================================================
# LOAD CAPTURES
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            CAPTURE_DIR,
            "*.jpg",
        )
    )
)

if not files:
    raise RuntimeError(
        f"No captures found: {CAPTURE_DIR}"
    )


# ============================================================
# WARMUP
# ============================================================

first_image = cv2.imread(files[0])
first_roi = extract_search_roi(
    first_image
)

print("[INFO] Warmup...")

for _ in range(2):
    list(
        detector.predict(
            input=first_roi
        )
    )

print("[INFO] Warmup complete.")
print()


# ============================================================
# STATS
# ============================================================

correct = 0
wrong = 0
no_ocr = 0

det_times = []
rec_times = []
total_times = []

total_text_boxes = 0


# ============================================================
# TEST
# ============================================================

print("=" * 110)

print(
    f"PADDLE MODULAR DET+REC | "
    f"EXPECTED={EXPECTED} | "
    f"CAPTURES={len(files)}"
)

print("=" * 110)


for file_index, path in enumerate(files):

    filename = os.path.basename(
        path
    )

    image = cv2.imread(path)

    if image is None:
        continue

    roi = extract_search_roi(
        image
    )

    if roi is None:
        continue

    frame_debug_dir = os.path.join(
        DEBUG_DIR,
        f"{file_index:03d}",
    )

    os.makedirs(
        frame_debug_dir,
        exist_ok=True,
    )

    cv2.imwrite(
        os.path.join(
            frame_debug_dir,
            "search_roi.jpg",
        ),
        roi,
    )

    # --------------------------------------------------------
    # TEXT DETECTION
    # --------------------------------------------------------

    total_started = time.perf_counter()

    det_started = time.perf_counter()

    det_results = list(
        detector.predict(
            input=roi
        )
    )

    det_ms = (
        time.perf_counter()
        - det_started
    ) * 1000.0

    det_times.append(det_ms)


    detections = []

    for result in det_results:

        detections.extend(
            extract_detection(
                result
            )
        )


    # Higher score first
    detections.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    total_text_boxes += len(
        detections
    )


    # --------------------------------------------------------
    # TEXT RECOGNITION
    # --------------------------------------------------------

    candidates = []

    frame_rec_ms = 0.0


    for box_index, detection in enumerate(
        detections
    ):

        crop = crop_text_polygon(
            roi,
            detection["poly"],
        )

        if crop is None:
            continue

        # Skip absurd tiny regions
        ch, cw = crop.shape[:2]

        if cw < 15 or ch < 8:
            continue

        crop_path = os.path.join(
            frame_debug_dir,
            (
                f"box_{box_index:02d}_"
                f"det{detection['score']:.2f}.jpg"
            ),
        )

        cv2.imwrite(
            crop_path,
            crop,
        )

        rec_started = time.perf_counter()

        rec_results = list(
            recognizer.predict(
                input=crop,
                batch_size=1,
            )
        )

        rec_ms = (
            time.perf_counter()
            - rec_started
        ) * 1000.0

        frame_rec_ms += rec_ms


        for rec_result in rec_results:

            text, confidence = (
                extract_recognition(
                    rec_result
                )
            )

            if not text:
                continue

            candidates.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "det_score": detection["score"],
                }
            )


    rec_times.append(
        frame_rec_ms
    )


    total_ms = (
        time.perf_counter()
        - total_started
    ) * 1000.0

    total_times.append(
        total_ms
    )


    # --------------------------------------------------------
    # STRICT EVALUATION
    # --------------------------------------------------------

    exact = [
        item
        for item in candidates
        if item["text"] == EXPECTED
    ]

    if exact:

        status = "CORRECT"
        correct += 1

    elif candidates:

        status = "WRONG"
        wrong += 1

    else:

        status = "NO_OCR"
        no_ocr += 1


    candidate_string = ", ".join(
        (
            f"{item['text']}"
            f"(rec={item['confidence']:.2f},"
            f"det={item['det_score']:.2f})"
        )
        for item in candidates
    )


    print(
        f"[{file_index:02d}] "
        f"{status:8s} | "
        f"boxes={len(detections):2d} | "
        f"det={det_ms:6.1f} ms | "
        f"rec={frame_rec_ms:6.1f} ms | "
        f"total={total_ms:6.1f} ms | "
        f"[{candidate_string}]"
    )


# ============================================================
# SUMMARY
# ============================================================

total = (
    correct
    + wrong
    + no_ocr
)

accuracy = (
    correct / total * 100.0
    if total
    else 0.0
)


def average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def median(values):

    if not values:
        return 0.0

    values = sorted(values)

    return values[
        len(values) // 2
    ]


print()
print("=" * 110)
print("RESULT")
print("=" * 110)

print()
print(
    "PADDLEOCR MODULAR DET + REC"
)

print(
    "Detector   : PP-OCRv5_mobile_det"
)

print(
    "Recognizer : en_PP-OCRv5_mobile_rec"
)

print()

print(
    f"Correct    : {correct}"
)

print(
    f"Wrong      : {wrong}"
)

print(
    f"No OCR     : {no_ocr}"
)

print(
    f"Accuracy   : {accuracy:.2f}%"
)

print()

print(
    f"Avg boxes  : "
    f"{total_text_boxes / total:.2f}"
    if total
    else "Avg boxes  : 0"
)

print()

print(
    f"Avg DET    : "
    f"{average(det_times):.1f} ms"
)

print(
    f"Avg REC    : "
    f"{average(rec_times):.1f} ms"
)

print(
    f"Avg TOTAL  : "
    f"{average(total_times):.1f} ms"
)

print(
    f"Median total: "
    f"{median(total_times):.1f} ms"
)

print()

print("-" * 110)

print(
    "PREVIOUS"
)

print(
    "Paddle full COLOR       : 29.27% | ~1119 ms"
)

print(
    "Paddle recognition-only :  7.32% | ~51 ms"
)

print(
    "EasyOCR COLOR           : 14.63%"
)

print(
    "EasyOCR GRAY            : 17.07%"
)

print("-" * 110)

print()
print(
    f"Debug crops: {DEBUG_DIR}"
)
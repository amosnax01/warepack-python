import os
import re
import glob
import time

import cv2

from paddleocr import TextRecognition


# ============================================================
# CONFIG
# ============================================================

CAPTURE_DIR = "runtime/capture_test/B12/track_0"

EXPECTED = "B12"

DEBUG_DIR = "runtime/paddle_rec_only/B12"

os.makedirs(
    DEBUG_DIR,
    exist_ok=True,
)


# ============================================================
# TEXT CLEANUP
# ============================================================

def normalize_text(text):
    """
    Cleanup ONLY.

    Tidak ada:
    I -> 1
    8 -> B
    substring matching
    semantic correction
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
# CODE ROI
# ============================================================

def extract_code_roi(image):
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]

    # ROI yang sama dengan test terakhir.
    #
    # Recognition-only tidak punya text detector,
    # jadi area harus relatif fokus ke lokasi kode.

    x1 = int(w * 0.18)
    x2 = int(w * 0.82)

    y1 = int(h * 0.17)
    y2 = int(h * 0.43)

    roi = image[
        y1:y2,
        x1:x2,
    ]

    if roi.size == 0:
        return None

    return roi


# ============================================================
# PARSE PADDLE RESULT
# ============================================================

def extract_result(result):
    """
    Return:
        text, confidence
    """

    try:
        data = result.json

        if callable(data):
            data = data()

    except Exception:
        data = None

    if isinstance(data, dict):

        if (
            "res" in data
            and isinstance(data["res"], dict)
        ):
            data = data["res"]

        text = data.get(
            "rec_text",
            "",
        )

        confidence = data.get(
            "rec_score",
            0.0,
        )

        return (
            normalize_text(text),
            float(confidence or 0.0),
        )

    # fallback dict-like
    try:
        data = dict(result)

        if (
            "res" in data
            and isinstance(data["res"], dict)
        ):
            data = data["res"]

        return (
            normalize_text(
                data.get("rec_text", "")
            ),
            float(
                data.get("rec_score", 0.0)
                or 0.0
            ),
        )

    except Exception:
        return None, 0.0


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("[INFO] Loading recognition-only model...")

recognizer = TextRecognition(
    model_name="en_PP-OCRv5_mobile_rec",
    device="cpu",
)

print("[INFO] Recognition model ready.")
print()


# ============================================================
# LOAD FILES
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
# PREPARE DATASET
# ============================================================

samples = []

for path in files:

    image = cv2.imread(path)

    if image is None:
        continue

    roi = extract_code_roi(
        image
    )

    if roi is None:
        continue

    filename = os.path.basename(
        path
    )

    debug_path = os.path.join(
        DEBUG_DIR,
        filename,
    )

    cv2.imwrite(
        debug_path,
        roi,
    )

    samples.append(
        (
            filename,
            roi,
        )
    )


if not samples:
    raise RuntimeError(
        "No valid OCR samples."
    )


# ============================================================
# WARMUP
# ============================================================

print("[INFO] Warmup...")

for _ in range(3):

    _ = recognizer.predict(
        input=samples[0][1],
        batch_size=1,
    )

print("[INFO] Warmup complete.")
print()


# ============================================================
# BENCHMARK
# ============================================================

correct = 0
wrong = 0
no_ocr = 0

times = []


print("=" * 100)

print(
    f"PADDLE RECOGNITION-ONLY | "
    f"MODEL=en_PP-OCRv5_mobile_rec | "
    f"EXPECTED={EXPECTED} | "
    f"CAPTURES={len(samples)}"
)

print("=" * 100)


for index, (
    filename,
    roi,
) in enumerate(samples):

    started = time.perf_counter()

    results = recognizer.predict(
        input=roi,
        batch_size=1,
    )

    elapsed_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    times.append(
        elapsed_ms
    )

    text = None
    confidence = 0.0

    for result in results:

        text, confidence = extract_result(
            result
        )

        # satu input = satu recognition result
        break

    if not text:

        status = "NO_OCR"
        no_ocr += 1

    elif text == EXPECTED:

        status = "CORRECT"
        correct += 1

    else:

        status = "WRONG"
        wrong += 1

    print(
        f"[{index:02d}] "
        f"{status:8s} | "
        f"{elapsed_ms:7.1f} ms | "
        f"text={repr(text):10s} | "
        f"conf={confidence:.3f} | "
        f"{filename}"
    )


# ============================================================
# RESULTS
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

avg_ms = (
    sum(times) / len(times)
    if times
    else 0.0
)

sorted_times = sorted(
    times
)

median_ms = (
    sorted_times[
        len(sorted_times) // 2
    ]
    if sorted_times
    else 0.0
)

min_ms = (
    min(times)
    if times
    else 0.0
)

max_ms = (
    max(times)
    if times
    else 0.0
)


print()
print("=" * 100)
print("RESULT")
print("=" * 100)

print()
print(
    "PADDLEOCR RECOGNITION-ONLY"
)

print(
    "Model    : en_PP-OCRv5_mobile_rec"
)

print(
    f"Correct  : {correct}"
)

print(
    f"Wrong    : {wrong}"
)

print(
    f"No OCR   : {no_ocr}"
)

print(
    f"Accuracy : {accuracy:.2f}%"
)

print()
print(
    f"Avg time : {avg_ms:.1f} ms/image"
)

print(
    f"Median   : {median_ms:.1f} ms/image"
)

print(
    f"Min      : {min_ms:.1f} ms"
)

print(
    f"Max      : {max_ms:.1f} ms"
)

print()
print("-" * 100)

print(
    "PREVIOUS RESULTS"
)

print(
    "PaddleOCR full COLOR : 29.27% | ~1119 ms"
)

print(
    "EasyOCR COLOR        : 14.63%"
)

print(
    "EasyOCR GRAY         : 17.07%"
)

print("-" * 100)

print()
print(
    f"ROI images: {DEBUG_DIR}"
)
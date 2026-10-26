import os
import re
import glob
import time
import json

import cv2

from paddleocr import PaddleOCR


# ============================================================
# CONFIG
# ============================================================

CAPTURE_DIR = "runtime/capture_test/B12/track_0"

EXPECTED = "B12"

DEBUG_DIR = "runtime/paddle_ocr_test/B12"

os.makedirs(
    DEBUG_DIR,
    exist_ok=True,
)


# ============================================================
# TEXT CLEANUP
# ============================================================

def normalize_text(text):
    """
    Cleanup saja.

    TIDAK ADA:
    I -> 1
    8 -> B
    0 -> O
    substring matching
    semantic correction

    Jadi:
    B12  = B12
    IB12 = IB12
    812  = 812
    B124 = B124
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
# ROI
# ============================================================

def extract_ocr_roi(image):
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]

    # ROI sempit yang terakhir kita test
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
# PREPROCESS VARIANTS
# ============================================================

def prepare_color(image):
    h, w = image.shape[:2]

    target_width = 700

    if w < target_width:
        scale = target_width / float(w)

        image = cv2.resize(
            image,
            (
                target_width,
                int(h * scale),
            ),
            interpolation=cv2.INTER_CUBIC,
        )

    return image


def prepare_gray(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    h, w = gray.shape[:2]

    target_width = 700

    if w < target_width:
        scale = target_width / float(w)

        gray = cv2.resize(
            gray,
            (
                target_width,
                int(h * scale),
            ),
            interpolation=cv2.INTER_CUBIC,
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0,
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        blur,
        -0.5,
        0,
    )

    return sharpened


# ============================================================
# PADDLE RESULT PARSER
# ============================================================

def result_to_dict(result):
    """
    PaddleOCR 3.x Result object bisa berbeda
    sedikit antar versi.

    Fungsi ini mencoba mengambil dict secara aman.
    """

    if isinstance(result, dict):
        return result

    # Banyak Paddle Result object bersifat dict-like
    try:
        return dict(result)
    except Exception:
        pass

    # Coba properti JSON
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

    return {}


def extract_paddle_outputs(results):
    """
    Return:

    [
        {
            "text": "B12",
            "confidence": 0.97
        }
    ]
    """

    outputs = []

    for result in results:

        data = result_to_dict(result)

        # Beberapa versi menyimpan hasil langsung,
        # beberapa di dalam key "res".
        if (
            "res" in data
            and isinstance(data["res"], dict)
        ):
            data = data["res"]

        texts = data.get(
            "rec_texts",
            [],
        )

        scores = data.get(
            "rec_scores",
            [],
        )

        if texts is None:
            texts = []

        if scores is None:
            scores = []

        for index, text in enumerate(texts):

            normalized = normalize_text(
                text
            )

            if not normalized:
                continue

            confidence = 0.0

            if index < len(scores):
                try:
                    confidence = float(
                        scores[index]
                    )
                except Exception:
                    confidence = 0.0

            outputs.append(
                {
                    "text": normalized,
                    "confidence": confidence,
                }
            )

    return outputs


# ============================================================
# LOAD PADDLE OCR
# ============================================================

print()
print("[INFO] Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",

    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,

    device="cpu",
)

print("[INFO] PaddleOCR ready.")
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


stats = {
    "color": {
        "correct": 0,
        "wrong": 0,
        "no_ocr": 0,
        "times": [],
    },

    "gray": {
        "correct": 0,
        "wrong": 0,
        "no_ocr": 0,
        "times": [],
    },
}


print("=" * 90)

print(
    f"PADDLE OCR BENCHMARK | "
    f"EXPECTED={EXPECTED} | "
    f"CAPTURES={len(files)}"
)

print("=" * 90)


# ============================================================
# BENCHMARK
# ============================================================

for file_index, path in enumerate(files):

    filename = os.path.basename(
        path
    )

    image = cv2.imread(
        path
    )

    if image is None:
        print(
            f"[ERROR] Cannot read: {filename}"
        )
        continue

    roi = extract_ocr_roi(
        image
    )

    if roi is None:
        print(
            f"[ERROR] Invalid ROI: {filename}"
        )
        continue

    variants = {
        "color": prepare_color(
            roi
        ),

        "gray": prepare_gray(
            roi
        ),
    }

    print()
    print(
        f"[{file_index:02d}] {filename}"
    )

    for variant_name, variant in variants.items():

        debug_path = os.path.join(
            DEBUG_DIR,
            f"{variant_name}_{filename}",
        )

        cv2.imwrite(
            debug_path,
            variant,
        )

        started = time.perf_counter()

        try:
            paddle_results = ocr.predict(
                debug_path
            )

            outputs = extract_paddle_outputs(
                paddle_results
            )

        except Exception as exc:
            print(
                f"  {variant_name.upper():5s} | "
                f"ERROR | {exc}"
            )

            stats[
                variant_name
            ]["no_ocr"] += 1

            continue

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        stats[
            variant_name
        ]["times"].append(
            elapsed_ms
        )

        exact_matches = [
            item
            for item in outputs
            if item["text"] == EXPECTED
        ]

        if exact_matches:

            status = "CORRECT"

            stats[
                variant_name
            ]["correct"] += 1

        elif outputs:

            status = "WRONG"

            stats[
                variant_name
            ]["wrong"] += 1

        else:

            status = "NO_OCR"

            stats[
                variant_name
            ]["no_ocr"] += 1

        output_string = ", ".join(
            (
                f"{item['text']}"
                f"({item['confidence']:.2f})"
            )
            for item in outputs
        )

        print(
            f"  {variant_name.upper():5s} | "
            f"{status:8s} | "
            f"{elapsed_ms:7.0f} ms | "
            f"[{output_string}]"
        )


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 90)
print("RESULT")
print("=" * 90)


for variant_name, data in stats.items():

    total = (
        data["correct"]
        + data["wrong"]
        + data["no_ocr"]
    )

    if total > 0:
        accuracy = (
            data["correct"]
            / total
            * 100.0
        )
    else:
        accuracy = 0.0

    if data["times"]:
        average_ms = (
            sum(data["times"])
            / len(data["times"])
        )
    else:
        average_ms = 0.0

    print()
    print(
        variant_name.upper()
    )

    print(
        f"Correct : {data['correct']}"
    )

    print(
        f"Wrong   : {data['wrong']}"
    )

    print(
        f"No OCR  : {data['no_ocr']}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print(
        f"Avg time: {average_ms:.0f} ms/image"
    )


print()
print("-" * 90)

print(
    "EASYOCR BASELINE"
)

print(
    "COLOR : 14.63%"
)

print(
    "GRAY  : 17.07%"
)

print("-" * 90)

print()
print(
    f"Debug images: {DEBUG_DIR}"
)
import os
import glob
import cv2

from recognition_service import (
    extract_ocr_roi_from_person_crop,
    prepare_ocr_image,
    recognize_generic,
    normalize_text,
)


CAPTURE_DIR = "runtime/capture_test/B12/track_0"
DEBUG_DIR = "runtime/ocr_test/B12"

EXPECTED = "B12"

os.makedirs(DEBUG_DIR, exist_ok=True)


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


total = 0
correct = 0
wrong = 0
no_ocr = 0


print()
print("=" * 80)
print(f"OCR TEST : {EXPECTED}")
print(f"CAPTURES : {len(files)}")
print("=" * 80)


for path in files:
    filename = os.path.basename(path)

    image = cv2.imread(path)

    if image is None:
        print(f"[ERROR] cannot read {filename}")
        continue

    roi = extract_ocr_roi_from_person_crop(image)

    if roi is None or roi.size == 0:
        print(f"[NO ROI] {filename}")
        continue

    # Simpan ROI original
    roi_path = os.path.join(
        DEBUG_DIR,
        f"roi_{filename}",
    )

    cv2.imwrite(
        roi_path,
        roi,
    )

    # Simpan gambar yang benar-benar diberikan
    # ke OCR setelah preprocessing
    processed = prepare_ocr_image(roi)

    processed_path = os.path.join(
        DEBUG_DIR,
        f"ocr_{filename}",
    )

    cv2.imwrite(
        processed_path,
        processed,
    )

    raw_text, candidates = recognize_generic(roi)

    normalized = normalize_text(raw_text)

    total += 1

    if normalized is None:
        status = "NO_OCR"
        no_ocr += 1

    elif normalized == EXPECTED:
        status = "CORRECT"
        correct += 1

    else:
        status = "WRONG"
        wrong += 1

    print(
        f"{status:8s} | "
        f"{filename:45s} | "
        f"raw={repr(raw_text):12s} | "
        f"normalized={repr(normalized)}"
    )


print()
print("=" * 80)
print("RESULT")
print("=" * 80)

print(f"Total   : {total}")
print(f"Correct : {correct}")
print(f"Wrong   : {wrong}")
print(f"No OCR  : {no_ocr}")

if total > 0:
    accuracy = (
        correct / total
    ) * 100.0

    print(
        f"Exact B12 accuracy : "
        f"{accuracy:.2f}%"
    )

print()
print(
    f"Debug images: {DEBUG_DIR}"
)
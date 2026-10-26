import os
import glob
import cv2

from recognition_service import (
    extract_ocr_roi_from_person_crop,
    prepare_ocr_variants,
    normalize_text,
    reader,
)


CAPTURE_DIR = (
    "runtime/capture_test/B12/track_0"
)

EXPECTED = "B12"

files = sorted(
    glob.glob(
        os.path.join(
            CAPTURE_DIR,
            "*.jpg",
        )
    )
)


stats = {
    "color": {
        "correct": 0,
        "wrong": 0,
        "none": 0,
    },
    "gray": {
        "correct": 0,
        "wrong": 0,
        "none": 0,
    },
}


for path in files:

    image = cv2.imread(path)

    if image is None:
        continue

    roi = extract_ocr_roi_from_person_crop(
        image
    )

    if roi is None:
        continue

    variants = prepare_ocr_variants(
        roi
    )

    print()
    print(
        os.path.basename(path)
    )

    for name, variant in variants:

        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
            ),
            decoder="greedy",
            text_threshold=0.45,
            low_text=0.25,
            link_threshold=0.4,
            canvas_size=768,
            mag_ratio=1.0,
        )

        outputs = []

        for _, text, confidence in results:

            normalized = normalize_text(
                text
            )

            if normalized:
                outputs.append(
                    (
                        normalized,
                        float(confidence),
                    )
                )

        exact = [
            item
            for item in outputs
            if item[0] == EXPECTED
        ]

        if exact:

            stats[name]["correct"] += 1

            status = "CORRECT"

        elif outputs:

            stats[name]["wrong"] += 1

            status = "WRONG"

        else:

            stats[name]["none"] += 1

            status = "NO_OCR"

        print(
            f"  {name:5s} "
            f"{status:8s} "
            f"{outputs}"
        )


print()
print("=" * 60)
print("RESULT")
print("=" * 60)

for name, data in stats.items():

    total = sum(
        data.values()
    )

    accuracy = (
        data["correct"]
        / total
        * 100
        if total
        else 0
    )

    print()
    print(name.upper())
    print(
        f"Correct : {data['correct']}"
    )
    print(
        f"Wrong   : {data['wrong']}"
    )
    print(
        f"No OCR  : {data['none']}"
    )
    print(
        f"Accuracy: {accuracy:.2f}%"
    )
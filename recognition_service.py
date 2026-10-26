import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import re
import shutil
import sqlite3
import time

import cv2
import easyocr
import torch

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "wearpack.db")
PENDING_DIR = os.path.join(BASE_DIR, "spool", "pending")
PROCESSED_DIR = os.path.join(BASE_DIR, "spool", "processed")
DEBUG_OCR_DIR = os.path.join(
    BASE_DIR,
    "debug_ocr"
)

os.makedirs(
    DEBUG_OCR_DIR,
    exist_ok=True
)

os.makedirs(PENDING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

POLL_INTERVAL = float(os.getenv("OCR_POLL_INTERVAL", "0.2"))
OCR_DISPLAY_MIN_CONFIDENCE = float(os.getenv("OCR_DISPLAY_MIN_CONFIDENCE", "0.30"))
OCR_MATCH_MIN_CONFIDENCE = float(os.getenv("OCR_MATCH_MIN_CONFIDENCE", "0.65"))
OCR_MAX_WIDTH = int(os.getenv("OCR_MAX_WIDTH", "700"))

try:
    torch.set_num_threads(2)
except Exception:
    pass

print("[INFO] Loading EasyOCR (CPU)...")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
print("[INFO] EasyOCR ready.")

# ============================================================
# DB HELPERS
# ============================================================
def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def reset_stale_processing():
    conn = db_connect()
    conn.execute(
        """
        UPDATE captures
        SET status='pending', updated_at=CURRENT_TIMESTAMP
        WHERE status='processing'
        """
    )
    conn.commit()
    conn.close()


def claim_pending_capture():
    conn = db_connect()
    row = conn.execute(
        """
        SELECT id, timestamp, camera, session_id, track_id, filename, sharpness
        FROM captures
        WHERE status='pending'
        ORDER BY id ASC
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        conn.close()
        return None

    capture_id = int(row["id"])
    result = conn.execute(
        """
        UPDATE captures
        SET status='processing', updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND status='pending'
        """,
        (capture_id,),
    )

    if result.rowcount != 1:
        conn.rollback()
        conn.close()
        return None

    conn.commit()
    data = dict(row)
    data["track_id"] = int(data["track_id"])
    conn.close()
    return data


def lookup_master(code):
    if not code:
        return None
    conn = db_connect()
    row = conn.execute(
        """
        SELECT code, person_name, department
        FROM code_master
        WHERE code=? AND active=1
        """,
        (code.upper(),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_capture_result(
    capture_id,
    status,
    raw_text,
    ocr_text,
    confidence,
    matched_code=None,
    person_name=None,
):
    conn = db_connect()
    conn.execute(
        """
        UPDATE captures
        SET
            status=?,
            ocr_raw_text=?,
            ocr_text=?,
            ocr_confidence=?,
            matched_code=?,
            person_name=?,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (
            status,
            raw_text,
            ocr_text,
            confidence,
            matched_code,
            person_name,
            int(capture_id),
        ),
    )
    conn.commit()
    conn.close()


def result_rank(status):
    return {
        "recognized": 3,
        "unrecognized": 2,
        "no_ocr": 1,
        "pending": 0,
    }.get(status, 0)


def upsert_best_track_result(
    camera,
    session_id,
    track_id,
    status,
    ocr_text,
    matched_code,
    person_name,
    confidence,
    capture_id,
):
    """
    Track overlay hanya menyimpan hasil terbaik:
    recognized > unrecognized > no_ocr.
    Pada rank sama, confidence lebih tinggi menang.
    """
    conn = db_connect()
    current = conn.execute(
        """
        SELECT status, confidence
        FROM track_results
        WHERE camera=? AND session_id=? AND track_id=?
        """,
        (camera, session_id, int(track_id)),
    ).fetchone()

    new_conf = float(confidence or 0.0)
    should_replace = False

    if current is None:
        should_replace = True
    else:
        old_status = current["status"]
        old_conf = float(current["confidence"] or 0.0)
        if result_rank(status) > result_rank(old_status):
            should_replace = True
        elif result_rank(status) == result_rank(old_status) and new_conf > old_conf:
            should_replace = True

    if should_replace:
        conn.execute(
            """
            INSERT INTO track_results (
                camera, session_id, track_id,
                status, ocr_text, matched_code,
                person_name, confidence, capture_id,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(camera, session_id, track_id)
            DO UPDATE SET
                status=excluded.status,
                ocr_text=excluded.ocr_text,
                matched_code=excluded.matched_code,
                person_name=excluded.person_name,
                confidence=excluded.confidence,
                capture_id=excluded.capture_id,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                camera,
                session_id,
                int(track_id),
                status,
                ocr_text,
                matched_code,
                person_name,
                new_conf,
                int(capture_id),
            ),
        )
        conn.commit()

    conn.close()


# ============================================================
# OCR HELPERS
# ============================================================
def extract_ocr_roi_from_person_crop(image):
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]

    # Fokus ke area kode upper torso.
    #
    # Sengaja lebih sempit supaya:
    # - tangan tidak ikut
    # - pinggir badan tidak ikut
    # - lipatan samping berkurang
    # - EasyOCR fokus ke tulisan

    x1 = int(w * 0.18)
    x2 = int(w * 0.82)

    y1 = int(h * 0.17)
    y2 = int(h * 0.43)

    roi = image[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return None

    return roi

def prepare_ocr_variants(image):
    if image is None or image.size == 0:
        return []

    variants = []

    # =====================================
    # VARIANT 1: ORIGINAL COLOR + UPSCALE
    # =====================================
    h, w = image.shape[:2]

    target_width = 700

    if w < target_width:
        scale = target_width / float(w)

        color = cv2.resize(
            image,
            (
                target_width,
                int(h * scale),
            ),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        color = image.copy()

    variants.append(
        ("color", color)
    )

    # =====================================
    # VARIANT 2: GRAYSCALE
    # =====================================
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    h, w = gray.shape[:2]

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

    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        1.0,
    )

    gray = cv2.addWeighted(
        gray,
        1.5,
        blur,
        -0.5,
        0,
    )

    variants.append(
        ("gray", gray)
    )

    return variants


def normalize_text(text):
    if not text:
        return None
    value = re.sub(r"[^A-Z0-9]", "", text.upper())
    return value or None


def prepare_ocr_image(image):
    if image is None or image.size == 0:
        return image

    # Jangan langsung threshold hitam-putih.
    # Informasi karakter pada wearpack justru rusak oleh Otsu global.
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    h, w = gray.shape[:2]

    # Besarkan karena tulisan di bagian depan cukup kecil.
    target_width = 900

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

    # Naikkan local contrast tanpa membuang grayscale.
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    # Mild sharpening.
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

def recognize_generic(image):
    """
    Return:
      raw_text:
        output aktual EasyOCR untuk logging/debug

      candidates:
        candidate hasil OCR tanpa semantic correction

    Tidak ada AI -> A1
    Tidak ada 812 -> B12
    Tidak ada substring correction.
    """

    image = prepare_ocr_image(image)

    started = time.perf_counter()

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        decoder="greedy",
        text_threshold=0.45,
        low_text=0.25,
        link_threshold=0.4,
        canvas_size=768,
        mag_ratio=1.0,
    )

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000.0

    print(
        f"[PERF] OCR {elapsed_ms:.0f} ms"
    )

    if not results:
        return "", []

    items = []

    for bbox, text, confidence in results:
        confidence = float(confidence)

        raw = str(text).strip()

        normalized = normalize_text(raw)

        if not normalized:
            continue

        x_left = min(
            float(point[0])
            for point in bbox
        )

        items.append(
            {
                "x": x_left,
                "raw": raw,
                "text": normalized,
                "confidence": confidence,
            }
        )

    if not items:
        return "", []

    # Susun hasil dari kiri ke kanan
    items.sort(
        key=lambda item: item["x"]
    )

    # Raw OCR tetap disimpan apa adanya
    raw_text = " | ".join(
        item["raw"]
        for item in items
    )

    candidates = []
    seen = set()

    # Candidate masing-masing OCR box
    for item in items:
        text = item["text"]

        if text in seen:
            continue

        seen.add(text)

        candidates.append(
            {
                "text": text,
                "confidence": float(
                    item["confidence"]
                ),
            }
        )

    # Candidate gabungan kiri -> kanan
    #
    # Contoh:
    # "B" + "12"
    # menjadi candidate "B12"
    #
    # Ini bukan semantic correction.
    # Ini hanya menggabungkan OCR box.
    if len(items) > 1:

        combined = "".join(
            item["text"]
            for item in items
        )

        combined = normalize_text(
            combined
        )

        combined_confidence = min(
            float(item["confidence"])
            for item in items
        )

        if (
            combined
            and combined not in seen
        ):
            candidates.append(
                {
                    "text": combined,
                    "confidence": combined_confidence,
                }
            )

    candidates.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    return raw_text, candidates


def choose_result(raw_text, candidates):
    # Prioritas pertama: kandidat yang memang ada di master dan cukup confident.
    recognized_matches = []
    for candidate in candidates:
        master = lookup_master(candidate["text"])
        if master and candidate["confidence"] >= OCR_MATCH_MIN_CONFIDENCE:
            recognized_matches.append((candidate, master))

    if recognized_matches:
        candidate, master = max(
            recognized_matches,
            key=lambda pair: pair[0]["confidence"],
        )
        return {
            "status": "recognized",
            "raw_text": raw_text,
            "ocr_text": candidate["text"],
            "confidence": float(candidate["confidence"]),
            "matched_code": master["code"],
            "person_name": master["person_name"],
        }

    # OCR ada, tapi tidak ada di master / confidence belum cukup.
    if candidates:
        best = candidates[0]
        return {
            "status": "unrecognized",
            "raw_text": raw_text,
            "ocr_text": best["text"],
            "confidence": float(best["confidence"]),
            "matched_code": None,
            "person_name": None,
        }

    return {
        "status": "no_ocr",
        "raw_text": raw_text,
        "ocr_text": None,
        "confidence": 0.0,
        "matched_code": None,
        "person_name": None,
    }


def move_to_processed(filename):
    source = os.path.join(PENDING_DIR, filename)
    destination = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(source):
        shutil.move(source, destination)


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(DB_PATH):
        raise RuntimeError("Database belum ada. Jalankan: python init_db.py")

    reset_stale_processing()
    print("[INFO] Recognition service running. Ctrl+C untuk stop.")

    try:
        while True:
            item = claim_pending_capture()
            if item is None:
                time.sleep(POLL_INTERVAL)
                continue

            capture_id = int(item["id"])
            filename = item["filename"]
            filepath = os.path.join(PENDING_DIR, filename)
            print(
                f"[PROCESS] ID={capture_id} | Track={item['track_id']} | {filename}"
            )

            try:
                image = cv2.imread(filepath)
                if image is None:
                    update_capture_result(
                        capture_id,
                        "error",
                        "",
                        None,
                        0.0,
                    )
                    print(f"[ERROR] ID={capture_id} image tidak bisa dibaca")
                    continue

                roi = extract_ocr_roi_from_person_crop(
                    image
                )

                if roi is not None:

                    debug_image = prepare_ocr_image(
                        roi
                    )

                    debug_path = os.path.join(
                        DEBUG_OCR_DIR,
                        f"capture_{capture_id}_track_{item['track_id']}.jpg"
                    )

                    cv2.imwrite(
                        debug_path,
                        debug_image
                    )

                    print(
                        f"[DEBUG OCR] {debug_path}"
                    )


                if roi is None:

                    result = {
                        "status": "no_ocr",
                        "raw_text": "",
                        "ocr_text": None,
                        "confidence": 0.0,
                        "matched_code": None,
                        "person_name": None,
                    }

                else:

                    raw_text, candidates = (
                        recognize_generic(
                            roi
                        )
                    )

                    result = choose_result(
                        raw_text,
                        candidates
                    )

                update_capture_result(
                    capture_id,
                    result["status"],
                    result["raw_text"],
                    result["ocr_text"],
                    result["confidence"],
                    result["matched_code"],
                    result["person_name"],
                )

                upsert_best_track_result(
                    item["camera"],
                    item["session_id"],
                    int(item["track_id"]),
                    result["status"],
                    result["ocr_text"],
                    result["matched_code"],
                    result["person_name"],
                    result["confidence"],
                    capture_id,
                )

                move_to_processed(filename)

                if result["status"] == "recognized":
                    print(
                        f"[RECOGNIZED] Track={item['track_id']} | "
                        f"OCR={result['ocr_text']} | {result['person_name']} | "
                        f"{result['confidence']:.2f}"
                    )
                elif result["status"] == "unrecognized":
                    print(
                        f"[UNRECOGNIZED] Track={item['track_id']} | "
                        f"OCR={result['ocr_text']} | {result['confidence']:.2f}"
                    )
                else:
                    print(f"[NO OCR] Track={item['track_id']}")

            except Exception as exc:
                print(f"[ERROR] ID={capture_id} | {exc}")
                update_capture_result(
                    capture_id,
                    "error",
                    "",
                    None,
                    0.0,
                )

    except KeyboardInterrupt:
        print("\n[INFO] Recognition stopped.")


if __name__ == "__main__":
    main()

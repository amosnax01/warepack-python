import json
import os
import sqlite3
import time

import cv2
from flask import Flask, Response, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "wearpack.db")
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
LIVE_FRAME_PATH = os.path.join(RUNTIME_DIR, "latest.jpg")
LIVE_META_PATH = os.path.join(RUNTIME_DIR, "detections.json")

PORT = int(os.getenv("LIVE_VIEW_PORT", "5002"))
STREAM_FPS = float(os.getenv("LIVE_STREAM_FPS", "6"))
OUTPUT_JPEG_QUALITY = int(os.getenv("LIVE_OUTPUT_JPEG_QUALITY", "80"))

app = Flask(__name__)


def read_meta():
    try:
        with open(LIVE_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_track_results(camera, session_id, track_ids):
    if not track_ids or not os.path.exists(DB_PATH):
        return {}

    placeholders = ",".join("?" for _ in track_ids)
    params = [camera, session_id] + [int(t) for t in track_ids]
    sql = f"""
        SELECT track_id, status, ocr_text, matched_code, person_name, confidence
        FROM track_results
        WHERE camera=? AND session_id=? AND track_id IN ({placeholders})
    """

    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    return {int(row["track_id"]): dict(row) for row in rows}


def label_for_track(track_id, result):
    if result is None:
        return f"#{track_id} | OCR..."

    status = result.get("status")
    ocr_text = result.get("ocr_text") or "-"
    confidence = float(result.get("confidence") or 0.0)

    if status == "recognized":
        name = result.get("person_name") or "-"
        return f"#{track_id} | {ocr_text} | {name} | {confidence * 100:.0f}%"

    if status == "unrecognized":
        return f"#{track_id} | {ocr_text} | UNRECOGNIZED | {confidence * 100:.0f}%"

    if status == "no_ocr":
        return (
            f"#{track_id} | "
            f"NO OCR | RETRY..."
        )

    return f"#{track_id} | OCR..."


def draw_label(
    frame,
    x1,
    y1,
    text,
    color
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    top = max(0, y1 - th - baseline - 10)
    left = max(0, x1)
    right = min(frame.shape[1] - 1, left + tw + 12)
    bottom = min(frame.shape[0] - 1, top + th + baseline + 10)

    cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
    cv2.putText(
        frame,
        text,
        (left + 6, bottom - baseline - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def build_overlay_frame():
    frame = cv2.imread(LIVE_FRAME_PATH)
    meta = read_meta()
    if frame is None or meta is None:
        return None

    detections = meta.get("detections", [])
    track_ids = [int(d["track_id"]) for d in detections]
    results = get_track_results(
        meta.get("camera", ""),
        meta.get("session_id", ""),
        track_ids,
    )

    # Capture zone.
    zone = meta.get("capture_zone") or {}
    if all(k in zone for k in ("x1", "y1", "x2", "y2")):
        cv2.rectangle(
            frame,
            (int(zone["x1"]), int(zone["y1"])),
            (int(zone["x2"]), int(zone["y2"])),
            (160, 160, 160),
            1,
        )
        cv2.putText(
            frame,
            "CAPTURE ZONE",
            (int(zone["x1"]) + 8, int(zone["y1"]) + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    for d in detections:
        track_id = int(d["track_id"])
        x1 = int(d["x1"])
        y1 = int(d["y1"])
        x2 = int(d["x2"])
        y2 = int(d["y2"])
        in_zone = bool(d.get("in_capture_zone"))
        result = results.get(track_id)
        status = (
            result.get("status")
            if result
            else None
        )

        if status == "recognized":
            # hijau
            color = (
                40,
                190,
                70
            )

        elif status == "unrecognized":
            # merah
            color = (
                40,
                40,
                220
            )

        elif status == "no_ocr":
            # orange, masih retry
            color = (
                0,
                140,
                255
            )

        elif in_zone:
            # kuning: menunggu OCR
            color = (
                0,
                190,
                255
            )

        else:
            # abu-abu: belum masuk capture zone
            color = (
                160,
                160,
                160
            )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        draw_label(
            frame,
            x1,
            y1,
            label_for_track(
                track_id,
                result
            ),
            color=color
        )

    camera = meta.get("camera", "CAM")
    stamp = meta.get("timestamp", "")
    cv2.putText(
        frame,
        f"{camera}  {stamp}",
        (12, frame.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def mjpeg_generator():
    interval = 1.0 / max(STREAM_FPS, 0.1)
    while True:
        started = time.time()
        frame = build_overlay_frame()
        if frame is not None:
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY],
            )
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded.tobytes()
                    + b"\r\n"
                )

        elapsed = time.time() - started
        if elapsed < interval:
            time.sleep(interval - elapsed)


@app.route("/stream")
def stream():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    meta = read_meta()
    return jsonify(
        {
            "ok": meta is not None and os.path.exists(LIVE_FRAME_PATH),
            "camera": meta.get("camera") if meta else None,
            "session_id": meta.get("session_id") if meta else None,
        }
    )


if __name__ == "__main__":
    print(f"[INFO] Live view: http://localhost:{PORT}/stream")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)

import json
import os
import sqlite3
import time

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    send_file,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "wearpack.db")
PENDING_DIR = os.path.join(BASE_DIR, "spool", "pending")
PROCESSED_DIR = os.path.join(BASE_DIR, "spool", "processed")

RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")

LIVE_FRAME_PATH = os.path.join(
    RUNTIME_DIR,
    "latest.jpg",
)

LIVE_META_PATH = os.path.join(
    RUNTIME_DIR,
    "detections.json",
)

PORT = int(
    os.getenv(
        "DASHBOARD_PORT",
        "5001",
    )
)

app = Flask(__name__)


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    return render_template("dashboard.html")

# ============================================================
# LIVE VIEW
# ============================================================

def generate_live_frames():

    last_mtime = None

    while True:

        if not os.path.exists(
            LIVE_FRAME_PATH
        ):
            time.sleep(0.1)
            continue

        try:

            mtime = os.path.getmtime(
                LIVE_FRAME_PATH
            )

            # Jangan kirim ulang frame yang sama terus-menerus.
            if (
                last_mtime is not None
                and mtime == last_mtime
            ):
                time.sleep(0.03)
                continue

            with open(
                LIVE_FRAME_PATH,
                "rb",
            ) as f:
                frame_bytes = f.read()

            last_mtime = mtime

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n"
                b"\r\n"
                + frame_bytes
                + b"\r\n"
            )

        except (
            FileNotFoundError,
            PermissionError,
        ):
            pass

        except Exception as exc:
            print(
                f"[LIVE ERROR] {exc}"
            )

        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_live_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
        headers={
            "Cache-Control": (
                "no-store, no-cache, "
                "must-revalidate, max-age=0"
            ),
            "Pragma": "no-cache",
        },
    )


@app.route("/live_frame")
def live_frame():

    if not os.path.exists(
        LIVE_FRAME_PATH
    ):
        return (
            "Live frame belum tersedia",
            404,
        )

    response = send_file(
        LIVE_FRAME_PATH,
        mimetype="image/jpeg",
        max_age=0,
    )

    response.headers[
        "Cache-Control"
    ] = (
        "no-store, no-cache, "
        "must-revalidate, max-age=0"
    )

    return response


@app.route("/api/live")
def api_live():

    if not os.path.exists(
        LIVE_META_PATH
    ):
        return jsonify(
            {
                "status": "offline",
                "detections": [],
            }
        )

    try:

        with open(
            LIVE_META_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            payload = json.load(f)

        payload["status"] = "online"

        return jsonify(
            payload
        )

    except Exception as exc:

        return jsonify(
            {
                "status": "error",
                "error": str(exc),
                "detections": [],
            }
        ), 500

@app.route("/api/captures")
def api_captures():
    conn = get_db()
    rows = conn.execute(
        """
        SELECT
            id,
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
        FROM captures
        ORDER BY id DESC
        LIMIT 100
        """
    ).fetchall()
    conn.close()

    payload = []
    for row in rows:
        item = dict(row)
        item["id"] = int(item["id"])
        item["track_id"] = int(item["track_id"])
        if item["sharpness"] is not None:
            item["sharpness"] = float(item["sharpness"])
        if item["ocr_confidence"] is not None:
            item["ocr_confidence"] = float(item["ocr_confidence"])
        payload.append(item)
    return jsonify(payload)


@app.route("/api/stats")
def api_stats():
    conn = get_db()
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS total
        FROM captures
        GROUP BY status
        """
    ).fetchall()
    conn.close()

    stats = {row["status"]: int(row["total"]) for row in rows}
    return jsonify(
        {
            "total": sum(stats.values()),
            "recognized": stats.get("recognized", 0),
            "unrecognized": stats.get("unrecognized", 0),
            "pending": stats.get("pending", 0),
            "processing": stats.get("processing", 0),
            "no_ocr": stats.get("no_ocr", 0),
            "error": stats.get("error", 0),
        }
    )


@app.route("/captures/<path:filename>")
def capture_image(filename):
    # Hindari path traversal.
    safe_name = os.path.basename(filename)
    for directory in (PENDING_DIR, PROCESSED_DIR):
        path = os.path.join(directory, safe_name)
        if os.path.exists(path):
            return send_file(path)
    return ("Not found", 404)


if __name__ == "__main__":
    print(f"[INFO] Dashboard: http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)

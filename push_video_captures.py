import os
import re
import glob
import shutil
import sqlite3

from datetime import datetime
from uuid import uuid4


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DB_PATH = os.path.join(
    BASE_DIR,
    "data",
    "wearpack.db",
)

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "runtime",
    "capture_test",
    "B12",
    "track_0",
)

PENDING_DIR = os.path.join(
    BASE_DIR,
    "spool",
    "pending",
)

os.makedirs(
    PENDING_DIR,
    exist_ok=True,
)


CAMERA_NAME = "VIDEO_B12"

SESSION_ID = (
    "B12_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + uuid4().hex[:6]
)

TRACK_ID = 0


def extract_sharpness(filename):
    match = re.search(
        r"sharp([0-9.]+)",
        filename,
    )

    if not match:
        return 0.0

    return float(
        match.group(1)
    )


def insert_capture(
    timestamp,
    filename,
    sharpness,
):
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
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """,
        (
            timestamp,
            CAMERA_NAME,
            SESSION_ID,
            int(TRACK_ID),
            filename,
            float(sharpness),
        ),
    )

    capture_id = int(
        cursor.lastrowid
    )

    conn.commit()
    conn.close()

    return capture_id


files = sorted(
    glob.glob(
        os.path.join(
            SOURCE_DIR,
            "*.jpg",
        )
    )
)

if not files:
    raise RuntimeError(
        f"No captures found: {SOURCE_DIR}"
    )


print()
print(
    f"SESSION : {SESSION_ID}"
)

print(
    f"CAPTURES: {len(files)}"
)

print()


for index, source_path in enumerate(files):

    original_name = os.path.basename(
        source_path
    )

    sharpness = extract_sharpness(
        original_name
    )

    # Supaya filename unique di spool
    filename = (
        f"{SESSION_ID}_"
        f"{index:03d}_"
        f"{original_name}"
    )

    destination = os.path.join(
        PENDING_DIR,
        filename,
    )

    shutil.copy2(
        source_path,
        destination,
    )

    timestamp = (
        datetime.now()
        .isoformat(
            timespec="milliseconds"
        )
    )

    capture_id = insert_capture(
        timestamp,
        filename,
        sharpness,
    )

    print(
        f"[PENDING] "
        f"ID={capture_id} | "
        f"Track={TRACK_ID} | "
        f"sharp={sharpness:.1f} | "
        f"{filename}"
    )


print()
print("DONE")
print(
    "Recognition service can process them now."
)
import argparse
import os
import shutil
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "wearpack.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "spool", "pending"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "spool", "processed"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "runtime"), exist_ok=True)


def backup_existing_db():
    if not os.path.exists(DB_PATH):
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{DB_PATH}.bak.{stamp}"
    shutil.move(DB_PATH, backup_path)
    return backup_path


def create_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            camera TEXT NOT NULL,
            session_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            filename TEXT NOT NULL UNIQUE,
            sharpness REAL,
            status TEXT NOT NULL DEFAULT 'pending',
            ocr_raw_text TEXT,
            ocr_text TEXT,
            ocr_confidence REAL,
            matched_code TEXT,
            person_name TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_captures_status
            ON captures(status);

        CREATE INDEX IF NOT EXISTS idx_captures_track
            ON captures(camera, session_id, track_id);

        CREATE TABLE IF NOT EXISTS code_master (
            code TEXT PRIMARY KEY,
            person_name TEXT NOT NULL,
            department TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS track_results (
            camera TEXT NOT NULL,
            session_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            ocr_text TEXT,
            matched_code TEXT,
            person_name TEXT,
            confidence REAL,
            capture_id INTEGER,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (camera, session_id, track_id),
            FOREIGN KEY (capture_id) REFERENCES captures(id)
        );
        """
    )

    # Seed awal sesuai PoC saat ini.
    conn.execute(
        """
        INSERT OR IGNORE INTO code_master (code, person_name, department, active)
        VALUES ('C1', 'DAROTO', 'RETORT', 1)
        """
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Backup database lama lalu buat schema baru dari nol.",
    )
    args = parser.parse_args()

    if args.reset:
        backup = backup_existing_db()
        if backup:
            print(f"[INFO] Database lama dibackup ke: {backup}")

    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    conn.close()

    print(f"[OK] Database siap: {DB_PATH}")
    print("[OK] Seed code_master: C1 -> DAROTO")


if __name__ == "__main__":
    main()

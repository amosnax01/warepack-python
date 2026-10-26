# Wearpack Recognition PoC v2

Arsitektur:

- `capture_service.py`: RTSP -> YOLO + ByteTrack -> full-body capture -> publish live frame + bbox.
- `recognition_service.py`: generic OCR A-Z/0-9 -> lookup `code_master` -> recognized/unrecognized -> best result per Track ID.
- `live_view_service.py`: membaca live frame + bbox dari capture service dan menambahkan OCR result berdasarkan `camera + session_id + track_id`.
- `dashboard.py`: live view + maximize/minimize + capture log.

## 1. Siapkan database baru

Karena schema berubah dari versi sebelumnya, paling aman jalankan:

```bash
python init_db.py --reset
```

Database lama tidak langsung dihapus; akan dibackup sebagai `wearpack.db.bak.<timestamp>`.

Seed awal:

```text
C1 -> DAROTO
```

Tambah master code contoh:

```bash
sqlite3 data/wearpack.db
```

```sql
INSERT OR REPLACE INTO code_master(code, person_name, department, active)
VALUES ('A1', 'NAMA_ORANG', 'KITCHEN', 1);
.quit
```

Jika `A1` tidak ada di `code_master`, OCR tetap boleh membaca `A1`, tetapi status menjadi `unrecognized` dan `OCR: A1` tetap tampil di card serta live overlay.

## 2. Jalankan capture

```bash
source wearpack-env/bin/activate
export RTSP_URL='rtsp://USERNAME:PASSWORD@192.168.3.2:8554'
export CAMERA_NAME='CAM01'
python capture_service.py
```

Capture service menyimpan full-body crop. Ketajaman dinilai dari torso, bukan seluruh background.

## 3. Jalankan recognition

Terminal kedua:

```bash
source wearpack-env/bin/activate
python recognition_service.py
```

EasyOCR sengaja `gpu=False` karena pada setup Mac sebelumnya concurrent MPS antara YOLO dan EasyOCR tidak stabil.

## 4. Jalankan live view service

Terminal ketiga:

```bash
source wearpack-env/bin/activate
python live_view_service.py
```

Default:

```text
http://localhost:5002/stream
```

Service ini tidak menjalankan YOLO atau OCR lagi. Ia hanya menggabungkan frame + detection metadata + best OCR result per Track ID.

## 5. Jalankan dashboard

Terminal keempat:

```bash
source wearpack-env/bin/activate
python dashboard.py
```

Buka:

```text
http://localhost:5001
```

Live view punya tombol minimize dan fullscreen/maximize.

## Status OCR

- `recognized`: OCR terbaca, code ada di `code_master`, confidence >= threshold.
- `unrecognized`: OCR terbaca tetapi tidak ada di master atau confidence belum cukup.
- `no_ocr`: tidak ada teks yang berhasil dibaca.
- `pending` / `processing`: masih antre / sedang OCR.

## Live overlay

Contoh:

```text
#21 | C1 | DAROTO | 99%
#22 | A1 | UNRECOGNIZED | 91%
#23 | OCR...
```

Track result memakai `(camera, session_id, track_id)`, bukan hanya angka Track ID, supaya tidak salah saat capture service restart dan ByteTrack memakai nomor yang sama lagi.

## Tuning awal

Environment opsional:

```bash
export TARGET_AI_FPS=10
export LIVE_PREVIEW_FPS=6
export LIVE_PREVIEW_WIDTH=960
export LIVE_JPEG_QUALITY=75
export MAX_CAPTURES_PER_TRACK=3
export SHARPNESS_THRESHOLD=50
export MIN_PERSON_HEIGHT=250
```

Capture zone default adalah area tengah frame. Bisa diubah:

```bash
export CAPTURE_ZONE_X1=0.12
export CAPTURE_ZONE_Y1=0.08
export CAPTURE_ZONE_X2=0.88
export CAPTURE_ZONE_Y2=0.92
```

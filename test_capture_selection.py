import os
import cv2

from vision.video_source import VideoSource
from vision.yolox_detector import YOLOXDetector
from vision.bytetrack_tracker import ByteTracker


VIDEO_PATH = "videos/B12.mp4"

FRAME_SELECTION_WINDOW = 0.4
MIN_PERSON_WIDTH = 80
MIN_PERSON_HEIGHT = 180

OUTPUT_DIR = "runtime/capture_test/B12"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_person_crop(frame, box, padding=0.03):
    x1, y1, x2, y2 = box

    h, w = frame.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def extract_torso(person_crop):
    h, w = person_crop.shape[:2]

    x1 = int(w * 0.15)
    x2 = int(w * 0.85)

    y1 = int(h * 0.05)
    y2 = int(h * 0.60)

    torso = person_crop[y1:y2, x1:x2]

    if torso.size == 0:
        return None

    return torso


def calculate_sharpness(image):
    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()


def calculate_quality(frame, box, torso_sharpness):
    frame_h, frame_w = frame.shape[:2]

    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2

    distance_x = abs(
        center_x - frame_center_x
    ) / frame_center_x

    distance_y = abs(
        center_y - frame_center_y
    ) / frame_center_y

    center_score = max(
        0.0,
        1.0 - ((distance_x + distance_y) / 2),
    )

    person_area = box_w * box_h
    frame_area = frame_w * frame_h

    size_score = min(
        1.0,
        person_area / (frame_area * 0.35),
    )

    return (
        torso_sharpness
        + center_score * 150
        + size_score * 100
    )


source = VideoSource(VIDEO_PATH)

detector = YOLOXDetector(
    model_path="models/yolox_s.onnx",
    conf_threshold=0.20,
)

tracker = ByteTracker(
    frame_rate=int(round(source.fps)),
)


track_states = {}


while True:
    ok, frame, frame_index, video_time = source.read()

    if not ok:
        break

    detections = detector.detect(frame)

    tracks = tracker.update(detections)

    active_track_ids = set()

    for track in tracks:
        track_id = int(track["track_id"])
        box = track["box"]

        active_track_ids.add(track_id)

        x1, y1, x2, y2 = box

        person_width = x2 - x1
        person_height = y2 - y1

        if (
            person_width < MIN_PERSON_WIDTH
            or person_height < MIN_PERSON_HEIGHT
        ):
            continue

        person_crop = extract_person_crop(
            frame,
            box,
        )

        if person_crop is None:
            continue

        torso = extract_torso(person_crop)

        torso_sharpness = calculate_sharpness(
            torso
        )

        quality = calculate_quality(
            frame,
            box,
            torso_sharpness,
        )

        state = track_states.setdefault(
            track_id,
            {
                "window_start": video_time,
                "best_score": -1.0,
                "best_frame": None,
                "best_crop": None,
                "best_sharpness": 0.0,
                "best_time": 0.0,
                "best_frame_index": 0,
                "capture_index": 0,
            },
        )

        # Simpan kandidat terbaik dalam window ini
        if quality > state["best_score"]:
            state["best_score"] = quality
            state["best_frame"] = frame.copy()
            state["best_crop"] = person_crop.copy()
            state["best_sharpness"] = torso_sharpness
            state["best_time"] = video_time
            state["best_frame_index"] = frame_index

        elapsed = (
            video_time
            - state["window_start"]
        )

        if elapsed < FRAME_SELECTION_WINDOW:
            continue

        # Window 0.4 detik selesai.
        # Save frame terbaiknya.
        if state["best_crop"] is not None:
            capture_index = state["capture_index"]

            track_dir = os.path.join(
                OUTPUT_DIR,
                f"track_{track_id}",
            )

            os.makedirs(
                track_dir,
                exist_ok=True,
            )

            filename = (
                f"{capture_index:03d}_"
                f"t{state['best_time']:.2f}_"
                f"f{state['best_frame_index']}_"
                f"sharp{state['best_sharpness']:.0f}.jpg"
            )

            path = os.path.join(
                track_dir,
                filename,
            )

            cv2.imwrite(
                path,
                state["best_crop"],
            )

            print(
                f"[CAPTURE] "
                f"track={track_id} "
                f"window={capture_index} "
                f"time={state['best_time']:.2f}s "
                f"frame={state['best_frame_index']} "
                f"sharp={state['best_sharpness']:.1f}"
            )

            state["capture_index"] += 1

        # Mulai window baru
        state["window_start"] = video_time
        state["best_score"] = -1.0
        state["best_frame"] = None
        state["best_crop"] = None
        state["best_sharpness"] = 0.0


source.release()

print()
print("CAPTURE SELECTION FINISHED")
print(f"Output: {OUTPUT_DIR}")
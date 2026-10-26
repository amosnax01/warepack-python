import cv2

from vision.video_source import VideoSource
from vision.yolox_detector import YOLOXDetector
from vision.bytetrack_tracker import ByteTracker


VIDEO_PATH = "videos/B12.mp4"
OUTPUT_PATH = "runtime/b12_tracking.mp4"


source = VideoSource(VIDEO_PATH)

detector = YOLOXDetector(
    model_path="models/yolox_s.onnx",
    conf_threshold=0.20,
)

tracker = ByteTracker(
    frame_rate=int(round(source.fps)),
)


writer = None


while True:
    ok, frame, frame_index, video_time = source.read()

    if not ok:
        break

    detections = detector.detect(frame)

    tracks = tracker.update(detections)

    if writer is None:
        h, w = frame.shape[:2]

        writer = cv2.VideoWriter(
            OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            source.fps,
            (w, h),
        )

    for track in tracks:
        x1, y1, x2, y2 = track["box"]
        track_id = track["track_id"]
        confidence = track["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"ID {track_id} | {confidence:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        frame,
        f"{video_time:.2f}s | frame {frame_index}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    writer.write(frame)

    if frame_index % 30 == 0:
        ids = [t["track_id"] for t in tracks]

        print(
            f"frame={frame_index:4d} "
            f"time={video_time:5.2f}s "
            f"detections={len(detections)} "
            f"tracks={ids}"
        )


source.release()

if writer is not None:
    writer.release()

print(f"saved: {OUTPUT_PATH}")
import numpy as np
import supervision as sv

from trackers import ByteTrackTracker


class ByteTracker:
    def __init__(
        self,
        frame_rate=30,
        track_activation_threshold=0.50,
        lost_track_buffer=30,
        minimum_consecutive_frames=2,
        minimum_iou_threshold=0.15,
        high_conf_det_threshold=0.50,
    ):
        self.tracker = ByteTrackTracker(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
        )

    def update(self, detections):
        if not detections:
            sv_detections = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )
        else:
            xyxy = np.array(
                [d["box"] for d in detections],
                dtype=np.float32,
            )

            confidence = np.array(
                [d["confidence"] for d in detections],
                dtype=np.float32,
            )

            class_id = np.array(
                [d.get("class_id", 0) for d in detections],
                dtype=np.int32,
            )

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )

        tracked = self.tracker.update(sv_detections)

        results = []

        if tracked.tracker_id is None:
            return results

        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])

            if track_id < 0:
                continue

            x1, y1, x2, y2 = tracked.xyxy[i]

            confidence = (
                float(tracked.confidence[i])
                if tracked.confidence is not None
                else 0.0
            )

            results.append(
                {
                    "track_id": track_id,
                    "box": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                    "confidence": confidence,
                    "class_id": 0,
                }
            )

        return results

    def reset(self):
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
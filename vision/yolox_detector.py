import cv2
import numpy as np
import onnxruntime as ort


class YOLOXDetector:
    def __init__(
        self,
        model_path="models/yolox_s.onnx",
        input_size=(640, 640),
        conf_threshold=0.35,
        nms_threshold=0.45,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        print(
            f"[YOLOX] Loaded: {model_path} | "
            f"input={input_size} | "
            f"provider={self.session.get_providers()}"
        )

    def preprocess(self, image):
        input_h, input_w = self.input_size

        h, w = image.shape[:2]

        ratio = min(
            input_h / float(h),
            input_w / float(w),
        )

        resized_w = int(w * ratio)
        resized_h = int(h * ratio)

        resized = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )

        padded = np.ones(
            (input_h, input_w, 3),
            dtype=np.uint8,
        ) * 114

        padded[:resized_h, :resized_w] = resized

        padded = padded.transpose(2, 0, 1)
        padded = np.ascontiguousarray(
            padded,
            dtype=np.float32,
        )

        return padded, ratio

    def demo_postprocess(self, outputs):
        grids = []
        expanded_strides = []

        input_h, input_w = self.input_size

        strides = [8, 16, 32]

        hsizes = [input_h // stride for stride in strides]
        wsizes = [input_w // stride for stride in strides]

        for hsize, wsize, stride in zip(
            hsizes,
            wsizes,
            strides,
        ):
            xv, yv = np.meshgrid(
                np.arange(wsize),
                np.arange(hsize),
            )

            grid = np.stack(
                (xv, yv),
                2,
            ).reshape(1, -1, 2)

            grids.append(grid)

            shape = grid.shape[:2]

            expanded_strides.append(
                np.full(
                    (*shape, 1),
                    stride,
                )
            )

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(
            expanded_strides,
            1,
        )

        outputs[..., :2] = (
            outputs[..., :2] + grids
        ) * expanded_strides

        outputs[..., 2:4] = (
            np.exp(outputs[..., 2:4])
            * expanded_strides
        )

        return outputs

    def nms(self, boxes, scores):
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (
            np.maximum(0, x2 - x1)
            * np.maximum(0, y2 - y1)
        )

        order = scores.argsort()[::-1]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            intersection = w * h

            union = (
                areas[i]
                + areas[order[1:]]
                - intersection
            )

            iou = intersection / np.maximum(
                union,
                1e-6,
            )

            remaining = np.where(
                iou <= self.nms_threshold
            )[0]

            order = order[remaining + 1]

        return keep

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []

        blob, ratio = self.preprocess(frame)

        outputs = self.session.run(
            None,
            {
                self.input_name: blob[None, :, :, :]
            },
        )[0]

        predictions = self.demo_postprocess(
            outputs.copy()
        )[0]

        # YOLOX COCO:
        # class 0 = person
        objectness = predictions[:, 4]
        person_probability = predictions[:, 5]

        person_scores = (
            objectness * person_probability
        )

        mask = (
            person_scores >= self.conf_threshold
        )

        predictions = predictions[mask]
        person_scores = person_scores[mask]

        if len(predictions) == 0:
            return []

        boxes = predictions[:, :4].copy()

        # cx, cy, w, h -> x1, y1, x2, y2
        boxes_xyxy = np.empty_like(boxes)

        boxes_xyxy[:, 0] = (
            boxes[:, 0] - boxes[:, 2] / 2
        )

        boxes_xyxy[:, 1] = (
            boxes[:, 1] - boxes[:, 3] / 2
        )

        boxes_xyxy[:, 2] = (
            boxes[:, 0] + boxes[:, 2] / 2
        )

        boxes_xyxy[:, 3] = (
            boxes[:, 1] + boxes[:, 3] / 2
        )

        boxes_xyxy /= ratio

        frame_h, frame_w = frame.shape[:2]

        boxes_xyxy[:, 0] = np.clip(
            boxes_xyxy[:, 0],
            0,
            frame_w - 1,
        )

        boxes_xyxy[:, 1] = np.clip(
            boxes_xyxy[:, 1],
            0,
            frame_h - 1,
        )

        boxes_xyxy[:, 2] = np.clip(
            boxes_xyxy[:, 2],
            0,
            frame_w - 1,
        )

        boxes_xyxy[:, 3] = np.clip(
            boxes_xyxy[:, 3],
            0,
            frame_h - 1,
        )

        keep = self.nms(
            boxes_xyxy,
            person_scores,
        )

        detections = []

        for index in keep:
            x1, y1, x2, y2 = boxes_xyxy[index]

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "box": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                    "confidence": float(
                        person_scores[index]
                    ),
                    "class_id": 0,
                }
            )

        return detections
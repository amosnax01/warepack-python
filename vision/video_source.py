import cv2


class VideoSource:
    def __init__(self, path, loop=False):
        self.path = path
        self.loop = loop

        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not self.fps or self.fps <= 0:
            self.fps = 30.0

        self.duration = self.total_frames / self.fps
        self.frame_index = 0

        print(
            f"[VIDEO] Opened: {self.path} | "
            f"{self.width}x{self.height} | "
            f"{self.fps:.2f} FPS | "
            f"{self.total_frames} frames | "
            f"{self.duration:.2f}s"
        )

    def read(self):
        ok, frame = self.cap.read()

        if not ok:
            if not self.loop:
                return False, None, None, None

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0

            ok, frame = self.cap.read()

            if not ok:
                return False, None, None, None

        current_frame_index = self.frame_index
        video_time = current_frame_index / self.fps

        self.frame_index += 1

        return True, frame, current_frame_index, video_time

    def release(self):
        if self.cap is not None:
            self.cap.release()
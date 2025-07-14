import cv2
import mediapipe as mp
from .eye import Eye
from .calibration import Calibration

class GazeTracking:
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @property
    def pupils_located(self):
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            image_h, image_w = self.frame.shape[:2]
            self.eye_left = Eye(self.frame, landmarks, "left", self.calibration, image_w, image_h)
            self.eye_right = Eye(self.frame, landmarks, "right", self.calibration, image_w, image_h)
        else:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        if self.pupils_located:
            return self.eye_left.pupil.absolute_coords()

    def pupil_right_coords(self):
        if self.pupils_located:
            return self.eye_right.pupil.absolute_coords()

    def horizontal_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / self.eye_left.width
            pupil_right = self.eye_right.pupil.x / self.eye_right.width
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / self.eye_left.height
            pupil_right = self.eye_right.pupil.y / self.eye_right.height
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        return self.pupils_located and self.horizontal_ratio() <= 0.35

    def is_left(self):
        return self.pupils_located and self.horizontal_ratio() >= 0.65

    def is_center(self):
        return self.pupils_located and not self.is_right() and not self.is_left()

    def is_blinking(self):
        return self.pupils_located and (self.eye_left.blinking + self.eye_right.blinking) / 2 > 3.8

    def annotated_frame(self):
        frame = self.frame.copy()
        if self.pupils_located:
            for pupil in [self.eye_left.pupil, self.eye_right.pupil]:
                x, y = pupil.absolute_coords()
                color = (0, 255, 0)
                cv2.line(frame, (x - 5, y), (x + 5, y), color)
                cv2.line(frame, (x, y - 5), (x, y + 5), color)
        return frame

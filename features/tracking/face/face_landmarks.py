import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Set this to static image or video stream usage
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

def detect_marks(image, face_mesh_model=face_mesh):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh_model.process(img_rgb)

    if not results.multi_face_landmarks:
        return []

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape

    # Convert normalized coordinates to pixel positions
    landmarks = []
    for lm in face_landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks.append((x, y))

    return np.array(landmarks, dtype=np.int32)

# Dummy stub (if still needed in code elsewhere)
def get_landmark_model():
    return face_mesh

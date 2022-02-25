from threading import Thread
import mediapipe as mp
import cv2

class FaceMesh:
    def __init__(self, static=False, max_num_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static,
            max_num_faces=max_num_faces,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.results = None
        self.image = None

    def process(self):
        self.results = self.face_mesh.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

    def draw(self):
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=self.image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)
        return self.image

    def set_image(self, image):
        self.image = image
    
    def get_results(self):
        return self.results
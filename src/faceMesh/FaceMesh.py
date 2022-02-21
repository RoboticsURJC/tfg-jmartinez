from threading import Thread
import mediapipe as mp
import cv2

class FaceMesh:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.results = None

    def process(self, image):
        imageToProcess = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(imageToProcess)

    def draw(self, image):
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)
        return image

    def get_results(self):
        return self.results

""" class FaceMesh:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.results = None

        self.frame = None
        self.stopped = False
        self.processing = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if self.processing:
                imageToProcess = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.results = self.face_mesh.process(imageToProcess)
                self.processing = False

    def process(self, image):
        self.frame = image
        self.processing = True

    def draw(self, image):
        if self.results:
            if self.results.multi_face_landmarks:
                for face_landmarks in self.results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec)
        return image

    def get_results(self):
        return self.results

    def stop(self):
        self.stopped = True """

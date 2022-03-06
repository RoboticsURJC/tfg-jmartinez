import mediapipe as mp
import numpy as np
import cv2
import math

class FaceMesh:
    def __init__(self, static=False, max_num_faces=1, refine=False):
        # Init FaceMesh variables
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static,
            max_num_faces=max_num_faces,
            refine_landmarks=refine,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Results and image
        self.results = None
        self.image = None

        # Index of FaceMesh points
        self.index_left_eye = [33, 160, 158, 133, 153, 144]
        self.index_right_eye = [362, 385, 387, 263, 373, 380]
        self.index_left_eyebrow = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55]
        self.index_right_eyebrow = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
        self.index_vertical_nose = [168, 6, 197, 195, 5, 4, 1]
        self.index_horizontal_nose = [48, 220, 440, 278]
        self.index_out_mouth = [0, 269, 409, 291, 375, 405, 17, 181, 146, 61, 185, 39]
        self.index_in_mouth = [13, 311, 308, 402, 14, 178, 78, 81]
        self.index_left_wrinkle = [129, 203, 206, 216, 212]
        self.index_right_wrinkle = [358, 423, 426, 436, 432]

        # Coordinates of FaceMesh points
        self.coord_left_eye = []
        self.coord_right_eye = []
        self.coord_left_eyebrow = []
        self.coord_right_eyebrow = []
        self.coord_vertical_nose = []
        self.coord_horizontal_nose = []
        self.coord_out_mouth = []
        self.coord_in_mouth = []
        self.coord_left_wrinkle = []
        self.coord_right_wrinkle = []

        # Relative coordinates of FaceMesh points
        self.rcoord_left_eye = []
        self.rcoord_right_eye = []
        self.rcoord_left_eyebrow = []
        self.rcoord_right_eyebrow = []
        self.rcoord_vertical_nose = []
        self.rcoord_horizontal_nose = []
        self.rcoord_out_mouth = []
        self.rcoord_in_mouth = []
        self.rcoord_left_wrinkle = []
        self.rcoord_right_wrinkle = []

        # Distances
        self.num_distances = 33 # 32 distances + class
        self.distances = np.zeros((1, self.num_distances))

    def reset_coords(self):
        self.coord_left_eye.clear()
        self.coord_right_eye.clear()
        self.coord_left_eyebrow.clear()
        self.coord_right_eyebrow.clear()
        self.coord_vertical_nose.clear()
        self.coord_horizontal_nose.clear()
        self.coord_out_mouth.clear()
        self.coord_in_mouth.clear()
        self.coord_left_wrinkle.clear()
        self.coord_right_wrinkle.clear()

    def reset_rcoords(self):
        self.rcoord_left_eye.clear()
        self.rcoord_right_eye.clear()
        self.rcoord_left_eyebrow.clear()
        self.rcoord_right_eyebrow.clear()
        self.rcoord_vertical_nose.clear()
        self.rcoord_horizontal_nose.clear()
        self.rcoord_out_mouth.clear()
        self.rcoord_in_mouth.clear()
        self.rcoord_left_wrinkle.clear()
        self.rcoord_right_wrinkle.clear()

    def reset_distances(self):
        for i in range(self.num_distances):
            self.distances[0][i] = 0

    def distance(self, point1, point2):
        x0 = point1[0]
        y0 = point1[1]
        x1 = point2[0]
        y1 = point2[1]
        return math.sqrt((x0 - x1)**2+(y0 - y1)**2)

    def calculate_distances(self):
        # DISTANCE INDEX
        # Eyebrows: 0 - 9
        # Eyes: 10 - 13
        # Wrinkles: 14 - 23
        # Mouth: 24 - 31
        index = 0

        # Eyebrows
        for i in range(5):
            self.distances[0][index] = self.distance(self.coord_left_eyebrow[i], self.coord_vertical_nose[5])
            index += 1
        for i in range(5):
            self.distances[0][index] = self.distance(self.coord_right_eyebrow[i], self.coord_vertical_nose[5])
            index += 1
        
        # Eyes
        self.distances[0][index] = self.distance(self.coord_left_eye[1], self.coord_left_eye[5])
        index += 1
        self.distances[0][index] = self.distance(self.coord_left_eye[2], self.coord_left_eye[4])
        index += 1
        self.distances[0][index] = self.distance(self.coord_right_eye[1], self.coord_right_eye[5])
        index += 1
        self.distances[0][index] = self.distance(self.coord_right_eye[2], self.coord_right_eye[4])
        index += 1

        # Wrinkles
        for i in range(5):
            self.distances[0][index] = self.distance(self.coord_left_wrinkle[i], self.coord_vertical_nose[5])
            index += 1
        for i in range(5):
            self.distances[0][index] = self.distance(self.coord_right_wrinkle[i], self.coord_vertical_nose[5])
            index += 1

        # Mouth
        self.distances[0][index] = self.distance(self.coord_out_mouth[9], self.coord_vertical_nose[5])
        index += 1
        self.distances[0][index] = self.distance(self.coord_out_mouth[11], self.coord_vertical_nose[5])
        index += 1
        self.distances[0][index] = self.distance(self.coord_out_mouth[0], self.coord_vertical_nose[5])
        index += 1
        self.distances[0][index] = self.distance(self.coord_out_mouth[1], self.coord_vertical_nose[5])
        index += 1
        self.distances[0][index] = self.distance(self.coord_out_mouth[3], self.coord_vertical_nose[5])
        index += 1
        self.distances[0][index] = self.distance(self.coord_in_mouth[7], self.coord_in_mouth[5])
        index += 1
        self.distances[0][index] = self.distance(self.coord_in_mouth[0], self.coord_in_mouth[4])
        index += 1
        self.distances[0][index] = self.distance(self.coord_in_mouth[1], self.coord_in_mouth[3])

    def update_coords(self, face_landmarks):
        height, width, _ = self.image.shape
        for index in self.index_left_eye:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coord_left_eye.append((x, y))
            self.rcoord_left_eye.append((int(x*width), int(y*height)))
        for index in self.index_right_eye:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coord_right_eye.append((x, y))
            self.rcoord_right_eye.append((int(x*width), int(y*height)))
        for index in self.index_left_eyebrow:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coord_left_eyebrow.append((x, y))
            self.rcoord_left_eyebrow.append((int(x*width), int(y*height)))
        for index in self.index_right_eyebrow:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coord_right_eyebrow.append((x, y))
            self.rcoord_right_eyebrow.append((int(x*width), int(y*height)))
        for index in self.index_vertical_nose:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coord_vertical_nose.append((x, y))
            self.rcoord_vertical_nose.append((int(x*width), int(y*height)))
        for index in self.index_horizontal_nose:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coord_horizontal_nose.append((x, y))
            self.rcoord_horizontal_nose.append((int(x*width), int(y*height)))
        for index in self.index_out_mouth:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coord_out_mouth.append((x, y))
            self.rcoord_out_mouth.append((int(x*width), int(y*height)))
        for index in self.index_in_mouth:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coord_in_mouth.append((x, y))
            self.rcoord_in_mouth.append((int(x*width), int(y*height)))
        for index in self.index_left_wrinkle:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coord_left_wrinkle.append((x, y))
            self.rcoord_left_wrinkle.append((int(x*width), int(y*height)))
        for index in self.index_right_wrinkle:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coord_right_wrinkle.append((x, y))
            self.rcoord_right_wrinkle.append((int(x*width), int(y*height)))

    def process(self):
        self.results = self.face_mesh.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.reset_coords()
        self.reset_rcoords()
        self.reset_distances()
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.update_coords(face_landmarks)
                self.calculate_distances()

    def draw(self):
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=self.image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)

    def draw_left_eye(self):
        for coord in self.rcoord_left_eye:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)

    def draw_right_eye(self):
        for coord in self.rcoord_right_eye:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)

    def draw_left_eyebrow(self):
        for coord in self.rcoord_left_eyebrow:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)

    def draw_right_eyebrow(self):
        for coord in self.rcoord_right_eyebrow:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)
    
    def draw_nose(self):
        for coord in self.rcoord_vertical_nose:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)
        for coord in self.rcoord_horizontal_nose:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)

    def draw_mouth(self):
        for coord in self.rcoord_out_mouth:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)
        for coord in self.rcoord_in_mouth:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)

    def draw_wrinkles(self):
        for coord in self.rcoord_left_wrinkle:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)
        for coord in self.rcoord_right_wrinkle:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)

    def set_image(self, image):
        self.image = image.copy()

    def get_image(self):
        return self.image
    
    def get_results(self):
        return self.results

    def get_distances(self):
        return self.distances

    def face_mesh_detected(self):
        if self.results.multi_face_landmarks:
            return True
        return False 

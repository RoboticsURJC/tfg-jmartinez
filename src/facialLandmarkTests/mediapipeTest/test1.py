# Test face mesh mediapipe

import cv2
import time
import mediapipe as mp
from picamera import PiCamera
from picamera.array import PiRGBArray
import sys
import argparse

# Init Picamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
widthImage = 640
heightImage = 480
cap = PiRGBArray(camera, size=(widthImage, heightImage))

# Variables to calculate and show FPS
counter, fps = 0, 0
fps_avg_frame_count = 10
text_location = (20, 24)
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
start_time = time.time()

# Init time of program
init_time = time.time()

# Init mediapipe variables
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_face_mesh(image, face_mesh):
    imageToProcess = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imageToProcess)
    return results

def draw_face_mesh(image, results):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image, 
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
    return image

def draw_fps(image, fps):
    fps_text = 'FPS = {:.1f}'.format(fps)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    return image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savefps',
                    action='store_true',
                    default=False, required=False,
                    help='Activa el guardado de fps en fichero .csv')
    parser.add_argument('--savebugs',
                    action='store_true',
                    default=False, required=False,
                    help='Activa el guardado de fallos en el algoritmo')
    parser.add_argument('--time',
                    type=int,
                    default=30, required=False,
                    help='Duracion en segundos del guardado de datos')
    return parser.parse_args()
    

if __name__ == '__main__':

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    args = parse_arguments()
    if args['--savefps']:
        f = open('../dataFPS/mediapipeTest/fps_mediapipe_1.csv', 'w')

    for frame in camera.capture_continuous(cap, format='bgr', use_video_port=True):
        image = frame.array
        image = cv2.flip(image, 0)

        results = process_face_mesh(image, face_mesh)
        image = draw_face_mesh(image, results)

        # Calculate the FPS
        counter += 1
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Save FPS
        if args['--savefps']:
            f.write('{:.1f}, {:.1f}\n'.format(fps, (time.time() - init_time)))
            if (time.time() - init_time) >= args['--time']:
                args['--savefps'] = False
                print("FPS save is over")

        # Show the FPS
        image = draw_fps(image, fps)
        
        # Show image
        cv2.imshow("imagen", image)
        cap.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


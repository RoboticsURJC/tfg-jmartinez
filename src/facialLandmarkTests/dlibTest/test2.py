# Test dlib with face detection using Haar cascades (v1)

from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import time
import numpy as np
import dlib
import argparse

# Init Picamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
widthImage = 640
heightImage = 480
cap = PiRGBArray(camera, size=(widthImage, heightImage))

# Init time of program
init_time = time.time()

# Variables to detect bugs
NUM_FACES = 1

# Variables to calculate and show FPS
counter, fps = 0, 0
fps_avg_frame_count = 10
text_location = (20, 24)
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
start_time = time.time()

# Init dlib variables
#detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
LANDMARK_POINTS = 68

# Init cascade
faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def detect_landmarks(gray, rect):
    # Get the landmark points
    shape = predictor(gray, rect)
    # Convert it to the NumPy Array
    shape_np = np.zeros((LANDMARK_POINTS, 2), dtype="int")
    for i in range(0, LANDMARK_POINTS):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    return shape_np

def draw_landmarks(shape, image):
    for i, (x, y) in enumerate(shape):
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savefps',
                    action='store_true',
                    default=False, required=False,
                    help='Activa el guardado de fps en fichero .csv')
    parser.add_argument('--savebugs',
                    action='store_true',
                    default=False, required=False,
                    help='Activa el guardado de fallos del algoritmo en fichero .csv')
    parser.add_argument('--time',
                    type=int,
                    default=30, required=False,
                    help='Duracion en segundos del guardado de datos (int). Default: 30s')
    return parser.parse_args()

def savefps(fps, max_time, fps_file):
    fps_file.write('{:.1f}, {:.1f}\n'.format(fps, (time.time() - init_time)))
    if (time.time() - init_time) >= max_time:
        print("FPS save is over")
        return False
    return True

def savebugs(rects, max_time, bugs_file):
    if len(rects) == NUM_FACES:
        bugs_file.write('0, {:.1f}\n'.format((time.time() - init_time)))
    else:
        bugs_file.write('1, {:.1f}\n'.format((time.time() - init_time)))
    if (time.time() - init_time) >= max_time:
            print("BUGS save is over")
            return False
    return True

def calculate_fps():
    global counter, fps, start_time
    counter += 1
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()
    return fps

def draw_fps(image, fps):
    fps_text = 'FPS = {:.1f}'.format(fps)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    return image

if __name__ == '__main__':
    args = parse_arguments()
    if args.savefps:
        fps_file = open('../dataFPS/dlibTest/fps_dlib_test2.csv', 'w')
    if args.savebugs:
        bugs_file = open('../dataBugs/dlibTest/bugs_dlib_test2.csv', 'w')

    for frame in camera.capture_continuous(cap, format='bgr', use_video_port=True):
        image = frame.array
        image = cv2.flip(image, 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
        )

        rects = []
        # Draw a rectangle and convert face to dlib.rectangle
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rect = dlib.rectangle(left=x , top=y, right=x+w, bottom=y+h)
            rects.append(rect)

        # Detect landmarks for each face
        for rect in rects:
            shape = detect_landmarks(gray, rect)
            draw_landmarks(shape, image)
        
        # Calculate the FPS
        fps = calculate_fps()

        # Save FPS
        if args.savefps:
            args.savefps = savefps(fps, args.time, fps_file)

        # Save bugs
        if args.savebugs:
            args.savebugs = savebugs(rects, args.time, bugs_file)
            
        # Show the FPS
        image = draw_fps(image, fps)

        # Show image
        cv2.imshow("imagen", image)
        cap.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
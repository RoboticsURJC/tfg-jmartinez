# Test dlib with dlib face detector

from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import time
import numpy as np
import dlib
import sys

# Init Picamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
widthImage = 640
heightImage = 480
cap = PiRGBArray(camera, size=(widthImage, heightImage))

# Init time of program
init_time = time.time()

# Variables to calculate and show FPS
counter, fps = 0, 0
fps_avg_frame_count = 10
text_location = (20, 24)
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
start_time = time.time()

# Init dlib variables
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
LANDMARK_POINTS = 68

# Variables to save fps data
saveFps = False
if len(sys.argv) >= 2:
    if sys.argv[1] == '--savefps':
        f = open('dataFPS/fps_dlib_basic.csv', 'w')
        saveFps = True
        saveFpsTime = 10
        if len(sys.argv) == 3:
            saveFpsTime = int(sys.argv[2])

for frame in camera.capture_continuous(cap, format='bgr', use_video_port=True):
    img = frame.array
    img = cv2.flip(img, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face
    rects = detector(gray, 1)

    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
	    # Convert it to the NumPy Array
        shape_np = np.zeros((LANDMARK_POINTS, 2), dtype="int")
        for i in range(0, LANDMARK_POINTS):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    
    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()

    # Save FPS
    if saveFps:
        f.write('{:.1f}, {:.1f}\n'.format(fps, (time.time() - init_time)))
        if (time.time() - init_time) >= saveFpsTime:
            saveFps = False
            print("FPS save is over")

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    cv2.putText(img, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Show image
    cv2.imshow("imagen", img)
    cap.truncate(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
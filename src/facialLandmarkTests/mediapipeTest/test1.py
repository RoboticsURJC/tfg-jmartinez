import cv2
import time
import mediapipe as mp
from picamera import PiCamera
from picamera.array import PiRGBArray
import sys

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

# Variables to save fps data
saveFps = False
if len(sys.argv) >= 2:
    if sys.argv[1] == '--savefps':
        f = open('../dataFPS/mediapipe/fps_mediapipe_1.csv', 'w')
        saveFps = True
        saveFpsTime = 10
        if len(sys.argv) == 3:
            saveFpsTime = int(sys.argv[2])

# Init time of program
init_time = time.time()

# Init mediapipe variables
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
    for frame in camera.capture_continuous(cap, format='bgr', use_video_port=True):
        img = frame.array
        img = cv2.flip(img, 0)
        imageToProcess = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(imageToProcess)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(img, face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1))

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



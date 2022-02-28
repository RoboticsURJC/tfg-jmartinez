import sys
sys.path.append('..')
from piVideoStream.PiVideoStream import PiVideoStream
from faceMesh.FaceMesh import FaceMesh
import cv2
import time

# Start video stream
vs = PiVideoStream(resolution=(640, 480)).start()
time.sleep(2.0)

facemesh = FaceMesh(static=False, max_num_faces=1)

while(True):
    image = vs.read()
    image = cv2.flip(image, 0)

    facemesh.set_image(image)
    facemesh.process()
    facemesh.draw_left_eye()
    facemesh.draw_right_eye()
    facemesh.draw_left_eyebrow()
    facemesh.draw_right_eyebrow()
    facemesh.draw_nose()
    facemesh.draw_mouth()
    facemesh.draw_wrinkles()
    image = facemesh.get_image()

    # Show image
    cv2.imshow("imagen", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cv2.destroyAllWindows()
vs.stop()
import sys
sys.path.append('../..')
from faceMesh.FaceMesh import FaceMesh
import cv2
import glob as gb
import math

stream = cv2.VideoCapture(1)
facemesh = FaceMesh(static=False, max_num_faces=1)

while(True):
    grabbed, image = stream.read()
    if not grabbed:
        break

    image = cv2.flip(image, 1)

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
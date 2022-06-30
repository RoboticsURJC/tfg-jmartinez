# Programa para probar a mostar por pantalla los datos
# ofrecidos por una webcam USB en Raspberry Pi 4

import cv2

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("error opening webcam")
    exit(1)

while (cam.isOpened()):
    ret, frame = cam.read()
    cv2.imshow('webcam', frame)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()

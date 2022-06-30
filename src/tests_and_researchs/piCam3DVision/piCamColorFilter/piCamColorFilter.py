import cv2  
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

def nothing(x):
	pass

cv2.namedWindow('trackbar')
cv2.createTrackbar('hmin','trackbar',0,180,nothing)
cv2.createTrackbar('hmax','trackbar',180,180,nothing)
cv2.createTrackbar('smin','trackbar',0,255,nothing)
cv2.createTrackbar('smax','trackbar',255,255,nothing)
cv2.createTrackbar('vmin','trackbar',0,255,nothing)
cv2.createTrackbar('vmax','trackbar',255,255,nothing)

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
cap = PiRGBArray(camera, size=(640, 480))

for frame in camera.capture_continuous(cap, format='bgr', use_video_port=True):
    img = frame.array
    img = cv2.flip(img, 0)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hmin = int(cv2.getTrackbarPos('hmin','trackbar'))
    hmax = int(cv2.getTrackbarPos('hmax','trackbar'))
    smin = int(cv2.getTrackbarPos('smin','trackbar'))
    smax = int(cv2.getTrackbarPos('smax','trackbar'))
    vmin = int(cv2.getTrackbarPos('vmin','trackbar'))
    vmax = int(cv2.getTrackbarPos('vmax','trackbar'))

    hsvmin = np.array([hmin, smin, vmin])
    hsvmax = np.array([hmax, smax, vmax])
    mask = cv2.inRange(imgHSV, hsvmin, hsvmax)

    cv2.imshow('imagen original', img)
    #cv2.imshow('imagen hsv', imgHSV)
    cv2.imshow('máscara', mask)

    cap.truncate(0)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
		
#destroys all window
cv2.destroyAllWindows()
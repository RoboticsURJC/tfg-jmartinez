from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import numpy as np
from piCamModel.PinholeCamera import PinholeCamera

class Point2D:
    def __init__(self, x=0., y=0., h=0.):
        self.x = x
        self.y = y
        self.h = h

class Point3D:
    def __init__(self, x=0., y=0., z=0., h=0.):
        self.x = x
        self.y = y
        self.z = z
        self.h = h

def filterPoint2D(imgHSV, hsvmin, hsvmax):
    mask = cv2.inRange(imgHSV, hsvmin, hsvmax)
    moments = cv2.moments(mask)
    if moments['m00'] != 0:
        x = int(moments['m10']/moments['m00'])
        y = int(moments['m01']/moments['m00'])
        point2D = Point2D(x, y, 1)
        return point2D
    pass

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
widthImage = 640
heightImage = 480
cap = PiRGBArray(camera, size=(widthImage, heightImage))

hsvmin = np.array([0, 86, 19])
hsvmax = np.array([180, 220, 255])

for frame in camera.capture_continuous(cap, format='bgr', use_video_port=True):
    img = frame.array
    img = cv2.flip(img, 0)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    point2D = filterPoint2D(imgHSV, hsvmin, hsvmax)
    cv2.circle(img, (point2D.x, point2D.y), 5, (255, 0, 0), -1)

    K = ("383.18987266 0. 327.20114715;"
        "0. 384.14439273 204.95382174;"
        "0. 0. 1.")
    # rotation angle: 25 degrees
    RT = ("0.906307787 0. -0.4226182617 0.;"
        "0. 1. 0. 0.;"
        "-0.4226182617 0. 0.906307787 50.;"
        "0. 0. 0. 1.")
    cameraPosition = Point3D(0., 0., 50.)
    camera = PinholeCamera(K, RT, widthImage, heightImage, cameraPosition)
    output, point3D = camera.backproject(point2D)
    print("point 3D: ", point3D.x, point3D.y, point3D.z, point3D.h)
    print("output: ", output)

    cv2.imshow("imagen", img)

    cap.truncate(0)
    cv2.waitKey(5)
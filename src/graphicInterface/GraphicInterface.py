from threading import Thread
import numpy as np
import cv2

class GraphicInterface:
    def __init__(self, resolution=(640, 480)):
        self.frame = np.zeros((resolution[0], resolution[1], 3), np.uint8)
        self.stopped = False
    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            cv2.imshow("Graphic Interface", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
                break

    def put_image(self, image):
        self.frame = image

    def put_text(self, text, location, color, size, thickness):
        cv2.putText(self.frame, text, location, cv2.FONT_HERSHEY_PLAIN,
                    size, color, thickness)

    def stop(self):
        self.stopped = True
# Test FPS without computational stress and PiVideoStream

from piVideoStream.PiVideoStream import PiVideoStream
import cv2
import time
import numpy as np
import sys
import dlib
import imutils

# Init PiVideoStream
vs = PiVideoStream(resolution=(640, 480))

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

# Variables to save fps data
saveFps = False
if len(sys.argv) >= 2:
    if sys.argv[1] == '--savefps':
        f = open('dataFPS/fps_basic_2.csv', 'w')
        saveFps = True
        saveFpsTime = 10
        if len(sys.argv) == 3:
            saveFpsTime = int(sys.argv[2])

vs.start()
time.sleep(2.0)
while(True):
    img = vs.read()
    img = cv2.flip(img, 0)
    
    # Calculate the FPS
    counter += 1
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
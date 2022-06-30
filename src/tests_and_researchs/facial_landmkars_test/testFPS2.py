# Test FPS without computational stress and PiVideoStream

import sys
sys.path.append('..')
from pi_video_stream.pi_video_stream import PiVideoStream
import cv2
import time
import argparse

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savefps',
                    action='store_true',
                    default=False, required=False,
                    help='Activa el guardado de fps en fichero .csv')
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
        fps_file = open('dataFPS/fps_basic_2.csv', 'w')
    
    vs.start()
    time.sleep(2.0)
    while(True):
        image = vs.read()
        image = cv2.flip(image, 0)
        
        # Calculate the FPS
        fps = calculate_fps()

        # Save FPS
        if args.savefps:
            args.savefps = savefps(fps, args.time, fps_file)
            
        # Show the FPS
        image = draw_fps(image, fps)

        # Show image
        cv2.imshow("imagen", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()
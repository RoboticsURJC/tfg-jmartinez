import sys
sys.path.append('../..')
from emotionalMesh.EmotionalMesh2 import EmotionalMesh2
import cv2
import pickle
import time
import argparse

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

def draw_fps(image, fps):
    fps_text = 'FPS = {:.1f}'.format(fps)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    return image

def calculate_fps():
    global counter, fps, start_time
    counter += 1
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()
    return fps

def savefps(fps, max_time, fps_file):
    fps_file.write('{:.1f}, {:.1f}\n'.format(fps, (time.time() - init_time)))
    if (time.time() - init_time) >= max_time:
        print("FPS save is over")
        return False
    return True

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

def draw_predict(pred, image):
    if pred == 1:
        emotion = "Anger"
    elif pred == 2:
        emotion = "Contempt"
    elif pred == 3:
        emotion = "Disgust"
    elif pred == 4:
        emotion = "Fear"
    elif pred == 5:
        emotion = "Happy"
    elif pred == 6:
        emotion = "Sadness"
    else:
        emotion = "Surprise"
    cv2.putText(image, emotion, (20, 456), cv2.FONT_HERSHEY_PLAIN,
                2, text_color, font_thickness)

def main():
    args = parse_arguments()
    if args.savefps:
        fps_file = open('../dataFPS/emotionalMesh/fps_MLP.csv', 'w')
    
    # Input video
    input_video = cv2.VideoCapture('../../piVideoStream/recordings/video.avi')

    # Init EmotionalMesh
    emotionalmesh = EmotionalMesh2(static=False, max_num_faces=1)

    # Load the model
    with open('../model/emotionalMesh/all_classes/model_MLP_allangles.pkl', 'rb') as modelfile:
        loaded = pickle.load(modelfile)
    model = loaded['model']
    pca = loaded['pca_fit']

    while (input_video.isOpened()):
        # Read image from input_video
        ret, image = input_video.read()

        if ret == True:
            # Process image
            emotionalmesh.set_image(image)
            emotionalmesh.process()
            emotionalmesh.draw_emotional_mesh()
            image = emotionalmesh.get_image()
            if emotionalmesh.face_mesh_detected():
                angles = emotionalmesh.get_angles()
                angles = pca.transform(angles)
                pred = model.predict(angles)
                draw_predict(pred, image)

            # Calculate the FPS
            fps = calculate_fps()

            # Save FPS
            if args.savefps:
                args.savefps = savefps(fps, args.time, fps_file)

            # Draw the FPS
            image = draw_fps(image, fps)

            # Show image
            cv2.imshow("imagen", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cv2.destroyAllWindows()
    input_video.release()

if __name__ == '__main__':
    main()
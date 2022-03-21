import sys
sys.path.append('../..')
from faceMesh.FaceMesh import FaceMesh
import cv2
import glob as gb
import numpy as np
import csv
import os
import re

def paths(images):
    path_image = images[-1]
    path_emotion = re.sub('extended-cohn-kanade-images/cohn-kanade-images', 
                          'Emotion_labels/Emotion', path_image)
    path_emotion = re.sub('.png', '_emotion.txt', path_emotion)
    return path_image, path_emotion

def class_emotion(path_emotion):
    try:
        f_emotion = open(path_emotion, 'r')
        line = f_emotion.readline()
        return int(float(line.strip()))
    except:
      return None

def emotion_lists():
    images_path = "../../../../CK+/extended-cohn-kanade-images/cohn-kanade-images"

    anger = []     # class 1
    contempt = []  # class 2
    disgust = []   # class 3
    fear = []      # class 4
    happy = []     # class 5
    sadness = []   # class 6
    surprise = []  # class 7

    for dirpath, _, _ in os.walk(images_path, topdown=True):
        images = gb.glob(dirpath+'/*.png')
        if images:
            images = sorted(images)
            path_image, path_emotion = paths(images)
            emotion = class_emotion(path_emotion)
            if emotion == 1:
                anger.append(path_image)
            elif emotion == 2:
                contempt.append(path_image)
            elif emotion == 3:
                disgust.append(path_image)
            elif emotion == 4:
                fear.append(path_image)
            elif emotion == 5:
                happy.append(path_image)
            elif emotion == 6:
                sadness.append(path_image)
            elif emotion == 7:
                surprise.append(path_image)

    return anger, contempt, disgust, fear, happy, sadness, surprise

def process_class(images, class_num):
    global process, not_process
    for image in images:
        facemesh.set_image(cv2.imread(image))
        facemesh.process(distances3D=False, mode=0)
        if facemesh.face_mesh_detected():
            distances = facemesh.get_distances()
            distances[0][-1] = class_num # set class
            np.savetxt(f, distances, delimiter=",")
            process += 1
        else:
            not_process += 1

if __name__ == '__main__':
    # Open dataset file to write
    f = open('../dataset/distances/datasetCK+.csv', 'w', newline='')

    # Init FaceMesh
    facemesh = FaceMesh(static=True, max_num_faces=1, refine=True)

    # Dataset file header
    for i in range(1, facemesh.num_distances):
        f.write('X'+repr(i)+',')
    f.write('y\n')

    anger, contempt, disgust, fear, happy, sadness, surprise = emotion_lists()

    not_process = 0 # Number of not process images
    process = 0 # Number of process images

    # Process images and write data in dataset file
    # ---------------------------------------------
    print("Procesando imagenes anger...")
    process_class(anger, 1)

    #print("Procesando imagenes contempt...")
    #process_class(contempt, 2)

    print("Procesando imagenes disgust...")
    process_class(disgust, 3)

    #print("Procesando imagenes fear...")
    #process_class(fear, 4)

    print("Procesando imagenes happy...")
    process_class(happy, 5)

    #print("Procesando imagenes sadness...")
    #process_class(sadness, 6)

    print("Procesando imagenes surprise...")
    process_class(surprise, 7)
    # ---------------------------------------------

    print("Imagenes procesadas: "+str(process))
    print("Imagenes no procesadas: "+str(not_process))

        
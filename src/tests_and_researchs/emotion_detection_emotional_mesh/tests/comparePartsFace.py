# Script for research: https://github.com/jmvega/tfg-jmartinez/wiki/Progreso-marzo-2022#elecci%C3%B3n-de-los-%C3%A1ngulos-de-entrenamiento-estudio-2

import sys
sys.path.append('../..')
from emotional_mesh.emotional_mesh_3 import EmotionalMesh3
import cv2
import glob as gb
import numpy as np
import pandas as pd
import math
import os
import re
import matplotlib.pyplot as plt

def paths(images):
    path_image = images[-1]
    path_emotion = re.sub('extended-cohn-kanade-images/cohn-kanade-images', 
                          'Emotion_labels/Emotion', path_image)
    path_emotion = re.sub('.png', '_emotion.txt', path_emotion)
    path_image_neutral = re.sub('000000*[0-9][0-9]', '00000001', path_image)
    return path_image, path_image_neutral, path_emotion

def class_emotion(path_emotion):
    try:
        f_emotion = open(path_emotion, 'r')
        line = f_emotion.readline()
        return int(float(line.strip()))
    except:
      return None

def emotion_lists():
    images_path = "../../../../CK+/extended-cohn-kanade-images/cohn-kanade-images"

    neutral = {
        "anger": [],
        "contempt": [],
        "disgust": [],
        "fear": [],
        "happy": [],
        "sadness": [],
        "surprise": []
    }
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
            path_image, path_image_neutral, path_emotion = paths(images)
            emotion = class_emotion(path_emotion)
            if emotion == 1:
                neutral["anger"].append(path_image_neutral)
                anger.append(path_image)
            elif emotion == 2:
                neutral["contempt"].append(path_image_neutral)
                contempt.append(path_image)
            elif emotion == 3:
                neutral["disgust"].append(path_image_neutral)
                disgust.append(path_image)
            elif emotion == 4:
                neutral["fear"].append(path_image_neutral)
                fear.append(path_image)
            elif emotion == 5:
                neutral["happy"].append(path_image_neutral)
                happy.append(path_image)
            elif emotion == 6:
                neutral["sadness"].append(path_image_neutral)
                sadness.append(path_image)
            elif emotion == 7:
                neutral["surprise"].append(path_image_neutral)
                surprise.append(path_image)

    return neutral, anger, contempt, disgust, fear, happy, sadness, surprise

def angle_difference(alpha, beta):
    phi = abs(beta-alpha)%360
    if phi > 180:
        return (360 - phi)
    return phi

def compare_left_and_right(images_emotion):
    results = []

    for i in range(len(images_emotion)):
        differences = []
        emotionalmesh.set_image(cv2.imread(images_emotion[i]))
        emotionalmesh.process()
        if emotionalmesh.face_mesh_detected():
            angles_left = emotionalmesh.get_angles_left()
            angles_right = emotionalmesh.get_angles_right()
            emotionalmesh.draw_emotional_mesh()
            e = emotionalmesh.get_image()

        for i in range(emotionalmesh.num_angles_left):
            differences.append(angle_difference(angles_left[0][i], angles_right[0][i]))
        
        results.append(differences)

    return results

def compute_mean(angle_differences):
    sum = [0] * len(angle_differences[0])
    mean = [0] * len(sum)
    for i in range(len(angle_differences)):
        for j in range(len(angle_differences[i])):
            sum[j] += angle_differences[i][j]
    
    for i in range(len(sum)):
        mean[i] = sum[i]/len(angle_differences)

    return mean

if __name__ == '__main__':
    # Init EmotionalMesh
    emotionalmesh = EmotionalMesh3(static=True, refine=True)

    neutral, anger, contempt, disgust, fear, happy, sadness, surprise = emotion_lists()

    # Angle names for plot
    angles_names = ["0", "1", "2", "3", "4", "5", "6", 
    "7", "8", "9"]

    # Anger -----------------------------------------------------
    anger_results = compare_left_and_right(anger)
    anger_results_mean = compute_mean(anger_results)
    df_anger = pd.DataFrame({'Angle': angles_names,
                            'Value': anger_results_mean})
    df_anger = df_anger.sort_values('Value')

    plt.bar('Angle', 'Value', data=df_anger)
    plt.ylabel("Diferencia entre izquierda y derecha (grados)")
    plt.xlabel("Nombre de los ángulos")
    #plt.title("Angle difference of left/right face in anger")
    plt.show()

    # Contempt ---------------------------------------------------
    contempt_results = compare_left_and_right(contempt)
    contempt_results_mean = compute_mean(contempt_results)
    df_contempt = pd.DataFrame({'Angle': angles_names,
                            'Value': contempt_results_mean})
    df_contempt = df_contempt.sort_values('Value')

    plt.bar('Angle', 'Value', data=df_contempt)
    plt.ylabel("Diferencia entre izquierda y derecha (grados)")
    plt.xlabel("Nombre de los ángulos")
    #plt.title("Angle difference of left/right face in contempt")
    plt.show()

    # Disgust ----------------------------------------------------
    disgust_results = compare_left_and_right(disgust)
    disgust_results_mean = compute_mean(disgust_results)
    df_disgust = pd.DataFrame({'Angle': angles_names,
                                'Value': disgust_results_mean})
    df_disgust = df_disgust.sort_values('Value')

    plt.bar('Angle', 'Value', data=df_disgust)
    plt.ylabel("Diferencia entre izquierda y derecha (grados)")
    plt.xlabel("Nombre de los ángulos")
    #plt.title("Angle difference of left/right face in disgust")
    plt.show()

    # Fear -------------------------------------------------------
    fear_results = compare_left_and_right(fear)
    fear_results_mean = compute_mean(fear_results)
    df_fear = pd.DataFrame({'Angle': angles_names,
                            'Value': fear_results_mean})
    df_fear = df_fear.sort_values('Value')

    plt.bar('Angle', 'Value', data=df_fear)
    plt.ylabel("Diferencia entre izquierda y derecha (grados)")
    plt.xlabel("Nombre de los ángulos")
    #plt.title("Angle difference of left/right face in fear")
    plt.show()

    # Happy -------------------------------------------------------
    happy_results = compare_left_and_right(happy)
    happy_results_mean = compute_mean(happy_results)
    df_happy = pd.DataFrame({'Angle': angles_names,
                            'Value': happy_results_mean})
    df_happy = df_happy.sort_values('Value')

    plt.bar('Angle', 'Value', data=df_happy)
    plt.ylabel("Diferencia entre izquierda y derecha (grados)")
    plt.xlabel("Nombre de los ángulos")
    #plt.title("Angle difference of left/right face in happy")
    plt.show()

    # Sadness -----------------------------------------------------
    sadness_results = compare_left_and_right(sadness)
    sadness_results_mean = compute_mean(sadness_results)
    df_sadness = pd.DataFrame({'Angle': angles_names,
                                'Value': sadness_results_mean})
    df_sadness = df_sadness.sort_values('Value')

    plt.bar('Angle', 'Value', data=df_sadness)
    plt.ylabel("Diferencia entre izquierda y derecha (grados)")
    plt.xlabel("Nombre de los ángulos")
    #plt.title("Angle difference of left/right face in sadness")
    plt.show()

    # Surprise -----------------------------------------------------
    surprise_results = compare_left_and_right(surprise)
    surprise_results_mean = compute_mean(surprise_results)
    df_surprise = pd.DataFrame({'Angle': angles_names,
                                'Value': surprise_results_mean})
    df_surprise = df_surprise.sort_values('Value')

    plt.bar('Angle', 'Value', data=df_surprise)
    plt.ylabel("Diferencia entre izquierda y derecha (grados)")
    plt.xlabel("Nombre de los ángulos")
    #plt.title("Angle difference of left/right face in surprise")
    plt.show()
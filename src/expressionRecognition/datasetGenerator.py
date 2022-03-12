import sys
from tkinter.font import families
sys.path.append('..')
from faceMesh.FaceMesh import FaceMesh
import cv2
import glob as gb
import numpy as np
import csv

# Lists of images path
angry = gb.glob('../../../dataset/train/angry/*.jpg')
fear = gb.glob('../../../dataset/train/fear/*.jpg')
happy = gb.glob('../../../dataset/train/happy/*.jpg')
neutral = gb.glob('../../../dataset/train/neutral/*.jpg')
sad = gb.glob('../../../dataset/train/sad/*.jpg')
surprise = gb.glob('../../../dataset/train/surprise/*.jpg')

# Open dataset file to write
f = open('dataset/dataset3D.csv', 'w', newline='')

# Init FaceMesh
facemesh = FaceMesh(static=False, max_num_faces=1, refine=True)

# Dataset file header
for i in range(1, facemesh.num_distances):
    f.write('X'+repr(i)+',')
f.write('y\n')

# Process images and write data in dataset file
not_process = 0 # Not process images
print("Procesando imagenes angry...")
for image in angry:
    facemesh.set_image(cv2.imread(image))
    facemesh.process(distances3D=True)
    distances = facemesh.get_distances()
    if facemesh.face_mesh_detected():
        distances[0][-1] = 0
        np.savetxt(f, distances, delimiter=",")
    else:
          not_process += 1

print("Procesando imagenes fear...")
for image in fear:
    facemesh.set_image(cv2.imread(image))
    facemesh.process(distances3D=True)
    distances = facemesh.get_distances()
    if facemesh.face_mesh_detected():
        distances[0][-1] = 1
        np.savetxt(f, distances, delimiter=",")
    else:
          not_process += 1

print("Procesando imagenes happy...")
for image in happy:
    facemesh.set_image(cv2.imread(image))
    facemesh.process(distances3D=True)
    distances = facemesh.get_distances()
    if facemesh.face_mesh_detected():
        distances[0][-1] = 2
        np.savetxt(f, distances, delimiter=",")
    else:
          not_process += 1

print("Procesando imagenes neutral...")
for image in neutral:
    facemesh.set_image(cv2.imread(image))
    facemesh.process(distances3D=True)
    distances = facemesh.get_distances()
    if facemesh.face_mesh_detected():
        distances[0][-1] = 3
        np.savetxt(f, distances, delimiter=",")
    else:
          not_process += 1

print("Procesando imagenes sad...")
for image in sad:
    facemesh.set_image(cv2.imread(image))
    facemesh.process(distances3D=True)
    distances = facemesh.get_distances()
    if facemesh.face_mesh_detected():
        distances[0][-1] = 4
        np.savetxt(f, distances, delimiter=",")
    else:
          not_process += 1

print("Procesando imagenes surprise...")
for image in surprise:
    facemesh.set_image(cv2.imread(image))
    facemesh.process(distances3D=True)
    distances = facemesh.get_distances()
    if facemesh.face_mesh_detected():
        distances[0][-1] = 5
        np.savetxt(f, distances, delimiter=",")
    else:
          not_process += 1

print("Imagenes no procesadas: "+str(not_process))


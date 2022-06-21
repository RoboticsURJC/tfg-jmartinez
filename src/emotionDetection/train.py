from random import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

# Data
data = pd.read_csv('dataset/emotionalMesh/dataset1_2CK+.csv')
data = shuffle(data, random_state = 0)
X = data.drop(columns = 'y')
y = data['y'].astype(int)

# PCA
pca = PCA(11)
X = pca.fit_transform(X)

start = time.time()

# SVM
""" model = SVC(C=11, kernel='rbf', gamma='scale')
model.fit(X, y.ravel()) """

# KNN
model = KNeighborsClassifier(n_neighbors=7, p=2, metric="minkowski")
model.fit(X, y.ravel())

# Multi layer perceptron
""" model = MLPClassifier(activation="relu", hidden_layer_sizes=(30,30,10), solver="adam", max_iter=5000)
model.fit(X, y.ravel()) """

stop = time.time()

# Save the model as a pickle in a file
with open('model/emotionalMesh/model.pkl', 'wb') as file:
    pickle.dump({
        'model': model,
        'pca_fit': pca
    }, file)

print(f"Training time: {(stop - start)*1000}ms")
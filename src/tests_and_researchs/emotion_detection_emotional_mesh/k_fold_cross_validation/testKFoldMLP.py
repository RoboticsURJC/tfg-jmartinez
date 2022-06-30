import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import unravel_index
import pickle
import time

from sklearn.svm import SVC
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

def test_values(neurons_values, X_train, X_test, y_train, y_test):
    accuracy_matrix_iteration = np.zeros((1, len(neurons_values)))
    for i in range(len(neurons_values)):
        # PCA
        pca = PCA()
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        # Multi layer perceptron
        model = MLPClassifier(activation="relu", hidden_layer_sizes=(neurons_values[i]), solver="adam", max_iter=5000)
        model.fit(X_train_pca, y_train.ravel())
        # Evaluation
        y_pred = model.predict(X_test_pca)
        score = accuracy_score(y_test, y_pred)
        accuracy_matrix_iteration[0][i] = score
    return accuracy_matrix_iteration

def mean_of_iterations(accuracy_matrix, num_neurons_values):
    accuracy_matrix_means = np.zeros((1, num_neurons_values))
    for i in range(num_neurons_values):
            accuracy_matrix_means[0][i] = np.mean(accuracy_matrix[:,i])
    return accuracy_matrix_means

# Data
data = pd.read_csv('../dataset/emotionalMesh/dataset2CK+.csv')
X = np.array(data.drop(columns = 'y'))
y = np.array(data['y'].astype(int))

# Values to test
# MLP neurons and layers
neurons_values = list(range(5, 25))

splits = 4
iteration_index = 0 # Index of splits in accuracy_matrix
accuracy_matrix = np.zeros((splits, len(neurons_values)))
skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    accuracy_matrix_iteration = test_values(neurons_values,
                                            X_train, X_test,
                                            y_train, y_test)
    accuracy_matrix[iteration_index] = accuracy_matrix_iteration
    iteration_index += 1

accuracy_matrix_means = mean_of_iterations(accuracy_matrix, len(neurons_values))
index = unravel_index(np.argmax(accuracy_matrix_means), accuracy_matrix_means.shape)
print("Neurons: "+str(neurons_values[index[1]]))
print("Accuracy mean: "+str(accuracy_matrix_means[0][index[1]]))
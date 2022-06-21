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

def test_values(k_values, X_train, X_test, y_train, y_test):
    accuracy_matrix_iteration = np.zeros((1, len(k_values)))
    for i in range(len(k_values)):
        # PCA
        pca = PCA()
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        # KNN
        model = KNeighborsClassifier(n_neighbors=k_values[i], p=2, metric="minkowski")
        model.fit(X_train_pca, y_train.ravel())
        # Evaluation
        y_pred = model.predict(X_test_pca)
        score = accuracy_score(y_test, y_pred)
        accuracy_matrix_iteration[0][i] = score
    return accuracy_matrix_iteration

def mean_of_iterations(accuracy_matrix, num_k_values):
    accuracy_matrix_means = np.zeros((1, num_k_values))
    for i in range(num_k_values):
            accuracy_matrix_means[0][i] = np.mean(accuracy_matrix[:,i])
    return accuracy_matrix_means

# Data
data = pd.read_csv('../dataset/emotionalMesh/dataset2CK+.csv')
X = np.array(data.drop(columns = 'y'))
y = np.array(data['y'].astype(int))

# Values to test
# KNN neighbours
k_values = [1, 3, 5, 7, 9, 11, 13]
# PCA n_components
pca_components = list(range(2, 21))

splits = 4
iteration_index = 0 # Index of splits in accuracy_matrix
accuracy_matrix = np.zeros((splits, len(k_values)))
skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    accuracy_matrix_iteration = test_values(k_values,
                                            X_train, X_test,
                                            y_train, y_test)
    accuracy_matrix[iteration_index] = accuracy_matrix_iteration
    iteration_index += 1

accuracy_matrix_means = mean_of_iterations(accuracy_matrix, len(k_values))
index = unravel_index(np.argmax(accuracy_matrix_means), accuracy_matrix_means.shape)
print("K neighbours: "+str(k_values[index[1]]))
print("Accuracy mean: "+str(accuracy_matrix_means[0][index[1]]))
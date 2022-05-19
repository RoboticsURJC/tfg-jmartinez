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

def test_split(k_values, pca_components, X_train, X_test, y_train, y_test):
    accuracy_matrix_split = np.zeros((len(pca_components), len(k_values)))
    for i in range(len(k_values)):
        for j in range(len(pca_components)):
            # PCA
            pca = PCA(pca_components[j])
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            # KNN
            model = KNeighborsClassifier(n_neighbors=k_values[i], p=2, metric="minkowski")
            model.fit(X_train_pca, y_train.ravel())
            # Evaluation
            y_pred = model.predict(X_test_pca)
            score = accuracy_score(y_test, y_pred)
            accuracy_matrix_split[j][i] = score
    return accuracy_matrix_split

def mean_of_splits(accuracy_matrix, num_k_values, num_pca_components):
    accuracy_matrix_means = np.zeros((num_pca_components, num_k_values))
    for i in range(num_k_values):
        for j in range(num_pca_components):
            accuracy_matrix_means[j][i] = np.mean(accuracy_matrix[:,j,i])
    return accuracy_matrix_means

def test_SVM_PCA_values(c_values, pca_components, X_train, X_test, y_train, y_test):
    max_accuracy_score = 0
    better_c = -1
    better_pc = -1
    for c in c_values:
        for pc in pca_components:
            # PCA
            pca = PCA(pc)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            # SVM
            model = SVC(C=c, kernel='rbf', gamma='scale')
            model.fit(X_train_pca, y_train.ravel())
            # Evaluation
            y_pred = model.predict(X_test_pca)
            score = accuracy_score(y_test, y_pred)
            if score > max_accuracy_score:
                max_accuracy_score = score
                better_c = c
                better_pc = pc
    return better_c, better_pc, max_accuracy_score

# Data
data = pd.read_csv('../dataset/emotionalMesh/dataset1_2CK+.csv')
X = np.array(data.drop(columns = 'y'))
y = np.array(data['y'].astype(int))

# Values to test
# KNN neighbours
k_values = [1, 3, 5, 7, 9, 11, 13]
# PCA n_components
pca_components = list(range(2, 21))

splits = 4
split_index = 0 # Index of splits in accuracy_matrix
accuracy_matrix = np.zeros((splits, len(pca_components), len(k_values)))
skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    accuracy_matrix_split = test_split(k_values, pca_components,
                                                X_train, X_test,
                                                y_train, y_test)
    accuracy_matrix[split_index] = accuracy_matrix_split
    split_index += 1

accuracy_matrix_means = mean_of_splits(accuracy_matrix, len(k_values), len(pca_components))
index = unravel_index(np.argmax(accuracy_matrix_means), accuracy_matrix_means.shape)
print("PCA components: "+str(pca_components[index[0]]))
print("K neighbours: "+str(k_values[index[1]]))
print("Accuracy mean: "+str(accuracy_matrix_means[index[0]][index[1]]))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.metrics import precision_recall_fscore_support

def mean_of_iterations(report_matrix, num_report_values):
    report_matrix_mean = np.zeros((1, num_report_values))
    for i in range(num_report_values):
            report_matrix_mean[0][i] = np.mean(report_matrix[:,i])
    return report_matrix_mean

# Data
data = pd.read_csv('../dataset/emotionalMesh/dataset1_2CK+.csv')

X = np.array(data.drop(columns = 'y'))
y = np.array(data['y'].astype(int))

splits = 4
num_report_values = 4
report_matrix = np.zeros((splits, num_report_values)) # splits x (precision, recall, f1-score, accuracy)
index_iteration = 0
skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # PCA
    pca = PCA(11)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # KNN
    model = KNeighborsClassifier(n_neighbors=7, p=2, metric="minkowski")
    model.fit(X_train_pca, y_train.ravel())

    # SVM
    """ model = SVC(C=21, kernel='rbf', gamma='scale', probability=True)
    model.fit(X_train_pca, y_train.ravel()) """

    # Multi layer perceptron
    """ model = MLPClassifier(activation="relu", hidden_layer_sizes=(17), solver="adam", max_iter=5000)
    model.fit(X_train_pca, y_train.ravel()) """

    y_pred = model.predict(X_test_pca)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_matrix[index_iteration][0] = report['macro avg']['precision']
    report_matrix[index_iteration][1] = report['macro avg']['recall']
    report_matrix[index_iteration][2] = report['macro avg']['f1-score']
    report_matrix[index_iteration][3] = report['accuracy']
    index_iteration += 1
    print(classification_report(y_test, y_pred))

report_matrix_mean = mean_of_iterations(report_matrix, num_report_values)
print("Precision: "+str(report_matrix_mean[0][0]))
print("Recall: "+str(report_matrix_mean[0][1]))
print("F1-score: "+str(report_matrix_mean[0][2]))
print("Accuracy: "+str(report_matrix_mean[0][3]))



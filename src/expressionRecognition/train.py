from math import gamma
import pandas as pd
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# Data
data = pd.read_csv('dataset/emotionalMesh/datasetCK+.csv')


# Suffle data and division in train and test
X = data.drop(columns = 'y')
y = data['y'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
                                      X.values,
                                      y.values.reshape(-1,1),
                                      train_size   = 0.8,
                                      random_state = 0,
                                      shuffle      = True
                                  )

# PCA
""" pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test) """

# SVM
model = SVC(C=275, kernel='rbf', gamma='scale')
model.fit(X_train, y_train.ravel())

# KNN
""" model = KNeighborsClassifier(n_neighbors=1, leaf_size=30, p=2, metric="minkowski", weights="uniform")
model.fit(X_train, y_train.ravel()) """

# Save the model as a pickle in a file
joblib.dump(model, 'model/model.pkl')

# Predictions and model evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

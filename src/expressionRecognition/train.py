import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Data
data = pd.read_csv('dataset/dataset.csv')

# Suffle data and division in train and test
X = data.drop(columns = 'y')
y = data['y'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
                                      X,
                                      y.values.reshape(-1,1),
                                      train_size   = 0.8,
                                      random_state = 1234,
                                      shuffle      = True
                                  )

# Train model
svclassifier = SVC(kernel='rbf', verbose=True)
svclassifier.fit(X_train, y_train.ravel())

# Predictions and model evaluation
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
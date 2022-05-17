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


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Function source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 2)

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[1].grid()
    axes[1].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[1].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[1].set_xlabel("fit_times")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Performance of the model")

    return plt

def test_KNN_PCA_values(k_values, pca_components, X_train, X_test, y_train, y_test):
    max_accuracy_score = 0
    better_k = -1
    better_pc = -1
    for k in k_values:
        for pc in pca_components:
            # PCA
            pca = PCA(pc)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            # KNN
            model = KNeighborsClassifier(n_neighbors=k, p=2, metric="minkowski")
            model.fit(X_train_pca, y_train.ravel())
            # Evaluation
            y_pred = model.predict(X_test_pca)
            score = accuracy_score(y_test, y_pred)
            if score > max_accuracy_score:
                max_accuracy_score = score
                better_k = k
                better_pc = pc
    return better_k, better_pc, max_accuracy_score

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
data = pd.read_csv('dataset/emotionalMesh/dataset1_2CK+.csv')
X = np.array(data.drop(columns = 'y'))
y = np.array(data['y'].astype(int))

# Values to test
# KNN
k_values = [1, 3, 5, 7, 9, 11, 13]
# SVM
c_values = list(range(1,1000,10))
# PCA
pca_components = list(range(2, 21))

skf = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    """ k, pca_component, score = test_KNN_PCA_values(k_values, pca_components,
                                                X_train, X_test,
                                                y_train, y_test) """

    c, pca_component, score = test_SVM_PCA_values(c_values, pca_components,
                                                X_train, X_test,
                                                y_train, y_test)
    
    print(c, pca_component, score)


# PCA
""" pca = PCA(15)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

start = time.time()

# SVM
model = SVC(C=300, kernel='rbf', gamma='scale')
model.fit(X_train, y_train.ravel())

# KNN
model = KNeighborsClassifier(n_neighbors=3, p=2, metric="minkowski")
model.fit(X_train, y_train.ravel())

# Multi layer perceptron
model = MLPClassifier(activation="relu", hidden_layer_sizes=(30,30,10), solver="adam", max_iter=5000)
model.fit(X_train, y_train.ravel())

stop = time.time()

# Save the model as a pickle in a file
with open('model/emotionalMesh/model.pkl', 'wb') as file:
    pickle.dump({
        'model': model,
        'pca_fit': pca
    }, file)

# Predictions and model evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Training time: {(stop - start)*1000}ms")

# Generate Learning Curves
title = "Learning Curves"
cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
plot_learning_curve(
    model, title, X, y, ylim=(0.5, 1.01), cv=cv, n_jobs=4
)

plt.show() """
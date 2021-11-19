import os
from time import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from feature_models import feature_model_A_1
import parameters as p

data_directory = "/home/denis/Desktop/CSYS 395B - Machine Learning/Project/" \
                 "Data/2019"


def main():
    features, targets = feature_model_A_1.get_data(data_directory)

    start_time = pd.Timestamp.now()
    print(f"Started Random Forest Classification at {start_time}...")
    random_forest_classification(features, targets)
    run_time = (pd.Timestamp.now()-start_time).total_seconds()/60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")

    # start_time = pd.Timestamp.now()
    # print(f"Started Gradient Boost Classification at {start_time}...")
    # gradient_boost_classification(features, targets)
    # run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    # print(f"Training Completed in {run_time:.2f} minutes.\n")

    # start_time = pd.Timestamp.now()
    # print(f"Started AdaBoost Classification at {start_time}...")
    # ada_boost_classification(features, targets)
    # run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    # print(f"Training Completed in {run_time:.2f} minutes.\n")

    # start_time = pd.Timestamp.now()
    # print(f"Started Neural Network Classification at {start_time}...")
    # neural_network_classification(features, targets)
    # run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    # print(f"Training Completed in {run_time:.2f} minutes.\n")


def random_forest_classification(features: np.ndarray,
                                 targets: np.ndarray):

    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=20_000)
    clf = RandomForestClassifier(n_jobs=4, verbose=1)

    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})"
                      .format(results['mean_test_score'][candidate],
                              results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    param_grid = {
        'n_estimators': [5, 10, 50, 100],
        'criterion': ['gini', 'entroy'],
        'max_depth': [None, 5, 10, 50, 100],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 20}],
        'max_samples': [None, 0.1, 0.5]
    }
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(X_train, y_train)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)

    # clf.fit(X_train, y_train)
    # predictions = clf.predict(X_test)
    # cm = confusion_matrix(y_test, predictions)
    # print("\nConfusion Matrix:")
    # print(cm)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()
    # print("\nClassification Report:")
    # print(classification_report(y_test, predictions))


def gradient_boost_classification(features: np.ndarray,
                                  targets: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(features, targets)
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def ada_boost_classification(features: np.ndarray,
                             targets: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(features, targets)
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def nearest_neighbors_classification(features: np.ndarray,
                                     targets: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(features, targets)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def neural_network_classification(features: np.ndarray,
                                     targets: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(features, targets)
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == '__main__':
    main()

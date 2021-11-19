import os
import pickle
from time import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    ConfusionMatrixDisplay, PrecisionRecallDisplay, \
    RocCurveDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV

import xgboost

from feature_models import feature_model_A_1, feature_model_D_1, \
    feature_model_D_2, feature_model_D_3, feature_model_D_4
import parameters as p

data_directory = p.DATA_DIRECTORY
parameter_data_directory = \
    os.path.join(p.PARAMETER_DATA_DIRECTORY, 'Basic XGBoost')
performance_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Basic XGBoost')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'basic_xgboost')


def main():
    # start_time = pd.Timestamp.now()
    # print(f"Started Parameter Grid Search at {start_time}...")
    # parameter_search(features, targets)
    # run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    # print(f"Grid Search Completed in {run_time:.2f} minutes.\n")

    start_time = pd.Timestamp.now()
    print(f"Started XGBoost Classification at {start_time}...")
    x_xgboost_classifier(
        feature_model_A_1,
        'basic_x_gradient_boost_fm_A_1',
        'Basic XGBoost, Feature Model A_1'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")

    start_time = pd.Timestamp.now()
    print(f"Started XGBoost Classification at {start_time}...")
    x_xgboost_classifier(
        feature_model_D_1,
        'basic_x_gradient_boost_fm_D_1',
        'Basic XGBoost, Feature Model D_1'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")

    start_time = pd.Timestamp.now()
    print(f"Started XGBoost Classification at {start_time}...")
    x_xgboost_classifier(
        feature_model_D_2,
        'basic_x_gradient_boost_fm_D_2',
        'Basic XGBoost, Feature Model D_2'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")

    start_time = pd.Timestamp.now()
    print(f"Started XGBoost Classification at {start_time}...")
    x_xgboost_classifier(
        feature_model_D_3,
        'basic_x_gradient_boost_fm_D_3',
        'Basic XGBoost, Feature Model D_3'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")

    start_time = pd.Timestamp.now()
    print(f"Started XGBoost Classification at {start_time}...")
    x_xgboost_classifier(
        feature_model_D_4,
        'basic_x_gradient_boost_fm_D_4',
        'Basic XGBoost, Feature Model D_4'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")


def x_xgboost_classifier(feature_model,
                                 filename: str,
                                 filename_pretty: str):
    """
    A basic, out of the box random forest classifier, using parameters
    optimized by a grid search algorithm.
    :param features:
    :param targets:
    :param filename:
    :param filename_pretty:
    :return:
    """
    print("Loading Data...")
    X_train, X_test, y_train, y_test = feature_model.get_data()
    clf = xgboost.XGBClassifier()
    print("Data loaded.\n")

    print("Fitting training data...")
    clf.fit(X_train, y_train)
    print("Fitting complete.\n")

    "Generating performance data..."
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=[1, 0])
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, labels=[1, 0]))
    print("\nArea Under ROC Curve:")
    print(1-roc_auc_score(y_test, clf.predict_proba(X_test)[:, 0]))

    print("\nSaving classifier...")
    destination = os.path.join(
        classifier_directory, filename
    )
    with open(destination, 'wb') as file:
        pickle.dump(clf, file)
    print(f"Classifier saved to {destination}\n")

    print("Generating precision recall figure...")
    display = PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test, pos_label=1
    )
    display.plot()
    destination = os.path.join(
        performance_data_directory, filename_pretty + ' Precision Recall Curve.png'
    )
    display.figure_.savefig(destination)
    print(f"Figure saved to {destination}\n")

    print("Generating ROC curve figure...")
    display = RocCurveDisplay.from_estimator(
        clf, X_test, y_test, pos_label=1
    )
    display.plot()
    plt.show()
    destination = os.path.join(
        performance_data_directory,
        performance_data_directory, filename_pretty + ' ROC Curve.png'
    )
    display.figure_.savefig(destination)
    print(f"Figure saved to {destination}\n")

    print("Generating confusion matrix figure...")
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
    display.plot()
    plt.show()
    destination = os.path.join(
        performance_data_directory, filename_pretty + ' Confusion Matrix.png'
    )
    display.figure_.savefig(destination)
    print(f"Figure saved to {destination}\n")

    print("All classifier training operations complete.")


def parameter_search(features: np.ndarray,
                     targets: np.ndarray,
                     filename: str):
    """
    Runs a parameter search on the basic random forest algorithm and prints
    and stores the results
    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=20_000)
    clf = RandomForestClassifier(n_jobs=5, verbose=1)

    # Utility function to report best scores
    def report(results, n_top=None):
        if n_top == None:
            n_top = len(results)
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
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 50],
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

    results = pd.DataFrame(grid_search.cv_results_)
    destination = os.path.join(
        parameter_data_directory, filename
    )
    with open(destination, 'wb') as file:
        pickle.dump(results, file)
    print("All classifier training operations complete.")


if __name__ == '__main__':
    main()

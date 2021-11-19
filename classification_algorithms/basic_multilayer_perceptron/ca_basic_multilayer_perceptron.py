import os
import pickle
from time import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    ConfusionMatrixDisplay, PrecisionRecallDisplay, \
    RocCurveDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from feature_models import feature_model_A_1, feature_model_D_1
import parameters as p

data_directory = p.DATA_DIRECTORY
parameter_data_directory = \
    os.path.join(p.PARAMETER_DATA_DIRECTORY, 'Basic Multilayer Perceptron')
performance_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Basic Multilayer Perceptron')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'basic_multilayer_perceptron')


def main():
    # start_time = pd.Timestamp.now()
    # print(f"Started Parameter Grid Search at {start_time}...")
    # parameter_search(features, targets)
    # run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    # print(f"Grid Search Completed in {run_time:.2f} minutes.\n")

    # features, targets = \
    #     feature_model_A_1.get_data(data_directory)
    # start_time = pd.Timestamp.now()
    # print(f"Started Random Forest Classification at {start_time}...")
    # multilayer_perceptron_classification(
    #     features, targets,
    #     'basic_multi_layer_perceptron_fm_A_1',
    #     'Basic Multi-Layer Perceptron Classification, Feature Model A_1'
    # )
    # run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    # print(f"Training Completed in {run_time:.2f} minutes.\n")

    features, targets = \
        feature_model_D_1.get_data(data_directory)
    start_time = pd.Timestamp.now()
    print(f"Started Random Forest Classification at {start_time}...")
    multilayer_perceptron_classification(
        features, targets,
        'basic_multi_layer_perceptron_fm_D_1',
        'Basic Multi-Layer Perceptron Classification, Feature Model D_1'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")


def multilayer_perceptron_classification(features: np.ndarray,
                                         targets: np.ndarray,
                                         filename: str,
                                         filename_pretty: str):
    """
    A basic, out of the box MLP classifier, using parameters
    optimized by a grid search algorithm.
    :param features:
    :param targets:
    :param filename:
    :param filename_pretty:
    :return:
    """
    print("Loading Data...")
    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=20_000)
    clf = MLPClassifier(
        verbose=1, solver='adam',
        learning_rate='adaptive', max_iter=2000,
        n_iter_no_change=20, hidden_layer_sizes=(200, 200, 200)
    )
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
    print(1 - roc_auc_score(y_test, clf.predict_proba(X_test)[:, 0]))

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
        performance_data_directory,
        filename_pretty + ' Precision Recall Curve.png'
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
    Runs a parameter search on the basic multi-layer perceptron algorithm and
    prints and stores the results
    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=20_000)
    clf = MLPClassifier(max_iter=2000, verbose=1)

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
        'hidden_layer_sizes': [(100,),
                               (100, 100),
                               (100, 100, 100),
                               (200, 200, 200)],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'solver': ['sgd', 'adam']
    }
    grid_search = GridSearchCV(clf, param_grid=param_grid, verbose=1)
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

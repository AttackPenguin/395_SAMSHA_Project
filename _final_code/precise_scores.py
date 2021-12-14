from __future__ import annotations

import copy
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    ConfusionMatrixDisplay, PrecisionRecallDisplay, \
    RocCurveDisplay, roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import parameters as p

data_directory = p.DATA_DIRECTORY
performance_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Final Code')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'final_code')

def main():

    start_time = pd.Timestamp.now()
    print(f"Started at {start_time}")

    ############################################################################

    destination = os.path.join(
        classifier_directory, "rf_prep_01"
    )
    with open(destination, 'rb') as file:
        preprocessing, rf_clf, feature_labels, \
        X_train, X_test, y_train, y_test \
            = pickle.load(file)

    t_X_test = preprocessing.transform(X_test)
    y_test_pred = rf_clf.predict_proba(t_X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test, y_test_pred
    )

    print('01')
    print('roc_auc:', roc_auc)

    ############################################################################

    destination = os.path.join(
        classifier_directory, "rf_prep_01_opt"
    )
    with open(destination, 'rb') as file:
        preprocessing, rf_clf, feature_labels, X_train, X_test, y_train, y_test \
            = pickle.load(file)

    t_X_test = preprocessing.transform(X_test)
    y_test_pred = rf_clf.predict_proba(t_X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test, y_test_pred
    )

    print('01_opt')
    print('roc_auc:', roc_auc)

    ############################################################################

    destination = os.path.join(
        classifier_directory, "rf_prep_02a"
    )
    with open(destination, 'rb') as file:
        preprocessing, rf_clf, feature_labels, X_train, X_test, y_train, y_test \
            = pickle.load(file)

    t_X_test = preprocessing.transform(X_test)
    y_test_pred = rf_clf.predict_proba(t_X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test, y_test_pred
    )

    print('02a')
    print('roc_auc:', roc_auc)

    ############################################################################

    destination = os.path.join(
        classifier_directory, "rf_prep_02b"
    )
    with open(destination, 'rb') as file:
        preprocessing, rf_clf, feature_labels, X_train, X_test, y_train, y_test \
            = pickle.load(file)

    t_X_test = preprocessing.transform(X_test)
    y_test_pred = rf_clf.predict_proba(t_X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test, y_test_pred
    )

    print('01')
    print('roc_auc:', roc_auc)

    ############################################################################

    end_time = pd.Timestamp.now()
    print(f"Finished at {end_time}")
    print(f"Run time {(end_time - start_time).total_seconds() / 60:.2f} "
          f"minutes.")




if __name__ == '__main__':
    main()
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
    RocCurveDisplay, roc_auc_score, make_scorer, accuracy_score, \
    precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from feature_models import feature_model_A_1, feature_model_D_1, \
    feature_model_D_101, feature_model_D_102, \
    feature_model_D_2, feature_model_D_3, feature_model_D_4, \
    feature_model_D_1_1
import parameters as p

data_directory = p.DATA_DIRECTORY
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'downsampled_random_forest')


def main():
    print(f"Start Time: {pd.Timestamp.now()}")

    clf_loc = os.path.join(classifier_directory,
                           'down_sampled_random_forest_fm_D_102')
    with open(clf_loc, 'rb') as file:
        clf = pickle.load(file)  # type: RandomForestClassifier

    X_train, _, _, _ = feature_model_D_102.get_data()
    feature_importances = clf.feature_importances_
    data = list(zip(X_train.columns, feature_importances))
    data.sort(key=lambda x: x[1], reverse=True)
    for feature, importance in data:
        print(feature, ',', importance)

    print(f"Finish Time: {pd.Timestamp.now()}")


if __name__ == '__main__':
    main()

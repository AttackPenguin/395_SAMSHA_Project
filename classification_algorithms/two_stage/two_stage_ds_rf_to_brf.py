import pickle
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    ConfusionMatrixDisplay, PrecisionRecallDisplay, \
    RocCurveDisplay, roc_auc_score
from sklearn.neural_network import MLPClassifier

from feature_models import feature_model_A_1
import parameters as p

data_directory = p.DATA_DIRECTORY
parameter_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Two Stage')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'two_stage')


def main():
    # features, targets = \
    #     feature_model_A_1.get_data(data_directory)
    #
    # start_time = pd.Timestamp.now()
    # print(f"Started Random Forest Classification at {start_time}...")
    # two_stage_ds_rf_to_brf_existing_clf(
    #     features, targets,
    #     'two_stage_ds_rf_to_brf_fm_A_1',
    #     'Two Stage Classification, Downsampled RF to Basic RF, '
    #     'Feature Model A_1',
    #     os.path.join(p.CLASSIFIER_DIRECTORY,
    #                  'downsampled_random_forest',
    #                  'down_sampled_random_forest_fm_A_1'),
    #     os.path.join(p.CLASSIFIER_DIRECTORY,
    #                  'basic_random_forest',
    #                  'basic_random_forest_fm_A_1')
    # )
    # run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    # print(f"Training Completed in {run_time:.2f} minutes.\n")

    features, targets = \
        feature_model_A_1.get_data(data_directory)

    clf_2 = MLPClassifier(
        hidden_layer_sizes=(200, 200, 200),
        verbose=1, solver='adam',
        learning_rate='adaptive', max_iter=500,
        tol=0.0001, n_iter_no_change=20
    )

    start_time = pd.Timestamp.now()
    print(f"Started Random Forest Classification at {start_time}...")
    two_stage_ds_rf_to_new_mlp(
        features, targets,
        'two_stage_ds_rf_to_brf_fm_A_1',
        'Two Stage Classification, Downsampled RF to Basic RF, '
        'Feature Model A_1',
        os.path.join(p.CLASSIFIER_DIRECTORY,
                     'downsampled_random_forest',
                     'down_sampled_random_forest_fm_A_1'),
        clf_2
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")


def two_stage_ds_rf_to_brf_existing_clf(features: np.ndarray,
                                        targets: np.ndarray,
                                        filename: str,
                                        filename_pretty: str,
                                        stage_1_path: str,
                                        stage_2_path: str):
    print("Loading classifiers...")
    with open(stage_1_path, 'rb') as file:
        clf_1 = pickle.load(file)
    with open(stage_2_path, 'rb') as file:
        clf_2 = pickle.load(file)
    print("Classifiers loaded.\n")

    print("Loading Data...")
    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=20_000)
    print("Data loaded.\n")

    print("Generating performance data...")
    predictions_1 = clf_1.predict(X_test)
    cm_1 = confusion_matrix(y_test, predictions_1, labels=[1, 0])
    print("\nStage I Confusion Matrix:")
    print(cm_1)

    X_test_2 = list()
    y_test_2 = list()
    for i in range(len(predictions_1)):
        if predictions_1[i] == 1:
            X_test_2.append(X_test[i])
            y_test_2.append(y_test[i])
    X_test_2 = np.array(X_test_2)
    y_test_2 = np.array(y_test_2)
    predictions_2 = clf_2.predict(X_test_2)
    cm_2 = confusion_matrix(y_test_2, predictions_2, labels=[1, 0])
    print("\nStage II Confusion Matrix:")
    print(cm_2)

    true_pos = cm_2[0][0]
    false_pos = cm_2[1][0]
    true_neg = cm_1[1][1] + cm_2[1][1]
    false_neg = cm_1[0][1] + cm_2[0][1]
    cm_3 = np.array([[true_pos, false_neg], [false_pos, true_neg]])
    print("\n\nCombined Confusion Matrix:")
    print(cm_3)
    # print("\nClassification Report:")
    # print(classification_report(y_test, predictions, labels=[1, 0]))
    # print("\nArea Under ROC Curve:")
    # print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 0]))

    # print("Generating precision recall figure...")
    # display = PrecisionRecallDisplay.from_estimator(
    #     clf, X_test, y_test, pos_label=1
    # )
    # display.plot()
    # destination = os.path.join(
    #     performance_data_directory,
    #     f"ca_ds_le_fm_B_2 "
    #     f"Precision Recall Curve.png"
    # )
    # display.figure_.savefig(destination)
    # print(f"Figure saved to {destination}\n")
    #
    # print("Generating ROC curve figure...")
    # display = RocCurveDisplay.from_estimator(
    #     clf, X_test, y_test, pos_label=1
    # )
    # display.plot()
    # plt.show()
    # destination = os.path.join(
    #     performance_data_directory,
    #     f"ca_ds_le_fm_B_2 "
    #     f"ROC Curve.png"
    # )
    # display.figure_.savefig(destination)
    # print(f"Figure saved to {destination}\n")
    #
    # print("Generating confusion matrix figure...")
    # display = ConfusionMatrixDisplay(confusion_matrix=cm_1, display_labels=[1, 0])
    # display.plot()
    # plt.show()
    # destination = os.path.join(
    #     performance_data_directory,
    #     f"ca_ds_le_fm_B_2 "
    #     f"Confusion Matrix.png"
    # )
    # display.figure_.savefig(destination)
    # print(f"Figure saved to {destination}\n")
    #
    # print("All classifier training operations complete.")


def two_stage_ds_rf_to_new_mlp(features: np.ndarray,
                               targets: np.ndarray,
                               filename: str,
                               filename_pretty: str,
                               stage_1_path: str,
                               clf_2):
    print("Loading classifiers...")
    with open(stage_1_path, 'rb') as file:
        clf_1 = pickle.load(file)
    print("Classifiers loaded.\n")

    print("Loading Data...")
    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=20_000)
    print("Data loaded.\n")

    print("Generating stage I training data...")
    predictions_train = clf_1.predict(X_train)

    print("Training second stage classifier...")
    X_train_2 = list()
    y_train_2 = list()
    for i in range(len(predictions_train)):
        if predictions_train[i] == 1:
            X_train_2.append(X_train[i])
            y_train_2.append(y_train[i])
    X_train_2 = np.array(X_train_2)
    y_train_2 = np.array(y_train_2)
    clf_2.fit(X_train_2, y_train_2)

    print("Generating stage I performance data...")
    predictions_1 = clf_1.predict(X_test)
    cm_1 = confusion_matrix(y_test, predictions_1, labels=[1, 0])
    print("\nStage I Confusion Matrix:")
    print(cm_1, '\n')

    print("Generating stage II performance data...")
    X_test_2 = list()
    y_test_2 = list()
    for i in range(len(predictions_1)):
        if predictions_1[i] == 1:
            X_test_2.append(X_test[i])
            y_test_2.append(y_test[i])
    X_test_2 = np.array(X_test_2)
    y_test_2 = np.array(y_test_2)
    predictions_2 = clf_2.predict(X_test_2)
    cm_2 = confusion_matrix(y_test_2, predictions_2, labels=[1, 0])
    print("\nStage II Confusion Matrix:")
    print(cm_2, '\n')

    print("Generating overall performance data...")
    true_pos = cm_2[0][0]
    false_pos = cm_2[1][0]
    true_neg = cm_1[1][1] + cm_2[1][1]
    false_neg = cm_1[0][1] + cm_2[0][1]
    cm_3 = np.array([[true_pos, false_neg], [false_pos, true_neg]])
    print("\n\nCombined Confusion Matrix:")
    print(cm_3)
    # print("\nClassification Report:")
    # print(classification_report(y_test, predictions, labels=[1, 0]))
    # print("\nArea Under ROC Curve:")
    # print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 0]))

    # print("Generating precision recall figure...")
    # display = PrecisionRecallDisplay.from_estimator(
    #     clf, X_test, y_test, pos_label=1
    # )
    # display.plot()
    # destination = os.path.join(
    #     performance_data_directory,
    #     f"ca_ds_le_fm_B_2 "
    #     f"Precision Recall Curve.png"
    # )
    # display.figure_.savefig(destination)
    # print(f"Figure saved to {destination}\n")
    #
    # print("Generating ROC curve figure...")
    # display = RocCurveDisplay.from_estimator(
    #     clf, X_test, y_test, pos_label=1
    # )
    # display.plot()
    # plt.show()
    # destination = os.path.join(
    #     performance_data_directory,
    #     f"ca_ds_le_fm_B_2 "
    #     f"ROC Curve.png"
    # )
    # display.figure_.savefig(destination)
    # print(f"Figure saved to {destination}\n")
    #
    # print("Generating confusion matrix figure...")
    # display = ConfusionMatrixDisplay(confusion_matrix=cm_1, display_labels=[1, 0])
    # display.plot()
    # plt.show()
    # destination = os.path.join(
    #     performance_data_directory,
    #     f"ca_ds_le_fm_B_2 "
    #     f"Confusion Matrix.png"
    # )
    # display.figure_.savefig(destination)
    # print(f"Figure saved to {destination}\n")
    #
    # print("All classifier training operations complete.")


if __name__ == '__main__':
    main()

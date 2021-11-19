import pickle
from statistics import mode
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    ConfusionMatrixDisplay, PrecisionRecallDisplay, \
    RocCurveDisplay, roc_auc_score

from feature_models import feature_model_A_1
import parameters as p

data_directory = p.DATA_DIRECTORY
parameter_data_directory = \
    os.path.join(p.PARAMETER_DATA_DIRECTORY, 'Downsampled Random Forest')
performance_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Downsampled Random Forest')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'downsampled_random_forest')


def main():
    print("Loading and setting up data...")
    features, targets = feature_model_A_1.get_data(data_directory)
    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=20_000)
    df = pd.DataFrame(X_train)
    df['targets'] = y_train
    df_majority = df[df.targets == 0]
    df_minority = df[df.targets == 1]
    min_size = df_minority.shape[0]  # // 2
    print("Data loaded.")

    print("Starting training...")
    rounds_of_training = 100
    clf = RandomForestClassifier(
        n_jobs=10, verbose=1,
        n_estimators=10, criterion='entropy',
        max_depth=None, max_features='sqrt',
        max_samples=None, class_weight=None,
        warm_start=True
    )
    for i in range(rounds_of_training):
        sample_majority = df_majority.sample(min_size)
        sample_minority = df_minority.sample(min_size)
        batch_data = pd.concat((sample_minority, sample_majority),
                               ignore_index=True)
        batch_y_train = np.array(batch_data.pop('targets'))
        batch_X_train = np.array(batch_data)
        clf.fit(batch_X_train, batch_y_train)
        clf.n_estimators += 10
    print("Training complete.")

    "Generating performance data..."
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=[1, 0])
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, labels=[1, 0]))
    print("\nArea Under ROC Curve:")
    print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 0]))

    print("\nSaving classifier...")
    destination = os.path.join(
        classifier_directory,
        f"ca_ds_le_fm_B_2"
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
        f"ca_ds_le_fm_B_2 "
        f"Precision Recall Curve.png"
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
        f"ca_ds_le_fm_B_2 "
        f"ROC Curve.png"
    )
    display.figure_.savefig(destination)
    print(f"Figure saved to {destination}\n")

    print("Generating confusion matrix figure...")
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
    display.plot()
    plt.show()
    destination = os.path.join(
        performance_data_directory,
        f"ca_ds_le_fm_B_2 "
        f"Confusion Matrix.png"
    )
    display.figure_.savefig(destination)
    print(f"Figure saved to {destination}\n")

    print("All classifier training operations complete.")


if __name__ == '__main__':
    main()

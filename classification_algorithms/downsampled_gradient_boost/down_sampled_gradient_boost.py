import pickle
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    ConfusionMatrixDisplay, PrecisionRecallDisplay, \
    RocCurveDisplay, roc_auc_score

from feature_models import feature_model_A_1, feature_model_D_1, \
    feature_model_D_2, feature_model_D_3, feature_model_D_4
import parameters as p

data_directory = p.DATA_DIRECTORY
parameter_data_directory = \
    os.path.join(p.PARAMETER_DATA_DIRECTORY, 'Downsampled Gradient Boost')
performance_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Downsampled Gradient Boost')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'downsampled_gradient_boost')


def main():
    start_time = pd.Timestamp.now()
    print(f"Started Gradient Boost Classification at {start_time}...")
    downsampled_gradient_boost(
        feature_model_A_1,
        'downsampled_gradient_boost_fm_A_1',
        'Downsampled Gradient Boost, Feature Model A_1'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")

    start_time = pd.Timestamp.now()
    print(f"Started Gradient Boost Classification at {start_time}...")
    downsampled_gradient_boost(
        feature_model_D_1,
        'downsampled_gradient_boost_fm_D_1',
        'Downsampled Gradient Boost, Feature Model D_1'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")

    start_time = pd.Timestamp.now()
    print(f"Started Gradient Boost Classification at {start_time}...")
    downsampled_gradient_boost(
        feature_model_D_2,
        'downsampled_gradient_boost_fm_D_2',
        'Downsampled Gradient Boost, Feature Model D_2'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")

    start_time = pd.Timestamp.now()
    print(f"Started Gradient Boost Classification at {start_time}...")
    downsampled_gradient_boost(
        feature_model_D_3,
        'downsampled_gradient_boost_fm_D_3',
        'Downsampled Gradient Boost, Feature Model D_3'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")

    start_time = pd.Timestamp.now()
    print(f"Started Gradient Boost Classification at {start_time}...")
    downsampled_gradient_boost(
        feature_model_D_4,
        'downsampled_gradient_boost_fm_D_4',
        'Downsampled Gradient Boost, Feature Model D_4'
    )
    run_time = (pd.Timestamp.now() - start_time).total_seconds() / 60.0
    print(f"Training Completed in {run_time:.2f} minutes.\n")


def downsampled_gradient_boost(feature_model,
                               filename: str,
                               filename_pretty: str):
    """
    Compensates for dramatic under-representation of desired target in data
    by breaking training data into two groups, based on target, then batch
    training on batches with equal representation of targets.
    :param features:
    :param targets:
    :param filename:
    :param filename_pretty:
    :return:
    """
    print("Setting up data...")
    X_train, X_test, y_train, y_test = feature_model.get_data()
    # Reconstruct data into single data frame so that we can split based on
    # target data, then split on target data.
    df = pd.DataFrame(X_train)
    df['targets'] = y_train
    df_majority = df[df.targets == 0]
    df_minority = df[df.targets == 1]
    # We will use the size of our underrepresented data as the size of
    # samples to pull from our overrepresented data for training.
    min_size = df_minority.shape[0]
    print("Data loaded.")

    print("Starting training...")
    rounds_of_training = 25
    clf = GradientBoostingClassifier(
        verbose=1,
        n_estimators=0, learning_rate=0.01,
        criterion='friedman_mse', max_depth=5,
        warm_start=True
    )
    for i in range(rounds_of_training):
        # Each time we train, we will randomly sample batches from our
        # training data, and add ten new trees to our Gradient Boost based on
        # that data.
        clf.n_estimators += 10
        sample_majority = df_majority.sample(min_size)
        sample_minority = df_minority.sample(min_size)
        batch_data = pd.concat((sample_minority, sample_majority),
                               ignore_index=True)
        batch_y_train = np.array(batch_data.pop('targets'))
        batch_X_train = np.array(batch_data)
        clf.fit(batch_X_train, batch_y_train)
    print("Training complete.")

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


if __name__ == '__main__':
    main()

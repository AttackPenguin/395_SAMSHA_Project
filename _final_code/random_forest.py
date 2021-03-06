import pickle
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    ConfusionMatrixDisplay, PrecisionRecallDisplay, \
    RocCurveDisplay, roc_auc_score

import parameters as p
import preprocessors
from subsampling import get_subsample

data_directory = p.DATA_DIRECTORY
performance_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Final Code', 'Preprocessor 01')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'final_code')


def main():
    subset_training()


def subset_training():
    start_time = pd.Timestamp.now()
    print(f"Started at {start_time}")

    ############################################################################

    target_file = os.path.join(data_directory, 'tedsd_puf_2019.csv')
    raw_data = pd.read_csv(target_file)

    ############################################################################

    # preprocessor, feature_labels = preprocessors.create_preprocessor_01()
    # downsampled_random_forest(preprocessor,
    #                           feature_labels,
    #                           'rf_prep_01',
    #                           'Random Forest, Preprocessor 01',
    #                           X_train, X_test, y_train, y_test)

    preprocessor, feature_labels = preprocessors.create_preprocessor_01_opt()
    subsample = get_subsample(raw_data.copy(), feature_labels, 3)
    targets = subsample.pop('REASON')
    targets = np.array((targets == 3).astype(int))
    X_train, X_test, y_train, y_test = \
        train_test_split(subsample, targets,
                         test_size=0.1, stratify=targets)
    random_forest(preprocessor,
                  feature_labels,
                              'rf_prep_01_opt_subsample',
                              'Random Forest, Preprocessor 01 Optimized, '
                              'Subsampled Data, Cutoff of 3',
                  X_train, X_test, y_train, y_test,
                  trees=1000)

    preprocessor, feature_labels = preprocessors.create_preprocessor_02a()
    subsample = get_subsample(raw_data.copy(), feature_labels, 3)
    targets = subsample.pop('REASON')
    targets = np.array((targets == 3).astype(int))
    X_train, X_test, y_train, y_test = \
        train_test_split(subsample, targets,
                         test_size=0.1, stratify=targets)
    random_forest(preprocessor,
                  feature_labels,
                              'rf_prep_02a_subsample',
                              'Random Forest, Preprocessor 02a, '
                              'Subsampled Data, Cutoff of 3',
                  X_train, X_test, y_train, y_test)

    preprocessor, feature_labels = preprocessors.create_preprocessor_02b()
    subsample = get_subsample(raw_data.copy(), feature_labels, 3)
    targets = subsample.pop('REASON')
    targets = np.array((targets == 3).astype(int))
    X_train, X_test, y_train, y_test = \
        train_test_split(subsample, targets,
                         test_size=0.1, stratify=targets)
    random_forest(preprocessor,
                  feature_labels,
                              'rf_prep_02b_subsample',
                              'Random Forest, Preprocessor 02b, '
                              'Subsampled Data, Cutoff of 3',
                  X_train, X_test, y_train, y_test)

    end_time = pd.Timestamp.now()
    print(f"Finished at {end_time}")
    print(f"Run time {(end_time - start_time).total_seconds() / 60:.2f} "
          f"minutes.")


def initial_training():
    start_time = pd.Timestamp.now()
    print(f"Started at {start_time}")

    ############################################################################

    # target_file = os.path.join(data_directory, 'tedsd_puf_2019.csv')
    # raw_data = pd.read_csv(target_file)
    #
    # targets = raw_data.pop('REASON')
    # targets = np.array((targets == 3).astype(int))
    # features = raw_data
    #
    # X_train, X_test, y_train, y_test = \
    #     train_test_split(features, targets,
    #                      test_size=0.1, stratify=targets)
    #
    # save_data = (X_train, X_test, y_train, y_test)
    # destination = os.path.join(
    #     classifier_directory, 'Split Data, Round 1.pickle'
    # )
    # with open(destination, 'wb') as file:
    #     pickle.dump(save_data, file)

    ############################################################################

    destination = os.path.join(
        classifier_directory, "Split Data, Round 1.pickle"
    )
    with open(destination, 'rb') as file:
        X_train, X_test, y_train, y_test = pickle.load(file)

    ############################################################################

    # preprocessor, feature_labels = preprocessors.create_preprocessor_01()
    # downsampled_random_forest(preprocessor,
    #                           feature_labels,
    #                           'rf_prep_01',
    #                           'Random Forest, Preprocessor 01',
    #                           X_train, X_test, y_train, y_test)

    preprocessor, feature_labels = preprocessors.create_preprocessor_01_opt()
    random_forest(preprocessor,
                  feature_labels,
                              'rf_prep_01_opt',
                              'Random Forest, Preprocessor 01 Optimized',
                  X_train, X_test, y_train, y_test)

    preprocessor, feature_labels = preprocessors.create_preprocessor_02a()
    random_forest(preprocessor,
                  feature_labels,
                              'rf_prep_02a',
                              'Random Forest, Preprocessor 02a',
                  X_train, X_test, y_train, y_test)

    preprocessor, feature_labels = preprocessors.create_preprocessor_02b()
    random_forest(preprocessor,
                  feature_labels,
                              'rf_prep_02b',
                              'Random Forest, Preprocessor 02b',
                  X_train, X_test, y_train, y_test)

    end_time = pd.Timestamp.now()
    print(f"Finished at {end_time}")
    print(f"Run time {(end_time - start_time).total_seconds() / 60:.2f} "
          f"minutes.")


def random_forest(preprocessor: ColumnTransformer,
                  feature_labels: list[str],
                  filename: str,
                  filename_pretty: str,
                  X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.DataFrame,
                  y_test: pd.DataFrame,
                  trees: int = 1000):
    """
    :param trees:
    :param y_test:
    :param y_train:
    :param X_test:
    :param X_train:
    :param preprocessor:
    :param feature_labels:
    :param filename:
    :param filename_pretty:
    :return:
    """

    t_X_train = preprocessor.fit_transform(X_train)
    t_X_test = preprocessor.transform(X_test)

    print("Data loaded.")

    print("Starting training...")
    clf = RandomForestClassifier(
        n_jobs=10, verbose=1,
        n_estimators=trees, criterion='entropy',
        max_depth=None, max_features='sqrt',
        max_samples=None, class_weight='balanced_subsample'
    )

    clf.fit(t_X_train, y_train)

    "Generating performance data..."
    predictions = clf.predict(t_X_test)
    cm = confusion_matrix(y_test, predictions, labels=[1, 0])
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, labels=[1, 0]))
    print("\nArea Under ROC Curve:")
    print(1 - roc_auc_score(y_test, clf.predict_proba(t_X_test)[:, 0]))

    print("Generating precision recall figure...")
    display = PrecisionRecallDisplay.from_estimator(
        clf, t_X_test, y_test, pos_label=1
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
        clf, t_X_test, y_test, pos_label=1
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

    save_data = (preprocessor, clf, feature_labels,
                 X_train, X_test, y_train, y_test)

    destination = os.path.join(
        classifier_directory, filename
    )
    with open(destination, 'wb') as file:
        pickle.dump(save_data, file)

    print("All classifier training operations complete.")


if __name__ == '__main__':
    main()

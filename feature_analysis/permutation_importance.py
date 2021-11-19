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
    RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import parameters as p

data_directory = p.DATA_DIRECTORY
performance_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Permutation Importance')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'permutation_importance')


def main():
    train_classifier_full_feature_set(
        get_2019_samsha_data(data_directory),
        "Classifier 1, Full Data Set, Full Feature Set"
    )
    permutation_analysis_full_feature_set(
        "Classifier 1, Full Data Set, Full Feature Set"
    )


def permutation_analysis_full_feature_set(file_name: str):

    # # Get categorical data that is binary
    # binary_category_cols = [
    #     'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG',
    #     'METHFLG', 'OPSYNFLG', 'MTHAMFLG', 'BENZFLG'
    # ]
    #
    # # Get categorical data with no missing datapoints:
    # complete_category_cols = [
    #     'SERVICES', 'ALCDRUG'
    # ]
    #
    # # Get categorical data with missing datapoints
    # incomplete_category_cols = [
    #     'GENDER', 'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'EMPLOY_D',
    #     'DETNLF', 'DETNLF_D', 'PREG', 'VET', 'LIVARAG', 'LIVARAG_D', 'PRIMINC',
    #     'ARRESTS', 'ARRESTS_D', 'PSOURCE', 'DETCRIM', 'NOPRIOR', 'DSMCRIT',
    #     'PSYPROB', 'HLTHINS', 'PRIMPAY', 'METHUSE', 'IDU'
    # ]
    #
    # incomplete_ordinal_cols = [
    #     'DAYWAIT', 'FREQ_ATND_SELF_HELP', 'FREQ_ATND_SELF_HELP_D'
    # ]
    #
    # complete_ordinal_cols = [
    #     'AGE'
    # ]

    destination = os.path.join(
        classifier_directory, file_name
    )
    with open(destination, 'rb') as file:
        preprocessing, rf_clf, X_train, X_test, y_train, y_test \
            = pickle.load(file)

    clf = Pipeline(
        [
            ('preprocess', preprocessing),
            ('classifier', rf_clf)
        ]
    )

    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=[1, 0])
    print(cm)
    print(classification_report(y_test, predictions, labels=[1, 0]))

    display = PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test, pos_label=1
    )
    display.plot()
    destination = os.path.join(
        performance_data_directory,
        file_name + ' Precision Recall Curve.png'
    )
    display.figure_.savefig(destination)

    display = RocCurveDisplay.from_estimator(
        clf, X_test, y_test, pos_label=1
    )
    display.plot()
    plt.show()
    destination = os.path.join(
        performance_data_directory,
        file_name + ' ROC Curve.png'
    )
    display.figure_.savefig(destination)

    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
    display.plot()
    plt.show()
    destination = os.path.join(
        performance_data_directory,
        file_name + ' Confusion Matrix.png'
    )
    display.figure_.savefig(destination)

    result = permutation_importance(
        clf, X_test, y_test, n_repeats=50, n_jobs=10
    )
    sorted_idx = result.importances_mean.argsort()

    fig: plt.Figure = plt.figure(figsize=[8.5, 11], dpi=400)
    ax: plt.Axes = fig.add_subplot()
    ax.boxplot(
        result.importances[sorted_idx].T, vert=False,
        labels=X_test.columns[sorted_idx]
    )
    ax.set_title("Permutation Feature Importance")
    fig.tight_layout()
    plt.show()
    destination = os.path.join(
        performance_data_directory,
        file_name + ' Confusion Matrix.png'
    )
    display.figure_.savefig(destination)


def train_classifier_full_feature_set(raw_data: pd.DataFrame,
                                      file_name: str):
    """
    Initial classifier, without pruning of low-value features.
    :param file_name:
    :param raw_data:
    :return:
    """

    # Get categorical data that is binary
    binary_category_cols = [
        'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG',
        'METHFLG', 'OPSYNFLG', 'MTHAMFLG', 'BENZFLG'
    ]
    binary_category_data = raw_data[binary_category_cols]

    # Get categorical data with no missing datapoints:
    complete_category_cols = [
        'SERVICES', 'ALCDRUG'
    ]
    complete_category_data = raw_data[complete_category_cols]

    # Get categorical data with missing datapoints
    incomplete_category_cols = [
        'GENDER', 'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'EMPLOY_D',
        'DETNLF', 'DETNLF_D', 'PREG', 'VET', 'LIVARAG', 'LIVARAG_D', 'PRIMINC',
        'ARRESTS', 'ARRESTS_D', 'PSOURCE', 'DETCRIM', 'NOPRIOR', 'DSMCRIT',
        'PSYPROB', 'HLTHINS', 'PRIMPAY', 'METHUSE', 'IDU'
    ]
    incomplete_category_data = raw_data[incomplete_category_cols]

    incomplete_ordinal_cols = [
        'DAYWAIT', 'FREQ_ATND_SELF_HELP', 'FREQ_ATND_SELF_HELP_D'
    ]
    incomplete_ordinal_data = raw_data[incomplete_ordinal_cols]

    complete_ordinal_cols = [
        'AGE'
    ]
    complete_ordinal_data = raw_data[complete_ordinal_cols]

    features = binary_category_data.join([
        complete_category_data,
        incomplete_category_data,
        incomplete_ordinal_data,
        complete_ordinal_data
    ])

    targets = get_targets(raw_data)

    preprocessing = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop='if_binary'),
             binary_category_cols),
            ('complete', OneHotEncoder(),
             complete_category_cols),
            ('inc_cat',
             OneHotEncoder(drop=[-9] * len(incomplete_category_cols)),
             incomplete_category_cols),
            ('inc_ord', OneHotEncoder(drop=[-9] * len(incomplete_ordinal_cols)),
             incomplete_ordinal_cols),
            ('cmp_ord', OrdinalEncoder(),
             complete_ordinal_cols)
        ], verbose=True, n_jobs=10
    )

    rf_clf = RandomForestClassifier(
        n_jobs=10, verbose=3,
        n_estimators=10, criterion='entropy',
        max_depth=None, max_features='sqrt',
        max_samples=None, class_weight='balanced_subsample'
    )

    clf = Pipeline(
        [
            ('preprocess', preprocessing),
            ('classifier', rf_clf)
        ]
    )

    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=0.1, stratify=targets)

    clf.fit(X_train, y_train)

    save_data = (preprocessing, rf_clf, X_train, X_test, y_train, y_test)

    destination = os.path.join(
        classifier_directory, file_name
    )
    with open(destination, 'wb') as file:
        pickle.dump(save_data, file)


def get_targets(data: pd.DataFrame):
    """
    Give it the raw uploaded pandas dataframe, and it will give you back the
    reason for discharge column, with 'Terminated by facility' set to 1 and
    all other values set to 0.
    :param data:
    :return:
    """
    targets = data['REASON']
    targets = (targets == 3).astype(int)
    return np.array(targets)


def get_2019_samsha_data(dir: str) -> pd.DataFrame:
    """
    This ended up so simple that it's almost superfluous, but I think it
    makes sense to keep it as a specific method in case we want to change our
    data access method in the future.
    :param dir:
    :return:
    """
    target_file = os.path.join(dir, 'tedsd_puf_2019.csv')
    return pd.read_csv(target_file)


if __name__ == '__main__':
    main()

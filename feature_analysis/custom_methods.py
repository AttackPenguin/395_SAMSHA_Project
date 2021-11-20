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
    RocCurveDisplay, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import parameters as p

data_directory = p.DATA_DIRECTORY
performance_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Permutation Importance')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'permutation_importance')


def main():

    start_time = pd.Timestamp.now()
    print(f"Started at {start_time}")

    # destination = os.path.join(
    #     classifier_directory, "Classifier 1, Full Data Set, Full Feature Set"
    # )
    # with open(destination, 'rb') as file:
    #     preprocessing, rf_clf, X_train, X_test, y_train, y_test \
    #         = pickle.load(file)
    #
    # results = permutation_feature_importance(
    #     preprocessing, rf_clf, roc_auc_score, X_test, y_test, 20,
    #     file_name='Permutation Feature Importance.pickle'
    # )

    destination = os.path.join(
        performance_data_directory, "Permutation Feature Importance.pickle"
    )
    with open(destination, 'rb') as file:
        results = pickle.load(file)

    fig_pf_importance(results,
                      'Fig XX: Permutation Feature Importance',
                      'Permutation Feature Importance')

    finish_time = pd.Timestamp.now()
    print(f"Run time = {finish_time}")


def fig_pf_importance(results: dict[str, list[float]],
                      title: str,
                      file_name: str):

    sorted_idx = np.array([np.mean(x) for x in results.values()]).argsort()
    importances = np.array([x for x in results.values()])

    fig: plt.Figure = plt.figure(figsize=[8.5, 11], dpi=400)
    ax: plt.Axes = fig.add_subplot()
    ax.boxplot(
        importances[sorted_idx].T, vert=False,
        labels=np.array(list(results.keys()))[sorted_idx]
    )
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    destination = os.path.join(
        performance_data_directory,
        file_name
    )
    fig.savefig(destination)


def permutation_feature_importance(preprocessor,
                                   classifier,
                                   scorer,
                                   features: pd.DataFrame,
                                   targets: pd.Series,
                                   iterations: int = 10,
                                   file_name: str | bool = False):

    results = {col: list() for col in features.columns}

    print(f"Generating initial baseline score...")
    t_features = preprocessor.transform(features)
    baseline_score = scorer(targets, classifier.predict_proba(t_features)[:, 1])
    print(f"Initial baseline score: {baseline_score}")

    for feature in results.keys():
        print(f"\nBeginning analysis of feature {feature}...")
        for i in range(1, iterations+1):
            print(f"\tIteration {i} of {iterations}...")
            test_features = features.copy(True)
            test_features[feature] = \
                np.random.permutation(test_features[feature])
            t_test_features = preprocessor.transform(test_features)
            score = scorer(targets,
                           classifier.predict_proba(t_test_features)[:, 1])
            results[feature].append(baseline_score-score)
        print(f"Analysis of feature {feature} complete.")
        print(f"Mean score delta = {np.mean(results[feature])}")

    if not file_name:
        destination = os.path.join(
            performance_data_directory, file_name + '.png'
        )
        with open(destination, 'wb') as file:
            pickle.dump(results, file)

    return results


if __name__ == '__main__':
    main()

import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

import seaborn as sns

# sns.set_theme(context='paper')

import parameters as p

data_directory = p.DATA_DIRECTORY
performance_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Final Code',
                 'Figures for Submission')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'final_code')


def main():
    gen_figures(
        'Permutation Feature Importance, Preprocessor 01.pickle',
        'rf_prep_01',
        'Correlation Heatmap and Dendrograms, Preprocessor 01.png'
    )
    gen_figures(
        'Permutation Feature Importance, Preprocessor 01 Optimized.pickle',
        'rf_prep_01_opt',
        'Correlation Heatmap and Dendrograms, Preprocessor 01 Optimized.png'
    )
    gen_figures(
        'Permutation Feature Importance, Preprocessor 02a.pickle',
        'rf_prep_02a',
        'Correlation Heatmap and Dendrograms, Preprocessor 02a.png'
    )
    gen_figures(
        'Permutation Feature Importance, Preprocessor 02b.pickle',
        'rf_prep_02b',
        'Correlation Heatmap and Dendrograms, Preprocessor 02b.png'
    )



def gen_figures(results_file: str,
                classifier_file: str,
                save_file: str):
    destination = os.path.join(
        performance_data_directory, results_file
    )
    with open(destination, 'rb') as file:
        results = pickle.load(file)

    importances = [row for row in results.values()]
    importances = np.array(importances)

    destination = os.path.join(
        classifier_directory, classifier_file
    )
    with open(destination, 'rb') as file:
        preprocessing, rf_clf, feature_labels, \
        X_train, X_test, y_train, y_test \
            = pickle.load(file)

    X = pd.concat((X_test, X_train), ignore_index=True)
    X = X[feature_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X.to_numpy()).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=feature_labels, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.savefig(os.path.join(performance_data_directory, save_file),
                dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
import pickle
import os

import numpy as np
import openpyxl as openpyxl
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

data_directory = p.DATA_DIRECTORY
performance_data_directory = \
    os.path.join(p.PERFORMANCE_DATA_DIRECTORY, 'Final Code', 'Preprocessor 01')
classifier_directory = \
    os.path.join(p.CLASSIFIER_DIRECTORY, 'final_code')
from parameters import FINAL_DATA


def main():
    start_time = pd.Timestamp.now()
    print(f"Started at {start_time}")

    target_file = os.path.join(data_directory, 'tedsd_puf_2019.csv')
    raw_data = pd.read_csv(target_file)

    _, features = preprocessors.create_preprocessor_01_opt()
    get_subsample(raw_data, features, 3)

    end_time = pd.Timestamp.now()
    print(f"Finished at {end_time}")
    print(f"Run time {(end_time - start_time).total_seconds() / 60:.2f} "
          f"minutes.")


def missing_feature_data_to_excel():
    """
    REPORT DATA - generates excel spreadsheet with missing data distributions.
    FINAL FORM - Output has been generated and this code does not need to be
    run again. Running it again will overwrite formatting to the spreadsheet
    performed post-generation.
    :return:
    """

    target_file = os.path.join(data_directory, 'tedsd_puf_2019.csv')
    raw_data = pd.read_csv(target_file)

    feature_sets = list()
    feature_sets.append(raw_data.columns)
    feature_sets.append(preprocessors.create_preprocessor_01()[1])
    feature_sets.append(preprocessors.create_preprocessor_01_opt()[1])
    feature_sets.append(preprocessors.create_preprocessor_02a()[1])
    feature_sets.append(preprocessors.create_preprocessor_02b()[1])

    set_names = ['raw data', '01', '01_opt', '02a', '02b']

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Summary Data'

    ws.cell(1, 1, 'Preprocessor')
    ws.cell(1, 2, 'Number of Features')
    ws.cell(2, 1, 'Raw Data')
    ws.cell(2, 2, len(feature_sets[0]))
    ws.cell(3, 1, '01')
    ws.cell(3, 2, len(feature_sets[1]))
    ws.cell(4, 1, '01_opt')
    ws.cell(4, 2, len(feature_sets[2]))
    ws.cell(5, 1, '02a')
    ws.cell(5, 2, len(feature_sets[3]))
    ws.cell(6, 1, '02b')
    ws.cell(6, 2, len(feature_sets[4]))

    for i, feature_set in enumerate(feature_sets):
        x_vals, y_vals = get_missing_data_dist(raw_data, feature_set)
        ws = wb.create_sheet(set_names[i])
        ws.cell(1, 1, 'Features With Data')
        ws.cell(1, 2, 'Samples')
        row = 2
        for x, y in list(zip(x_vals, y_vals)):
            ws.cell(row, 1, x)
            ws.cell(row, 2, y)
            row += 1

    destination = os.path.join(
        FINAL_DATA,
        'Distribution of Missing Data in Feature Sets.xlsx'
    )
    wb.save(destination)


def get_missing_data_dist(raw_data: pd.DataFrame,
                          features: list[str]):
    x_vals = list()
    y_vals = list()

    for i in range(len(features)):
        sub_sample = raw_data[features].copy()
        sub_sample['sum'] = (sub_sample == -9).sum(axis=1)
        sub_sample = sub_sample[sub_sample['sum'] == i][features]
        x_vals.append(len(features) - i)
        y_vals.append(sub_sample.shape[0])
        print(f"{len(features) - i}\t{sub_sample.shape[0]}")

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    ax.plot(x_vals, y_vals)
    ax.set_xlabel('Number of Data Points Known')
    ax.set_ylabel('Number of Samples')
    fig.show()

    return x_vals, y_vals


def get_subsample(raw_data: pd.DataFrame,
                  features: list[str],
                  max_missing: int):

    raw_data = raw_data
    targets = raw_data['REASON']
    targets = np.array((targets == 3).astype(int))
    terminated = (targets == 1).sum()
    fx_term_raw = terminated / len(targets)

    sub_sample = raw_data[features + ['REASON']]
    sub_sample['sum'] = (sub_sample == -9).sum(axis=1)
    sub_sample = \
        sub_sample[sub_sample['sum'] <= max_missing][features + ['REASON']]

    terminated = len(sub_sample[sub_sample['REASON'] == 3])
    fx_term_ss = terminated / len(sub_sample)

    # Scenario where fx_term_ss is less than fx_term_raw: need to trim
    # non-terminated patients.
    # if fx_term_ss < fx_term_raw:
    #     not_term = sub_sample[sub_sample['REASON'] != 3]
    #     term = sub_sample[sub_sample['REASON'] == 3]
    #     needed = int((len(term) / fx_term_raw) - len(term))
    #     not_term = not_term.sample(needed)
    #     sub_sample = pd.concat([term, not_term])

    # Scenario where fx_term_ss is greater than fx_term_raw: need to trim
    # terminated patients.
    # if fx_term_ss > fx_term_raw:
    #     not_term = sub_sample[sub_sample['REASON'] != 3]
    #     term = sub_sample[sub_sample['REASON'] == 3]
    #     needed = int((fx_term_raw * len(not_term)) / (1 - fx_term_raw))
    #     term = term.sample(needed)
    #     sub_sample = pd.concat([term, not_term])

    print(f"Returning subsample of length {len(sub_sample)}")
    return sub_sample


if __name__ == '__main__':
    main()

import copy
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
SVD linear embedding of the same feature inputs as fm_A_1, with -9 values 
converted to -1. Ratio values constructed for fm_A_1 also constructed here 
prior to linear embedding, as well as imputation for education.

Due to the enormous size of the required intermediate matrices, we are unable
to directly use SVD to convert the entire initial data set at once. 
Therefore, this model prepares the data set to be linearized, and provides a 
linearizing function that can be called as needed on subsets of the data, 
which must be processed as batches.

We are having trouble making this work. Results are far inferior to other 
built-in embedding techniques in sci-kit learn. Either the batch embedding 
causes issues or there's a fundamental problem with how we're implementing 
our SVD.

Only utilizes demographic/history data and facility data - Data Subset 1.

See parameters.py for grouping of data by type and summary descriptions of 
fields.
"""

data_directory = "/home/denis/Desktop/CSYS 395B - Machine Learning/Project/" \
                 "Data/2019"


def get_linearizable_data(data_directory: str = data_directory):
    # Import raw data
    raw_data = get_2019_samsha_data(data_directory)

    # Extract Targets
    targets = get_targets(raw_data)

    # Get working subset of raw data
    features = raw_data[[
        'GENDER', 'RACE', 'ETHNIC', 'MARSTAT',
        'EMPLOY', 'EMPLOY_D', 'DETNLF', 'DETNLF_D',
        'PREG', 'VET', 'LIVARAG', 'LIVARAG_D', 'PRIMINC',
        'ARRESTS', 'ARRESTS_D', 'DAYWAIT', 'PSOURCE', 'DETCRIM',
        'NOPRIOR', 'DSMCRIT', 'PSYPROB', 'HLTHINS', 'PRIMPAY',
        'SERVICES', 'METHUSE'
    ]]

    # Make specific column modifications
    features['AGE'] = get_age_encoding(raw_data)
    features['FREQ_ATND_SELF_HELP'] = \
        get_self_help_att_pre_tx_encoding(raw_data)
    features['FREQ_ATND_SELF_HELP_D'] = \
        get_self_help_att_pre_dc_encoding(raw_data)
    features['EDUC'] = \
        get_education_encoding(raw_data)

    # Convert -9 values to -1s
    features = features.replace([-9], -1)

    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=20_000)

    return X_train, X_test, y_train, y_test


def linearize_data(X_train: np.ndarray):

    # Perform singular value decomposition on fragment features
    X_in = X_train.T
    U, S, Vt = np.linalg.svd(X_in)
    lin_X_train = copy.deepcopy(Vt[:, :X_train.shape[1]])
    return lin_X_train


def get_self_help_att_pre_dc_encoding(data: pd.DataFrame):
    """
    I'm experimenting with turning this into continuous data based on the
    groupings. I'm lumping unknowns in with 'no attendance', by far the most
    frequent category. I'm putting 'some attendance, frequency unknown',
    into the median attendance group.
    :param data:
    :return:
    """

    def transform_age(val: int) -> float:
        if val == 1:
            return_val = 0.0
        elif val == 2:
            return_val = 2.0
        elif val == 3:
            return_val = 5.5
        elif val == 4:
            return_val = 19.0
        elif val == 5:
            return_val = 5.5
        elif val == -9:
            return_val = 0.0
        else:
            raise ValueError('val must be an integer in the range [1,5] or -9')
        return return_val

    self_help_pre_dc = data['FREQ_ATND_SELF_HELP_D']
    self_help_pre_dc = self_help_pre_dc.transform(transform_age)
    return np.array(self_help_pre_dc).reshape((-1, 1))


def get_self_help_att_pre_tx_encoding(data: pd.DataFrame):
    """
    I'm experimenting with turning this into continuous data based on the
    groupings. I'm lumping unknowns in with 'no attendance', by far the most
    frequent category. I'm putting 'some attendance, frequency unknown',
    into the median attendance group.
    :param data:
    :return:
    """

    def transform_age(val: int) -> float:
        if val == 1:
            return_val = 0.0
        elif val == 2:
            return_val = 2.0
        elif val == 3:
            return_val = 5.5
        elif val == 4:
            return_val = 19.0
        elif val == 5:
            return_val = 5.5
        elif val == -9:
            return_val = 0.0
        else:
            raise ValueError('val must be an integer in the range [1,5] or -9')
        return return_val

    self_help_pre_tx = data['FREQ_ATND_SELF_HELP']
    self_help_pre_tx = self_help_pre_tx.transform(transform_age)
    return np.array(self_help_pre_tx).reshape((-1, 1))


def get_education_encoding(data: pd.DataFrame):
    """
    I'm going to opt to treat this as ordinal for now, and set the value of
    -9 for missing data to 3, for 'grade 12 (or GED)', because that is the
    most frequent value otherwise.
    :param data:
    :return:
    """

    def transform_education(val: int):
        if val == -9:
            return_val = 3
        else:
            return_val = val
        return return_val

    education = data['EDUC']
    education = np.array(education.transform(transform_education))
    return education.reshape((-1, 1))


def get_age_encoding(data: pd.DataFrame):
    """
    Converts age data to emulate continuous data, by setting each ordinal
    value to the midpoint of the associated age range. We might want to do a
    little research to support my choice of 70 for the '65 and over bracket'.
    We could actually get distribution data on actual participation and pick
    weighted instead of central midpoints.
    :param data:
    :return:
    """

    def transform_age(val: int) -> float:
        if val == 1:
            return_val = 13.0
        elif val == 2:
            return_val = 16.0
        elif val == 3:
            return_val = 19.0
        elif val == 4:
            return_val = 22.5
        elif val == 5:
            return_val = 27.0
        elif val == 6:
            return_val = 32.0
        elif val == 7:
            return_val = 37.0
        elif val == 8:
            return_val = 42.0
        elif val == 9:
            return_val = 47.0
        elif val == 10:
            return_val = 52.0
        elif val == 11:
            return_val = 59.5
        elif val == 12:
            return_val = 70.0
        else:
            raise ValueError('val must be an integer in the range [1,12]')
        return return_val

    age = data['AGE']
    age = age.transform(transform_age)
    return np.array(age).reshape((-1, 1))


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

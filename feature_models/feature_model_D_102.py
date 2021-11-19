import copy
import os

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

"""
See feature_model_D_1.py

This variation removes the ratio encoding of AGE and the ordinal encoding. 
All non-binary data is one-hot encoded.

This variation also removes 

"""

pd.set_option('mode.chained_assignment', None)

data_directory = "/home/denis/Desktop/CSYS 395B - Machine Learning/Project/" \
                 "Data/2019"


def main():
    data = get_data()


def get_data(data_directory: str = data_directory):
    # Import raw data
    raw_data = get_2019_samsha_data(data_directory)

    # Extract Targets
    targets = get_targets(raw_data)

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
    # Missing data points represented by -9. Replace with -1
    incomplete_category_data.replace(-9, -1, inplace=True)

    # Get ordinal data that can be approximated with ratio data.
    convert_to_ratio_cols = [
        'AGE'
    ]
    convert_to_ratio_data = raw_data[convert_to_ratio_cols]

    # Get ordinal data with missing values to be imputed
    incomplete_ordinal_cols = [
        'DAYWAIT', 'FREQ_ATND_SELF_HELP', 'FREQ_ATND_SELF_HELP_D'
    ]
    incomplete_ordinal_data = raw_data[incomplete_ordinal_cols]

    features = binary_category_data.join([
        complete_category_data,
        incomplete_category_data,
        convert_to_ratio_data,
        incomplete_ordinal_data
    ])

    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=0.2)
    X_train = pd.DataFrame(
        X_train, columns=(
            binary_category_cols + complete_category_cols +
            incomplete_category_cols + convert_to_ratio_cols +
            incomplete_ordinal_cols
        )
    )
    X_test = pd.DataFrame(
        X_test, columns=(
            binary_category_cols + complete_category_cols +
            incomplete_category_cols + convert_to_ratio_cols +
            incomplete_ordinal_cols
        )
    )

    encoder = ce.OneHotEncoder(
        verbose=0, return_df=True, use_cat_names=True,
        cols=(
            complete_category_cols +
            incomplete_category_cols +
            convert_to_ratio_cols +
            incomplete_ordinal_cols
        )
    )
    encoder.fit(X_train)
    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)

    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='-1')))]
    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='-1')))]

    return X_train, X_test, y_train, y_test


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
            return_val = 13.0 / 70.0
        elif val == 2:
            return_val = 16.0 / 70.0
        elif val == 3:
            return_val = 19.0 / 70.0
        elif val == 4:
            return_val = 22.5 / 70.0
        elif val == 5:
            return_val = 27.0 / 70.0
        elif val == 6:
            return_val = 32.0 / 70.0
        elif val == 7:
            return_val = 37.0 / 70.0
        elif val == 8:
            return_val = 42.0 / 70.0
        elif val == 9:
            return_val = 47.0 / 70.0
        elif val == 10:
            return_val = 52.0 / 70.0
        elif val == 11:
            return_val = 59.5 / 70.0
        elif val == 12:
            return_val = 70.0 / 70.0
        else:
            raise ValueError('val must be an integer in the range [1,12]')
        return return_val

    # age = data['AGE']
    # age = age.transform(transform_age)
    # return np.array(age).reshape((-1, 1))
    return data['AGE'].transform(transform_age)


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
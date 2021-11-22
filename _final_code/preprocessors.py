import category_encoders as ce
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def main():
    preprocessor = create_preprocessor_01()


def create_preprocessor_01():
    """
    Initial preprocessor, without pruning of low-value features.
    :return:
    """

    # Get categorical data that is binary
    binary_category_cols = [
        'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG',
        'METHFLG', 'OPSYNFLG', 'MTHAMFLG', 'BENZFLG'
    ]

    # Get categorical data with no missing datapoints:
    complete_category_cols = [
        'SERVICES', 'ALCDRUG'
    ]

    # Get categorical data with missing datapoints
    incomplete_category_cols = [
        'GENDER', 'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'EMPLOY_D',
        'DETNLF', 'DETNLF_D', 'PREG', 'VET', 'LIVARAG', 'LIVARAG_D', 'PRIMINC',
        'ARRESTS', 'ARRESTS_D', 'PSOURCE', 'DETCRIM', 'NOPRIOR', 'DSMCRIT',
        'PSYPROB', 'HLTHINS', 'PRIMPAY', 'METHUSE', 'IDU'
    ]

    incomplete_ordinal_cols = [
        'DAYWAIT', 'FREQ_ATND_SELF_HELP', 'FREQ_ATND_SELF_HELP_D'
    ]

    ratio_columns = [
        'AGE'
    ]

    inc_ordinal_pipe = Pipeline([
        ('imputer',
         SimpleImputer(missing_values=-9, strategy='most_frequent')),
        ('inc_ord', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        [
            ('binary',
             OneHotEncoder(drop='if_binary', sparse=False),
             binary_category_cols),
            ('helmert',
             ce.HelmertEncoder(return_df=True),
             complete_category_cols + incomplete_category_cols),
            ('inc_ord',
             inc_ordinal_pipe,
             incomplete_ordinal_cols),
            ('ratio',
             AgeRatioTransformer(),
             ratio_columns)
        ], n_jobs=10
    )

    features = (binary_category_cols +
                complete_category_cols +
                incomplete_category_cols +
                incomplete_ordinal_cols +
                ratio_columns)

    return preprocessor, features


def create_preprocessor_01_opt():
    """
    Initial preprocessor, with pruning of features with negative or zero mean
    permutation feature importance for area under ROC.
    In theory, this should be our top performing classifier.
    :return:
    """

    # Get categorical data that is binary
    binary_category_cols = [
        'COKEFLG', 'MARFLG', 'HERFLG',
        'OPSYNFLG', 'MTHAMFLG', 'BENZFLG'
    ]

    # Get categorical data with no missing datapoints:
    complete_category_cols = [
        'SERVICES', 'ALCDRUG'
    ]

    # Get categorical data with missing datapoints
    incomplete_category_cols = [
        'GENDER', 'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'EMPLOY_D',
        'DETNLF', 'DETNLF_D', 'PREG', 'VET', 'LIVARAG', 'LIVARAG_D', 'PRIMINC',
        'ARRESTS', 'ARRESTS_D', 'PSOURCE', 'DETCRIM', 'NOPRIOR', 'DSMCRIT',
        'PSYPROB', 'HLTHINS', 'PRIMPAY', 'METHUSE', 'IDU'
    ]

    incomplete_ordinal_cols = [
        'DAYWAIT', 'FREQ_ATND_SELF_HELP', 'FREQ_ATND_SELF_HELP_D'
    ]

    ratio_columns = [
        'AGE'
    ]

    inc_ordinal_pipe = Pipeline([
        ('imputer',
         SimpleImputer(missing_values=-9, strategy='most_frequent')),
        ('inc_ord', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        [
            ('binary',
             OneHotEncoder(drop='if_binary', sparse=False),
             binary_category_cols),
            ('helmert',
             ce.HelmertEncoder(return_df=True),
             complete_category_cols + incomplete_category_cols),
            ('inc_ord',
             inc_ordinal_pipe,
             incomplete_ordinal_cols),
            ('ratio',
             AgeRatioTransformer(),
             ratio_columns)
        ], n_jobs=10
    )

    features = (binary_category_cols +
                complete_category_cols +
                incomplete_category_cols +
                incomplete_ordinal_cols +
                ratio_columns)

    return preprocessor, features


def create_preprocessor_02a():
    """
    Second stage. Features with an ROC permutation importance of under 0.0025
    pruned.
    :return:
    """

    # Get categorical data that is binary
    binary_category_cols = [
        'MARFLG'
    ]

    # Get categorical data with no missing datapoints:
    complete_category_cols = [
        'SERVICES', 'ALCDRUG'
    ]

    # Get categorical data with missing datapoints
    incomplete_category_cols = [
        'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'EMPLOY_D',
        'DETNLF', 'DETNLF_D', 'VET', 'LIVARAG', 'LIVARAG_D', 'PRIMINC',
        'ARRESTS', 'ARRESTS_D', 'PSOURCE', 'DETCRIM', 'NOPRIOR', 'DSMCRIT',
        'PSYPROB', 'HLTHINS', 'PRIMPAY', 'METHUSE', 'IDU'
    ]

    incomplete_ordinal_cols = [
        'DAYWAIT', 'FREQ_ATND_SELF_HELP', 'FREQ_ATND_SELF_HELP_D'
    ]

    ratio_columns = [
        'AGE'
    ]

    inc_ordinal_pipe = Pipeline([
        ('imputer',
         SimpleImputer(missing_values=-9, strategy='most_frequent')),
        ('inc_ord', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        [
            ('binary',
             OneHotEncoder(drop='if_binary', sparse=False),
             binary_category_cols),
            ('helmert',
             ce.HelmertEncoder(return_df=True),
             complete_category_cols + incomplete_category_cols),
            ('inc_ord',
             inc_ordinal_pipe,
             incomplete_ordinal_cols),
            ('ratio',
             AgeRatioTransformer(),
             ratio_columns)
        ], n_jobs=10
    )

    features = (binary_category_cols +
                complete_category_cols +
                incomplete_category_cols +
                incomplete_ordinal_cols +
                ratio_columns)

    return preprocessor, features


def create_preprocessor_02b():
    """
    Second stage. Features with an ROC permutation importance of under 0.005
    pruned.
    :return:
    """

    # Get categorical data that is binary
    binary_category_cols = [

    ]

    # Get categorical data with no missing datapoints:
    complete_category_cols = [
        'SERVICES'
    ]

    # Get categorical data with missing datapoints
    incomplete_category_cols = [
        'RACE', 'MARSTAT', 'EDUC', 'EMPLOY_D',
        'DETNLF', 'DETNLF_D', 'VET', 'LIVARAG', 'LIVARAG_D', 'PRIMINC',
        'ARRESTS_D', 'PSOURCE', 'DETCRIM', 'NOPRIOR', 'DSMCRIT',
        'PSYPROB', 'HLTHINS', 'PRIMPAY', 'METHUSE'
    ]

    incomplete_ordinal_cols = [
        'DAYWAIT', 'FREQ_ATND_SELF_HELP_D'
    ]

    ratio_columns = [
        'AGE'
    ]

    inc_ordinal_pipe = Pipeline([
        ('imputer',
         SimpleImputer(missing_values=-9, strategy='most_frequent')),
        ('inc_ord', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        [
            ('binary',
             OneHotEncoder(drop='if_binary', sparse=False),
             binary_category_cols),
            ('helmert',
             ce.HelmertEncoder(return_df=True),
             complete_category_cols + incomplete_category_cols),
            ('inc_ord',
             inc_ordinal_pipe,
             incomplete_ordinal_cols),
            ('ratio',
             AgeRatioTransformer(),
             ratio_columns)
        ], n_jobs=10
    )

    features = (binary_category_cols +
                complete_category_cols +
                incomplete_category_cols +
                incomplete_ordinal_cols +
                ratio_columns)

    return preprocessor, features


class AgeRatioTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform_age(self, val: int) -> float:
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

    def transform(self, X):
        X = pd.DataFrame(X).copy(deep=True)
        for i in range(X.shape[0]):
            x = self.transform_age(X.iloc[i, 0])
            X.iloc[i, 0] = x
        return X


if __name__ == '__main__':
    main()

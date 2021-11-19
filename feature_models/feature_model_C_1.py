import copy
import os

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

"""
Performs linear embedding using SVD in the same manner as model B_1, 
but using the one-hot preprocessing model for categorical data used in model 
A_1.

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
fields."""

data_directory = "/home/denis/Desktop/CSYS 395B - Machine Learning/Project/" \
                 "Data/2019"


def get_linearizable_data(data_directory: str = data_directory):
    # Import raw data
    raw_data = get_2019_samsha_data(data_directory)

    # Extract Targets
    targets = get_targets(raw_data)

    # Preprocesss Demographic and History Data
    age = get_age_encoding(raw_data)
    race = get_race_encoding(raw_data)
    gender = get_gender_encoding(raw_data)
    ethnicity = get_ethnicity_encoding(raw_data)
    marital_status = get_marital_status_encoding(raw_data)
    education = get_education_encoding(raw_data)
    employment_pre_tx = get_employment_pre_tx_encoding(raw_data)
    employment_post_tx = get_employment_post_tx_encoding(raw_data)
    pre_tx_not_in_labor_force = get_pre_tx_not_in_labor_force_encoding(raw_data)
    post_tx_not_in_labor_force = \
        get_post_tx_not_in_labor_force_encoding(raw_data)
    pregnant = get_pregnant_encoding(raw_data)
    veteran = get_veteran_encoding(raw_data)
    living_arr_pre_tx = get_living_arr_pre_tx_encoding(raw_data)
    living_arr_post_tx = get_living_arr_post_tx_encoding(raw_data)
    primary_income = get_primary_income_encoding(raw_data)
    arrests_pre_tx = get_arrests_pre_tx_encoding(raw_data)
    arrests_pre_dc = get_arrests_pre_dc_encoding(raw_data)
    tx_wait_time = get_tx_wait_time_encoding(raw_data)
    referral_source = get_referral_source_encoding(raw_data)
    det_crim_just_referral = get_det_crim_just_referral_encoding(raw_data)
    no_prior_tx = get_no_prior_tx_encoding(raw_data)
    dsm_diagnosis = get_dsm_diagnosis_encoding(raw_data)
    dual_diagnosis = get_dual_diagnosis_encoding(raw_data)
    self_help_att_pre_tx = get_self_help_att_pre_tx_encoding(raw_data)
    self_help_att_pre_dc = get_self_help_att_pre_dc_encoding(raw_data)
    health_ins = get_health_ins_encoding(raw_data)
    primary_payment_src = get_primary_payment_src_encoding(raw_data)

    # Facility Data
    tx_setting = get_tx_setting_encoding(raw_data)
    med_assisted_opiod_tx = get_med_assisted_opioid_tx_encoding(raw_data)

    # Assemble data into single ndarray of features
    features = np.concatenate((
        age, race, gender, ethnicity, marital_status, education,
        employment_pre_tx, employment_post_tx,
        pre_tx_not_in_labor_force, post_tx_not_in_labor_force,
        pregnant, veteran, living_arr_pre_tx, living_arr_post_tx,
        primary_income, arrests_pre_tx, arrests_pre_dc,
        tx_wait_time, referral_source, det_crim_just_referral,
        no_prior_tx, dsm_diagnosis, dual_diagnosis,
        self_help_att_pre_tx, self_help_att_pre_dc,
        health_ins, primary_payment_src,
        tx_setting, med_assisted_opiod_tx
    ), axis=1)

    X_train, X_test, y_train, y_test = \
        train_test_split(features, targets, test_size=20_000)

    return X_train, X_test, y_train, y_test


def linearize_data(X_train: np.ndarray):

    # Perform singular value decomposition on fragment features
    X_in = X_train.T
    U, S, Vt = np.linalg.svd(X_in)
    lin_X_train = copy.deepcopy(Vt[:, :X_train.shape[1]])
    return lin_X_train


def get_med_assisted_opioid_tx_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    med_assisted_opiod_tx = np.array(data['SERVICES_D'])
    med_assisted_opiod_tx = med_assisted_opiod_tx.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(med_assisted_opiod_tx)
    med_assisted_opiod_tx = enc.transform(med_assisted_opiod_tx).toarray()
    return med_assisted_opiod_tx


def get_tx_setting_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    tx_setting = np.array(data['SERVICES'])
    tx_setting = tx_setting.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(tx_setting)
    tx_setting = enc.transform(tx_setting).toarray()
    return tx_setting


def get_primary_payment_src_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    primary_patment_src = np.array(data['PRIMPAY'])
    primary_patment_src = primary_patment_src.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(primary_patment_src)
    primary_patment_src = enc.transform(primary_patment_src).toarray()
    return primary_patment_src


def get_health_ins_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    health_ins = np.array(data['HLTHINS'])
    health_ins = health_ins.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(health_ins)
    health_ins = enc.transform(health_ins).toarray()
    return health_ins


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


def get_dual_diagnosis_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    dual_diagnosis = np.array(data['PSYPROB'])
    dual_diagnosis = dual_diagnosis.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(dual_diagnosis)
    dual_diagnosis = enc.transform(dual_diagnosis).toarray()
    return dual_diagnosis


def get_dsm_diagnosis_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    dsm_diagnosis = np.array(data['DSMCRIT'])
    dsm_diagnosis = dsm_diagnosis.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(dsm_diagnosis)
    dsm_diagnosis = enc.transform(dsm_diagnosis).toarray()
    return dsm_diagnosis


def get_no_prior_tx_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    no_prior_tx = np.array(data['NOPRIOR'])
    no_prior_tx = no_prior_tx.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(no_prior_tx)
    no_prior_tx = enc.transform(no_prior_tx).toarray()
    return no_prior_tx


def get_det_crim_just_referral_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    det_crim_just_referral = np.array(data['DETCRIM'])
    det_crim_just_referral = det_crim_just_referral.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(det_crim_just_referral)
    det_crim_just_referral = enc.transform(det_crim_just_referral).toarray()
    return det_crim_just_referral


def get_referral_source_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    referral_source = np.array(data['PSOURCE'])
    referral_source = referral_source.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(referral_source)
    referral_source = enc.transform(referral_source).toarray()
    return referral_source


def get_tx_wait_time_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    This could potentially be ordinal, but more than 50% of values are missing.
    :param data:
    :return:
    """
    tx_wait_time = np.array(data['DAYWAIT'])
    tx_wait_time = tx_wait_time.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(tx_wait_time)
    tx_wait_time = enc.transform(tx_wait_time).toarray()
    return tx_wait_time


def get_arrests_pre_dc_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    arrests_pre_dc = np.array(data['ARRESTS_D'])
    arrests_pre_dc = arrests_pre_dc.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(arrests_pre_dc)
    arrests_pre_dc = enc.transform(arrests_pre_dc).toarray()
    return arrests_pre_dc


def get_arrests_pre_tx_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    arrests_pre_tx = np.array(data['ARRESTS'])
    arrests_pre_tx = arrests_pre_tx.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(arrests_pre_tx)
    arrests_pre_tx = enc.transform(arrests_pre_tx).toarray()
    return arrests_pre_tx


def get_primary_income_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    primary_income = np.array(data['PRIMINC'])
    primary_income = primary_income.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(primary_income)
    primary_income = enc.transform(primary_income).toarray()
    return primary_income


def get_living_arr_post_tx_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    living_arr_post_tx = np.array(data['LIVARAG_D'])
    living_arr_post_tx = living_arr_post_tx.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(living_arr_post_tx)
    living_arr_post_tx = enc.transform(living_arr_post_tx).toarray()
    return living_arr_post_tx


def get_living_arr_pre_tx_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    living_arr_pre_tx = np.array(data['LIVARAG'])
    living_arr_pre_tx = living_arr_pre_tx.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(living_arr_pre_tx)
    living_arr_pre_tx = enc.transform(living_arr_pre_tx).toarray()
    return living_arr_pre_tx


def get_veteran_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    veteran = np.array(data['VET'])
    veteran = veteran.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(veteran)
    veteran = enc.transform(veteran).toarray()
    return veteran


def get_pregnant_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    pregnant = np.array(data['PREG'])
    pregnant = pregnant.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(pregnant)
    pregnant = enc.transform(pregnant).toarray()
    return pregnant


def get_post_tx_not_in_labor_force_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    post_tx_not_in_labor_force = np.array(data['DETNLF_D'])
    post_tx_not_in_labor_force = post_tx_not_in_labor_force.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(post_tx_not_in_labor_force)
    post_tx_not_in_labor_force = \
        enc.transform(post_tx_not_in_labor_force).toarray()
    return post_tx_not_in_labor_force


def get_pre_tx_not_in_labor_force_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    pre_tx_not_in_labor_force = np.array(data['DETNLF'])
    pre_tx_not_in_labor_force = pre_tx_not_in_labor_force.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(pre_tx_not_in_labor_force)
    pre_tx_not_in_labor_force = \
        enc.transform(pre_tx_not_in_labor_force).toarray()
    return pre_tx_not_in_labor_force


def get_employment_post_tx_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    employment_post_tx = np.array(data['EMPLOY_D'])
    employment_post_tx = employment_post_tx.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(employment_post_tx)
    employment_post_tx = enc.transform(employment_post_tx).toarray()
    return employment_post_tx


def get_employment_pre_tx_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    employment_pre_tx = np.array(data['EMPLOY'])
    employment_pre_tx = employment_pre_tx.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(employment_pre_tx)
    employment_pre_tx = enc.transform(employment_pre_tx).toarray()
    return employment_pre_tx


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


def get_marital_status_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    marital_status = np.array(data['MARSTAT'])
    marital_status = marital_status.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(marital_status)
    marital_status = enc.transform(marital_status).toarray()
    return marital_status


def get_ethnicity_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    ethnicity = np.array(data['ETHNIC'])
    ethnicity = ethnicity.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(ethnicity)
    ethnicity = enc.transform(ethnicity).toarray()
    return ethnicity


def get_gender_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    gender = np.array(data['GENDER'])
    gender = gender.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(gender)
    gender = enc.transform(gender).toarray()
    return gender


def get_race_encoding(data: pd.DataFrame):
    """
    Currently simply performs one-hot encoding with scikitlearn's preprocessing.
    :param data:
    :return:
    """
    race = np.array(data['RACE'])
    race = race.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(race)
    race = enc.transform(race).toarray()
    return race


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


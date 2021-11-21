import os

import numpy as np
import pandas as pd

"""

"""

data_directory = "/home/denis/Desktop/CSYS 395B - Machine Learning/Project/" \
                 "Data/2019"


def main():
    data = get_2019_samsha_data(data_directory)
    print_feature_value_data(data, 'SERVICES', 1, 'REASON')
    print_feature_value_data(data, 'SERVICES', 2, 'REASON')
    print_feature_value_data(data, 'SERVICES', 3, 'REASON')
    print_feature_value_data(data, 'SERVICES', 4, 'REASON')
    print_feature_value_data(data, 'SERVICES', 5, 'REASON')
    print_feature_value_data(data, 'SERVICES', 6, 'REASON')
    print_feature_value_data(data, 'SERVICES', 7, 'REASON')
    print_feature_value_data(data, 'SERVICES', 8, 'REASON')


def get_feature_value_data(data: pd.DataFrame,
                           f_in: str,
                           f_in_value: int,
                           f_out: str):
    all_results = data[f_out].value_counts()
    all_results = dict(all_results)
    all_results = list(all_results.items())
    all_results.sort(key=lambda x: x[0])
    all_results = {x[0]: x[1] for x in all_results}
    total = sum(all_results.values())
    for key, value in all_results.items():
        all_results[key] = value/total

    matches = data.loc[data[f_in] == f_in_value]
    value_results = matches[f_out].value_counts()
    value_results = dict(value_results)
    value_results = list(value_results.items())
    value_results.sort(key=lambda x: x[0])
    value_results = {x[0]: x[1] for x in value_results}
    total = sum(value_results.values())
    for key in all_results:
        if key in value_results:
            value_results[key] = value_results[key]/total
        else:
            value_results[key] = 0.0

    return value_results, all_results


def print_feature_value_data(data: pd.DataFrame,
                           f_in: str,
                           f_in_value: int,
                           f_out: str):
    value_results, all_results = \
        get_feature_value_data(data, f_in, f_in_value, f_out)

    print(f"\nFor {f_in} equal to {f_in_value}, the breakdown of {f_out} is:")
    print(f"\n\t{f_in}={f_in_value}\tAll Pts")
    for key in value_results:
        print(f"{key}\t"
              f"{value_results[key]*100:.2f}%\t"
              f"{all_results[key]*100:.2f}%")


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
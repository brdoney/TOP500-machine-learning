from typing import Dict, List, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split

from data_cleaning import Transformer, prep_dataframe, select_past
from read_data import read_datasets


def train_test_random(
    dataframe: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Splitting also scrambles the values by default
    train_test_list = train_test_split(dataframe, test_size=test_size, random_state=10)
    train_test_list = cast(List[pd.DataFrame], train_test_list)

    # There should only be two elements, but just in case
    train, test = train_test_list
    return train, test


def train_test_year(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    max_date = dataframe["Date"].max()

    # Use the current year as the testing data, everything else is training
    is_most_recent: pd.Series[bool] = dataframe["Date"] == max_date  # type: ignore
    not_is_most_recent: pd.Series[bool] = ~is_most_recent

    # Also remove the Date column, since it won't be used in training
    test = dataframe[is_most_recent].drop(columns="Date")
    train = dataframe[not_is_most_recent].drop(columns="Date")

    return train, test


def split_x_y(
    dataframes: Union[List[pd.DataFrame], pd.DataFrame], dependent_var: str
) -> List[Tuple[pd.DataFrame, pd.Series]]:
    splits = []

    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    for df in dataframes:
        y = df[dependent_var]
        X = df.drop(columns=dependent_var)
        splits.append((X, y))
    return splits


def _toa_all_data(dep_var: str, preprocessor: Transformer) -> pd.DataFrame:
    # Get the processed data
    all_data = read_datasets()
    data = prep_dataframe(all_data, dep_var)
    data = preprocessor.fit_transform(data)

    # Remove the Date column, since it won't be used in training
    data = data.drop(columns="Date", errors="ignore")

    return data


def toa_data(dep_var: str, preprocessor: Transformer) -> List[Tuple[pd.DataFrame, pd.Series]]:
    data = _toa_all_data(dep_var, preprocessor)
    non_holdout, holdout = train_test_random(data, 0.1)
    train, test = train_test_random(non_holdout, 0.1)

    # Do splits for all data
    return split_x_y([train, test, holdout], dep_var)


def toa_data_nohold(dep_var: str, preprocessor: Transformer) -> List[Tuple[pd.DataFrame, pd.Series]]:
    data = _toa_all_data(dep_var, preprocessor)
    train, test = train_test_random(data, 0.1)

    # Do splits for all data
    return split_x_y([train, test], dep_var)


def top_data(dep_var: str, preprocessor: Transformer) -> List[Tuple[pd.DataFrame, pd.Series]]:
    all_data = read_datasets()

    data = prep_dataframe(all_data, dep_var)
    data = select_past(data, 3)
    data = preprocessor.fit_transform(data)
    train, test = train_test_year(data)

    # Do splits for all data
    return split_x_y([train, test], dep_var)


def calc_stats(
    exp_y: Union[pd.Series, np.ndarray],
    pred_y: Union[pd.Series, np.ndarray],
    print_res: bool = True,
    prefix: str = ""
) -> Dict[str, float]:
    results = {
        "r2": r2_score(exp_y, pred_y),
        "mae": mean_absolute_error(exp_y, pred_y),
        "mape": mean_absolute_percentage_error(exp_y, pred_y),
        "mse": mean_squared_error(exp_y, pred_y)
    }

    if print_res:
        for name, result in results.items():
            print(f"{prefix} {name}: {result}")

    return results

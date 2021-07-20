from typing import List, Tuple, Union, cast
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from data_cleaning import prep_dataframe, preprocess_data, Transformer, select_past
from read_data import read_datasets


def train_test_random(
    dataframe: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Splitting also scrambles the values by default
    train_test_list = train_test_split(dataframe, test_size=test_size)
    train_test_list = cast(List[pd.DataFrame], train_test_list)

    # Remove the Date column if it still exists, since it won't be used in training
    train_test_list = [df.drop(columns="Date", errors="ignore") for df in train_test_list]

    # There should only be two elements, but just in case
    train, test = train_test_list
    return train, test


def train_test_year(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    max_date = dataframe["Date"].max()

    # Use the current year as the testing data, everything else is training
    is_most_recent = cast(pd.Series[bool], dataframe["Date"] == max_date)
    not_is_most_recent: pd.Series[bool] = ~is_most_recent

    # Also remove the Date column, since it won't be used in training
    test = dataframe[is_most_recent].drop(columns="Date")
    train = dataframe[not_is_most_recent].drop(columns="Date")

    return train, test


def split_x_y(
    dataframes: List[pd.DataFrame], dependent_var: str
) -> List[Tuple[pd.DataFrame, pd.Series]]:
    splits = []
    for df in dataframes:
        y = df[dependent_var]
        X = df.drop(columns=dependent_var)
        splits.append((X, y))
    return splits


def toa_data(dep_var: str, scaler: Transformer) -> List[Tuple[pd.DataFrame, pd.Series]]:
    all_data = read_datasets()

    data = prep_dataframe(all_data, dep_var)
    data, _ = preprocess_data(data, dep_var, scaler, True)
    non_holdout, holdout = train_test_random(data, 0.1)
    train, test = train_test_random(non_holdout, 0.1)

    # Do splits for all data
    return split_x_y([train, test, holdout], dep_var)


def top_data(dep_var: str, scaler: Transformer) -> List[Tuple[pd.DataFrame, pd.Series]]:
    all_data = read_datasets()

    data = prep_dataframe(all_data, dep_var)
    data = select_past(data, 3)
    data, _ = preprocess_data(data, dep_var, scaler, True)
    train, test = train_test_year(data)

    # Do splits for all data
    return split_x_y([train, test], dep_var)


def print_stats(
    exp_y: Union[pd.Series, np.ndarray], pred_y: Union[pd.Series, np.ndarray], prefix: str
):
    r2 = r2_score(exp_y, pred_y)
    mae = mean_absolute_error(exp_y, pred_y)
    mape = mean_absolute_percentage_error(exp_y, pred_y)
    mse = mean_squared_error(exp_y, pred_y)
    print(f"{prefix} R^2: {r2}")
    print(f"{prefix} MAE: {mae}")
    print(f"{prefix} MAPE: {mape}")
    print(f"{prefix} MSE: {mse}")

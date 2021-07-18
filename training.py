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
    # Also remove the Date column, since it won't be used in training
    is_most_recent = dataframe["Date"] == max_date
    test = dataframe[is_most_recent].drop(columns="Date")
    train = dataframe[~is_most_recent].drop(columns="Date")

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

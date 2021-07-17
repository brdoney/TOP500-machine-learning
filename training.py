from typing import List, Literal, Tuple, cast

import pandas as pd
from sklearn.model_selection import train_test_split


def train_test(dataframe: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Splitting also scrambles the values by default
    train, test = train_test_split(dataframe, test_size=test_size)
    train = cast(pd.DataFrame, train)
    test = cast(pd.DataFrame, test)
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


def select_past(
    dataframe: pd.DataFrame,
    current_year: int,
    current_month: Literal[6, 11],
    select_num_datasets: int,
) -> pd.DataFrame:
    valid_months = ["06", "11"]

    selected_dates: List[str] = []

    month_index = valid_months.index(f"{current_month:02}")
    year = current_year
    for _ in range(select_num_datasets):
        if month_index == 0:
            year -= 1
        month_index = (month_index - 1) % len(valid_months)

        date_str = f"{year}{valid_months[month_index]}"
        selected_dates.append(date_str)

    selected = dataframe[dataframe["Date"].isin(selected_dates)].copy()

    # Sorts in ascending order (earlier date first), dropping date since it won't be used
    # in training
    sorted_df = selected.sort_values(by="Date")
    sorted_df = sorted_df.drop(columns="Date")

    return sorted_df

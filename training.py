from typing import List, Tuple, cast

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

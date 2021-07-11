from typing import List, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler

from data_cleaning import preprocess_data
from read_data import read_datasets


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


if __name__ == "__main__":
    dep_var = "Log(Rmax)"
    all_data = read_datasets()
    data, _ = preprocess_data(all_data, dep_var, RobustScaler(), True)
    # dep_var = "Log(Efficiency)"
    # data = get_data(dep_var, StandardScaler())

    non_holdout, holdout = train_test(data, 0.1)
    train, test = train_test(non_holdout, 0.1)

    (train_X, train_y), (test_X, test_y) = split_x_y([train, test], dep_var)
    print("Done prepping")

    model = RandomForestRegressor(n_estimators=1000)
    model.fit(train_X, train_y)
    print("Done training")

    # Testing score
    pred_y = model.predict(test_X)
    r2 = r2_score(test_y, pred_y)
    print(f"Testing R^2: {r2}")

    # Holdout score
    [(hold_X, hold_y)] = split_x_y([holdout], dep_var)
    pred_y = model.predict(hold_X)
    r2 = r2_score(hold_y, pred_y)
    mae = mean_absolute_error(hold_y, pred_y)
    mape = mean_absolute_percentage_error(hold_y, pred_y)
    mse = mean_squared_error(hold_y, pred_y)
    print(f"Holdout R^2: {r2}")
    print(f"Holdout MAE: {mae}")
    print(f"Holdout MAPE: {mape}")
    print(f"Holdout MSE: {mse}")

    # K-fold cross validation, with a default of 5 folds
    score: np.ndarray = cross_val_score(model, train_X, train_y, scoring="r2")
    print(score, score.mean(), score.std())

from typing import Any, Callable, Tuple
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


def cross_validate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    cv: int,
) -> Tuple[np.ndarray, Any]:
    """
    Cross validate the given model, returning the scores from evaluating each fold and the
    best trained estimator.

    :param model: the model to train; calling fit() on it should reset it and fit it from
                  scratch
    :type model: Any
    :param X: the features to fit the model on
    :type X: pd.DataFrame
    :param y: the dependent variable to predict
    :type y: pd.Series
    :param scoring: the function to use to score the model
    :type scoring: Callable[[np.ndarray, np.ndarray], float]
    :param cv: how many folds to use
    :type cv: int
    :return: the scores from each fold along with the best trained estimator
    :rtype: Tuple[np.ndarray, Any]
    """
    r2 = np.empty(cv)
    models = []

    fold = 0
    folds = KFold(n_splits=cv)
    for train, test in folds.split(X):
        train_X, train_y = X.iloc[train], y.iloc[train]
        test_X, test_y = X.iloc[test], y.iloc[test]

        model.fit(train_X, train_y)
        test_pred = model.predict(test_X)
        test_score = scoring(test_y, test_pred)

        r2[fold] = test_score
        models.append(model)
        fold += 1

    best = models[np.argmax(r2)]  # type: ignore

    return r2, best

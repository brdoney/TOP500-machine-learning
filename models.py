from typing import Dict

import tensorflow.keras as keras
from tensorflow.keras import layers
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from data_cleaning import Transformer


class DNN1(BaseEstimator):
    """
    Implements the deep neural network described in the study 'Predicting New Workload or
    CPU Performance by Analyzing Public Datasets' by Yu Wang, Victor Lee, Gu-Yeon Wei, and
    David Brooks.

    Uses the Keras library, but is designed to use Scikit-Learn style methods.
    """

    def reset(self):
        self.dnn = keras.Sequential(
            [
                layers.Dense(100, activation='tanh', name='first'),  # first hidden layer
                # second hidden layer
                layers.Dense(100, activation='tanh', name='second'),
                layers.Dense(100, activation='tanh', name='third'),  # third hidden layer
                layers.Dense(1, name='fourth')  # output layer
            ])
        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        self.dnn.compile(optimizer=optimizer, loss=keras.losses.mean_absolute_error)

    def fit(self, x, y):
        # We need to make sure we're working from scratch each time fit is called
        self.reset()

        self.dnn.fit(x, y, epochs=300, verbose=0)
        return self

    def predict(self, x):
        pred_y = self.dnn.predict(x)
        return pred_y


class DNN2(BaseEstimator):
    """
    Implements the deep neural network described in the study 'Benchmarking Machine
    Learning Methods for Performance Modeling of Scientific Applications' by Preeti
    Malakar, Prasanna Balaprakash, Venkatram Vishwanath, et al.

    Uses the Keras library, but is designed to use Scikit-Learn style methods.
    """

    def reset(self):
        nodes_per_layer = 512
        act = 'relu'
        drop_rate = 0.2

        self.dnn = keras.Sequential(
            [
                layers.Dense(nodes_per_layer, activation=act, name='first'),
                layers.Dropout(rate=drop_rate),
                layers.Dense(nodes_per_layer, activation=act, name='second'),
                layers.Dropout(rate=drop_rate),
                layers.Dense(nodes_per_layer, activation=act, name='third'),
                layers.Dropout(rate=drop_rate),
                layers.Dense(1, name='fourth')
            ])
        self.dnn.compile(optimizer="adam", loss=keras.losses.mean_squared_error)

    def fit(self, x, y):
        # We need to make sure we're working from scratch each time fit is called
        self.reset()

        self.dnn.fit(x, y, epochs=100, verbose=0, batch_size=512)
        return self

    def predict(self, x):
        pred_y = self.dnn.predict(x)
        return pred_y


# TODO: Find source that justifies which models are chosen
models = {
    "lr_1": LinearRegression(),

    "knn_1": KNeighborsRegressor(n_neighbors=3, weights="distance", p=2),
    "knn_2": KNeighborsRegressor(n_neighbors=5, weights="distance", p=2),
    "knn_3": KNeighborsRegressor(n_neighbors=7, weights="distance", p=2),
    "knn_4": KNeighborsRegressor(n_neighbors=5, weights="distance", p=1),
    "knn_5": KNeighborsRegressor(n_neighbors=5, weights="distance", p=3),
    "knn_tuned": KNeighborsRegressor(n_neighbors=5, weights="distance", p=1),

    "svr_1": SVR(kernel="rbf"),
    "svr_2": SVR(kernel="poly"),
    "svr_tuned": SVR(kernel="rbf", epsilon=0.2, gamma=0.021544346900318846, cache_size=1000),

    "rf_1": RandomForestRegressor(n_estimators=100, max_depth=None),
    "rf_2": RandomForestRegressor(n_estimators=1000, max_depth=None),
    "rf_3": RandomForestRegressor(n_estimators=100, max_depth=5),
    "rf_tuned": RandomForestRegressor(n_estimators=5000, max_depth=None, max_features="auto"),

    "gbt_1": GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=1),
    "gbt_2": GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=2),
    "gbt_3": GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3),
    "gbt_4": GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=4),
    "gbt_5": GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000, max_depth=5),

    "mlp_1": MLPRegressor(),

    "dnn1_1": DNN1(),
    "dnn2_1": DNN2(),

    "xgb_1": XGBRegressor(),

    "lgbm_1": LGBMRegressor(),
}

model_classes = {k: type(v) for k, v in models.items()}

scalers: Dict[str, Transformer] = {
    "Robust": RobustScaler(),
    "Standard": StandardScaler(),
    "MinMax": MinMaxScaler()
}

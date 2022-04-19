from pathlib import Path
import pickle
from lightgbm import LGBMRegressor

from xgboost import XGBRegressor
from models import DNN1, DNN2


ML_MODEL_DIR = Path("./out/models")
"""
Model directory contains packages for each model which contains:
- model-name/  (the name of the model will be indicated by the directory name)
  - rmax.* (model file, see [1])
  - efficiency.* (model file, see [1])
  - rmax-preprocessor.pkl (the preprocessor/cleaner to use w/ the rmax model)
  - efficiency-preprocessor.pkl (the preprocessor/cleaner to use w/ the efficiency model)

[1]: Model files can be in one of the following formats. Extensions are important; the
file will be deserialised into the correct format based on the extension.
- model.pkl (SKLearn models)
- model.xgb (XGBoost)
- model folder (SavedModel w/ Tensorflow)
- model.lgbm (LightGBM)
"""


def save_model(name: str, model, cleaner, dep_var):
    if dep_var == "Log(Rmax)":
        dep_var = "rmax"
    elif dep_var == "Log(Efficiency)":
        dep_var = "efficiency"

    model_dir = ML_MODEL_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    mtype = type(model)
    print(mtype)
    if mtype is DNN1 or mtype is DNN2:
        model.dnn.save(model_dir / dep_var)
    elif mtype is XGBRegressor:
        dest = model_dir / f"{dep_var}.xgb"
        print(f"XGBoost: Model written to {dest}")
        model.save_model(dest)
    elif mtype is LGBMRegressor:
        dest = str(model_dir / f"{dep_var}.lgbm")  # type: ignore
        print(f"LightGBM: Model written to {dest}")
        model.booster_.save_model(dest)
    else:
        dest = model_dir / f"{dep_var}.pkl"
        print(f"Sklearn: Model written to {dest}")
        pickle.dump(model, open(dest, "wb"))

    pickle.dump(cleaner, open(model_dir / f"{dep_var}-preprocessor.pkl", "wb"))

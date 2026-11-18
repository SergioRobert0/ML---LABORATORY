import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


REGRESSORS = {
    "Regressão Linear": LinearRegression,
    "Árvore de Decisão": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "Ridge": Ridge,
    "Lasso": Lasso,
}
CLASSIFIERS = {
    "Regressão Logística": LogisticRegression,
    "Árvore de Decisão": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "KNN": KNeighborsClassifier,
}


def infer_task(y):
    if not pd.api.types.is_numeric_dtype(y) or y.nunique(dropna=True) <= max(20, int(len(y) * .05)):
        return "classification"
    return "regression"


def make_preprocessor(X, scaler="StandardScaler"):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    scaler_obj = StandardScaler() if scaler == "StandardScaler" else MinMaxScaler()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler_obj)])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])


def make_model(task, name, params=None):
    params = dict(params or {})
    if "random_state" in (REGRESSORS if task == "regression" else CLASSIFIERS)[name]().get_params():
        params.setdefault("random_state", 42)
    if task == "classification" and name == "Regressão Logística":
        params.setdefault("max_iter", 2000)
    return (REGRESSORS if task == "regression" else CLASSIFIERS)[name](**params)


def make_pipeline(X, task, model_name, scaler="StandardScaler", params=None):
    return Pipeline([("preprocessor", make_preprocessor(X, scaler)), ("model", make_model(task, model_name, params))])


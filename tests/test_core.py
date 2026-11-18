import pandas as pd
from sklearn.model_selection import train_test_split

from ml_lab.data import dataframe_to_bytes, demo_dataset
from ml_lab.evaluation import regression_metrics
from ml_lab.modeling import make_pipeline


def test_demo_classification_loads():
    df, target, task = demo_dataset("Câncer de mama (classificação)")
    assert task == "classification" and target in df and len(df) > 100


def test_pipeline_handles_missing_and_category():
    X = pd.DataFrame({"idade": [20, 30, None, 50, 60, 70], "cidade": ["A", "B", "A", None, "B", "A"]})
    y = pd.Series([1, 2, 3, 4, 5, 6])
    model = make_pipeline(X, "regression", "Regressão Linear")
    model.fit(X, y)
    assert len(model.predict(X)) == 6


def test_regression_metrics_and_export():
    df, target, _ = demo_dataset("Progressão de diabetes (regressão)")
    X, y = df.drop(columns=target), df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
    model = make_pipeline(Xtr, "regression", "Ridge")
    model.fit(Xtr, ytr)
    metrics, pred = regression_metrics(model, Xtr, ytr, Xte, yte)
    assert "RMSE" in metrics and len(pred) == len(yte)
    assert dataframe_to_bytes(pd.DataFrame({"p": pred}), "JSON")

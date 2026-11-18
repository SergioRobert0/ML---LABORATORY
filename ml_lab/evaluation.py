import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)


def regression_metrics(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    pred = model.predict(X_test)
    return {
        "RMSE": mean_squared_error(y_test, pred) ** .5,
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "R²": r2_score(y_test, pred),
        "Treino R²": r2_score(y_train, train_pred),
    }, pred


def classification_metrics(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    pred = model.predict(X_test)
    avg = "weighted"
    metrics = {
        "Acurácia": accuracy_score(y_test, pred),
        "Precisão": precision_score(y_test, pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, pred, average=avg, zero_division=0),
        "Treino acurácia": accuracy_score(y_train, train_pred),
    }
    return metrics, pred, confusion_matrix(y_test, pred), classification_report(y_test, pred, zero_division=0)


def generalization_note(task, metrics):
    train = metrics["Treino R²"] if task == "regression" else metrics["Treino acurácia"]
    test = metrics["R²"] if task == "regression" else metrics["Acurácia"]
    gap = train - test
    if gap > .15:
        return "Possível overfitting: o desempenho de treino está bem acima do teste, indicando variância elevada."
    if train < .6 and test < .6:
        return "Possível underfitting: treino e teste apresentam desempenho baixo, indicando viés elevado."
    return "Treino e teste estão relativamente próximos, sinal favorável de generalização."


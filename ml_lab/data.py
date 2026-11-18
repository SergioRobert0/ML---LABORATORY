import io
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes


def load_uploaded(file, separator=","):
    suffix = Path(file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file)
    if suffix == ".txt":
        return pd.read_csv(file, sep=separator)
    if suffix == ".json":
        raw = file.getvalue().decode("utf-8")
        obj = json.loads(raw)
        return pd.json_normalize(obj)
    raise ValueError("Formato não suportado. Use CSV, TXT ou JSON.")


def demo_dataset(name):
    if name == "Preços de imóveis (regressão)":
        return pd.DataFrame({
            "area_m2": [45, 52, 60, 68, 75, 82, 90, 105, 120, 140, 160, 185],
            "quartos": [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5],
            "idade_imovel": [28, 20, 18, 15, 12, 10, 8, 7, 6, 5, 3, 2],
            "distancia_centro_km": [12, 10, 9, 8, 8, 7, 6, 5, 5, 4, 3, 2],
            "preco_mil": [155, 190, 220, 255, 295, 330, 370, 430, 510, 600, 710, 850],
        }), "preco_mil", "regression"
    if name == "Câncer de mama (classificação)":
        bunch = load_breast_cancer(as_frame=True)
        return bunch.frame, "target", "classification"
    if name == "Progressão de diabetes (regressão)":
        bunch = load_diabetes(as_frame=True)
        return bunch.frame, "target", "regression"
    if name == "Satisfação populacional (regressão)":
        return pd.DataFrame({
            "pib_per_capita": [12000, 18000, 23000, 28000, 33000, 39000, 45000, 52000, 60000, 70000],
            "expectativa_vida": [67, 69, 71, 72, 74, 75, 77, 78, 79, 81],
            "desemprego": [14, 13, 12, 11, 10, 9, 8, 7, 6, 5],
            "satisfacao": [4.1, 4.5, 4.9, 5.2, 5.7, 6.0, 6.4, 6.8, 7.2, 7.7],
        }), "satisfacao", "regression"
    raise ValueError("Dataset demonstrativo desconhecido.")


def dataframe_to_bytes(df, fmt):
    if fmt == "CSV":
        return df.to_csv(index=False).encode("utf-8")
    if fmt == "JSON":
        return df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")
    return df.to_csv(index=False, sep="\t").encode("utf-8")


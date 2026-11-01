"""Nível B: componentes NÃO ajustados; o chamador deve separar treino/teste antes do fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .preprocessing import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


def select_features(data):
    """Lista positiva de dez preditores, igual para classificação e agrupamento."""
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
        raise ValueError("Forneça um DataFrame com colunas únicas.")
    missing = set(FEATURE_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Preditores ausentes: {sorted(missing)}")
    selected = data.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in NUMERIC_FEATURES + BINARY_FEATURES:
        selected[column] = pd.to_numeric(selected[column], errors="raise").astype(float)
    for column in CATEGORICAL_FEATURES:
        selected[column] = selected[column].astype(object).where(selected[column].notna(), np.nan)
    return selected


def supervised_data(data):
    """Recusa alvos desconhecidos: não os transforma em classe zero nem em preditores."""
    if "churn" not in data or data["churn"].isna().any() or not data["churn"].isin([0, 1]).all():
        raise ValueError("A amostra supervisionada precisa ter churn conhecido em {0,1}.")
    return select_features(data), data["churn"].astype(int).copy()


class FeatureSelector(TransformerMixin, BaseEstimator):
    """Descarta metadados por contrato e evita inventar mediana/moda em treino inteiramente vazio."""

    def __init__(self, *, require_observed_categories=False):
        self.require_observed_categories = require_observed_categories

    def fit(self, X, y=None):
        selected = select_features(X)
        if selected.empty:
            raise ValueError("Não é possível ajustar em treino vazio.")
        required = NUMERIC_FEATURES + BINARY_FEATURES
        if self.require_observed_categories:
            required += CATEGORICAL_FEATURES
        empty = [c for c in required if selected[c].isna().all()]
        if empty:
            raise ValueError(f"Treino sem valores disponíveis para imputação em {empty}.")
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return select_features(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_in_.copy()


def build_ml_preprocessor(*, scaling="none", categorical_imputation="constant"):
    """Retorna pipeline sem fit. Árvores: none; métodos de distância: avaliar standard/minmax.

    Medianas e modas (quando solicitadas), vocabulário do encoder e escalas são
    aprendidos exclusivamente no fit. Por padrão, categoria ausente tem um marcador
    próprio, evitando substituir desconhecidos por uma categoria de negócio válida.
    """
    if scaling not in {"none", "standard", "minmax"}:
        raise ValueError("scaling deve ser none, standard ou minmax.")
    if categorical_imputation not in {"constant", "most_frequent"}:
        raise ValueError("Estratégia categórica inválida.")
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scaling == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax":
        numeric_steps.append(("scaler", MinMaxScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy=categorical_imputation, fill_value="[ausente]", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    columns = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            ("categorical", Pipeline(categorical_steps), list(CATEGORICAL_FEATURES)),
            ("binary", SimpleImputer(strategy="most_frequent", keep_empty_features=True), list(BINARY_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    selector = FeatureSelector(require_observed_categories=categorical_imputation == "most_frequent")
    return Pipeline([("features", selector), ("columns", columns)])

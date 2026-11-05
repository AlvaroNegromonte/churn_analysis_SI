"""Nível B: componentes NÃO ajustados; o chamador deve separar treino/teste antes do fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .preprocessing import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from .imputation import CustomerImputer


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


def build_ml_preprocessor(*, scaling="none", categorical_imputation="most_frequent",
                          numeric_imputation="median", n_neighbors=5):
    """Retorna pipeline sem fit. Árvores: none; métodos de distância: avaliar standard/minmax.

    Medianas e modas (quando solicitadas), vocabulário do encoder e escalas são
    aprendidos exclusivamente no fit. Moda categórica é a política padrão autorizada.
    O marcador constante permanece como alternativa explícita. KNN pode substituir
    mediana; a escolha deve ocorrer exclusivamente dentro do treino.
    """
    if scaling not in {"none", "standard", "minmax"}:
        raise ValueError("scaling deve ser none, standard ou minmax.")
    if categorical_imputation not in {"constant", "most_frequent"}:
        raise ValueError("Estratégia categórica inválida.")
    if numeric_imputation not in {"median", "knn"}:
        raise ValueError("numeric_imputation deve ser median ou knn.")
    numeric_transformer = "passthrough"
    if scaling == "standard":
        numeric_transformer = Pipeline([("scaler", StandardScaler())])
    elif scaling == "minmax":
        numeric_transformer = Pipeline([("scaler", MinMaxScaler())])
    columns = ColumnTransformer(
        [
            ("numeric", numeric_transformer, list(NUMERIC_FEATURES)),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL_FEATURES)),
            ("binary", "passthrough", list(BINARY_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    selector = FeatureSelector(require_observed_categories=categorical_imputation == "most_frequent")
    imputation = CustomerImputer(numeric_strategy=numeric_imputation,
                                 categorical_strategy=categorical_imputation,
                                 n_neighbors=n_neighbors)
    return Pipeline([("features", selector), ("imputation", imputation), ("columns", columns)])

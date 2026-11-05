"""Imputação aprendida no treino; estimativas não substituem a base canônica."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .preprocessing import (
    BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_DOMAINS,
    NUMERIC_FEATURES,
)

SUPPORT_COLUMNS = ("chamados_30d", "chamados_60d", "chamados_90d")


def predictor_frame(data):
    """Seleciona apenas preditores; id, datas, alvo e flags não afetam vizinhanças."""
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
        raise ValueError("Forneça um DataFrame com colunas únicas.")
    missing = set(FEATURE_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Preditores ausentes: {sorted(missing)}")
    result = data.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in NUMERIC_FEATURES + BINARY_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        present = result[column].dropna()
        low, high, integer = NUMERIC_DOMAINS[column]
        if not present.between(low, high).all() or (integer and present.mod(1).ne(0).any()):
            raise ValueError(f"Use dados canônicos: domínio inválido em {column}.")
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    for _, row in result.loc[:, list(SUPPORT_COLUMNS)].iterrows():
        known = row.dropna().to_numpy()
        if (np.diff(known) < 0).any():
            raise ValueError("Contagens observadas de suporte não respeitam 30 <= 60 <= 90 dias.")
    return result


class CustomerImputer(TransformerMixin, BaseEstimator):
    """Mediana ou KNN escalado para numéricos; moda para categóricos/binários.

    KNN usa também binários e categorias codificadas para definir semelhança.
    As três janelas de suporte recebem peso conjunto de um atributo numérico
    na distância, reduzindo a repetição de informação sobre os mesmos eventos.
    Somente células originalmente ausentes são preenchidas ou arredondadas.
    """

    def __init__(self, *, numeric_strategy="median", categorical_strategy="most_frequent",
                 n_neighbors=5):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        if self.numeric_strategy not in {"median", "knn"}:
            raise ValueError("numeric_strategy deve ser median ou knn.")
        if self.categorical_strategy not in {"most_frequent", "constant"}:
            raise ValueError("Estratégia categórica inválida.")
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("n_neighbors deve ser inteiro positivo.")
        frame = predictor_frame(X)
        if frame.empty:
            raise ValueError("Não é possível ajustar em treino vazio.")
        required = NUMERIC_FEATURES + BINARY_FEATURES
        if self.categorical_strategy == "most_frequent":
            required += CATEGORICAL_FEATURES
        empty = [c for c in required if frame[c].isna().all()]
        if empty:
            raise ValueError(f"Treino sem valores disponíveis para imputação em {empty}.")
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.categorical_imputer_ = SimpleImputer(
            strategy=self.categorical_strategy, fill_value="[ausente]", keep_empty_features=True,
        ).fit(frame.loc[:, list(CATEGORICAL_FEATURES)])
        self.binary_imputer_ = SimpleImputer(strategy="most_frequent").fit(
            frame.loc[:, list(BINARY_FEATURES)]
        )
        numeric = frame.loc[:, list(NUMERIC_FEATURES)]
        if self.numeric_strategy == "median":
            self.numeric_imputer_ = SimpleImputer(strategy="median").fit(numeric)
        else:
            self.distance_scaler_ = StandardScaler().fit(numeric)
            self.distance_weights_ = np.asarray([
                1 / np.sqrt(3) if c in SUPPORT_COLUMNS else 1 for c in NUMERIC_FEATURES
            ])
            categories = self.categorical_imputer_.transform(frame.loc[:, list(CATEGORICAL_FEATURES)])
            self.distance_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(categories)
            self.numeric_imputer_ = KNNImputer(n_neighbors=self.n_neighbors, weights="uniform").fit(
                self._distance_matrix(frame)
            )
        return self

    def _distance_matrix(self, frame):
        numeric = self.distance_scaler_.transform(frame.loc[:, list(NUMERIC_FEATURES)])
        categories = self.categorical_imputer_.transform(frame.loc[:, list(CATEGORICAL_FEATURES)])
        binary = self.binary_imputer_.transform(frame.loc[:, list(BINARY_FEATURES)])
        return np.column_stack([
            numeric * self.distance_weights_, binary,
            self.distance_encoder_.transform(categories) / np.sqrt(2),
        ])

    def transform(self, X):
        check_is_fitted(self)
        frame = predictor_frame(X)
        if frame.empty:
            return frame
        result = frame.copy()
        if self.numeric_strategy == "median":
            estimates = self.numeric_imputer_.transform(frame.loc[:, list(NUMERIC_FEATURES)])
        else:
            distances = self.numeric_imputer_.transform(self._distance_matrix(frame))
            scaled = distances[:, :len(NUMERIC_FEATURES)] / self.distance_weights_
            estimates = self.distance_scaler_.inverse_transform(scaled)
        for position, column in enumerate(NUMERIC_FEATURES):
            absent = frame[column].isna()
            low, high, integer = NUMERIC_DOMAINS[column]
            values = np.clip(estimates[:, position], low, high)
            if integer:
                values = np.floor(values + 0.5)
            result.loc[absent, column] = values[absent.to_numpy()]
        for columns, imputer in [
            (CATEGORICAL_FEATURES, self.categorical_imputer_),
            (BINARY_FEATURES, self.binary_imputer_),
        ]:
            estimates = imputer.transform(frame.loc[:, list(columns)])
            for position, column in enumerate(columns):
                absent = frame[column].isna()
                result.loc[absent, column] = estimates[absent.to_numpy(), position]
        # Limites dados por contagens observadas e por estimativas anteriores.
        # Nenhuma célula observada é alterada para resolver inconsistências.
        for index in result.index:
            for position, column in enumerate(SUPPORT_COLUMNS):
                if pd.notna(frame.at[index, column]):
                    continue
                lower = result.at[index, SUPPORT_COLUMNS[position - 1]] if position else 0
                later = frame.loc[index, list(SUPPORT_COLUMNS[position + 1:])].dropna()
                upper = later.min() if len(later) else 50
                result.at[index, column] = np.clip(result.at[index, column], lower, upper)
        if result.isna().any().any():
            raise ValueError("A imputação deixou preditores sem estimativa.")
        return result.loc[:, list(FEATURE_COLUMNS)]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_in_.copy()

    def missing_indicators(self, X):
        """Proveniência por célula; não se confunde estimativa com valor observado."""
        return predictor_frame(X).isna().astype(int).add_prefix("imputed_")

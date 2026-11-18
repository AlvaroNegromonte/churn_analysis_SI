"""Etapa 3 (miniprojeto 2): utilitários de modelagem com Árvore de Decisão e XGBoost.

Contratos principais
--------------------
* A base de entrada é a coorte principal determinística do miniprojeto 1
  (``data/processed/churn_main.csv``), ainda com ausências auditadas.
* O split é 70/30 estratificado, semente 42, feito ANTES de qualquer ajuste.
* Todo componente que aprende parâmetros (imputação KNN/mediana, moda,
  vocabulário do one-hot) fica dentro de um ``Pipeline`` e, portanto, é
  reajustado em cada fold da validação cruzada — sem data leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, fbeta_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import _tree

from .ml_preprocessing import build_ml_preprocessor
from .preprocessing import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "data" / "processed" / "churn_main.csv"
MODELING_DIR = ROOT / "data" / "processed" / "modeling"
RANDOM_STATE = 42
TEST_SIZE = 0.30


# --------------------------------------------------------------------------- dados
def load_main(path: Path | str = MAIN_PATH) -> pd.DataFrame:
    """Lê a coorte principal; exige alvo binário conhecido e ids únicos."""
    data = pd.read_csv(path)
    if data["id_cliente"].duplicated().any():
        raise ValueError("id_cliente duplicado na coorte principal.")
    if data["churn"].isna().any() or not data["churn"].isin([0, 1]).all():
        raise ValueError("Alvo desconhecido na coorte principal.")
    return data


def split_70_30(data: pd.DataFrame, *, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split estratificado; devolve X com metadados (o pipeline seleciona os preditores)."""
    train, test = train_test_split(
        data, test_size=test_size, stratify=data["churn"], random_state=random_state,
    )
    if set(train["id_cliente"]) & set(test["id_cliente"]):
        raise AssertionError("Sobreposição de clientes entre treino e teste.")
    X_train = train.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True)
    X_test = test.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True)
    y_train = train["churn"].astype(int).reset_index(drop=True)
    y_test = test["churn"].astype(int).reset_index(drop=True)
    ids = {"train": train["id_cliente"].reset_index(drop=True),
           "test": test["id_cliente"].reset_index(drop=True)}
    return X_train, X_test, y_train, y_test, ids


# ----------------------------------------------------------------- engenharia
def clean_name(name: str) -> str:
    """Remove prefixos do ColumnTransformer: numeric__x -> x."""
    return name.split("__", 1)[-1]


class EngineeredFeatures(TransformerMixin, BaseEstimator):
    """Hipótese da equipe: atributos derivados após a imputação (sem parâmetros aprendidos).

    * ``aceleracao_suporte``: fração dos chamados de 90 dias concentrada nos últimos 30.
    * ``chamados_recentes_60_90``: chamados entre 61 e 90 dias antes do corte.
    * ``valor_acumulado``: mensalidade × permanência (proxy de valor do cliente).
    * ``tem_atraso``: indicador de pelo menos um atraso em 6 meses.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        cols = {clean_name(c): c for c in X.columns}
        out = X.copy()
        c30, c90 = X[cols["chamados_30d"]], X[cols["chamados_90d"]]
        out["eng__aceleracao_suporte"] = c30 / (c90 + 1.0)
        out["eng__chamados_61_90d"] = (c90 - X[cols["chamados_60d"]]).clip(lower=0)
        out["eng__valor_acumulado"] = X[cols["mensalidade_rs"]] * X[cols["tempo_cliente_meses"]]
        out["eng__tem_atraso"] = (X[cols["atrasos_6m"]] > 0).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + [
            "eng__aceleracao_suporte", "eng__chamados_61_90d",
            "eng__valor_acumulado", "eng__tem_atraso",
        ], dtype=object)


def make_pipeline(model, *, numeric_imputation="knn", engineered=False, memory=None):
    """Pré-processamento do miniprojeto 1 + (opcional) atributos derivados + modelo."""
    steps = [("prep", build_ml_preprocessor(scaling="none", numeric_imputation=numeric_imputation))]
    if engineered:
        steps.append(("eng", EngineeredFeatures()))
    steps.append(("model", model))
    pipe = Pipeline(steps, memory=memory)
    pipe.set_output(transform="pandas")
    return pipe


def feature_names(pipe) -> list[str]:
    """Nomes legíveis das colunas que chegam ao modelo."""
    names = pipe[:-1].get_feature_names_out()
    return [clean_name(n) for n in names]


# --------------------------------------------------------------------- métricas
def metrics_at(y_true, proba, threshold=0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def threshold_for_recall(y_true, proba, target_recall):
    """Maior limiar cujo recall (em dados de treino/OOF) ainda atinge a meta."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    best = None
    for t in np.unique(np.round(proba, 4))[::-1]:
        if recall_score(y_true, (proba >= t).astype(int)) >= target_recall:
            best = float(t)
            break
    return best if best is not None else 0.0


def threshold_min_cost(y_true, proba, cost_fn, cost_fp):
    """Limiar que minimiza custo esperado FN·c_fn + FP·c_fp (hipótese de custos)."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in grid:
        pred = (proba >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        costs.append(fn * cost_fn + fp * cost_fp)
    costs = np.asarray(costs)
    return float(grid[costs.argmin()]), grid, costs


# ------------------------------------------------------------------- regras
def tree_rules(tree, names, *, max_depth=None, min_samples=1, X=None, y=None) -> pd.DataFrame:
    """Extrai regras SE-ENTÃO de cada folha (ou nó no limite ``max_depth``).

    Retorna condição, suporte (n e %), taxa REAL de churn no nó e classe prevista.

    Com ``class_weight="balanced"`` os valores internos da árvore (``tree_.value``)
    são ponderados e superestimam a taxa da classe minoritária. Por isso, quando
    ``X``/``y`` (treino transformado) são fornecidos, a taxa e o suporte vêm das
    contagens reais de clientes que passam por cada nó (``decision_path``):
    P(churn | folha) = churners na folha / clientes na folha (definição da aula).
    A coluna ``ENTÃO`` segue a decisão do próprio modelo (limiar 0,5 ponderado).
    """
    t = tree.tree_
    if X is not None and y is not None:
        path = tree.decision_path(X)
        n_node = np.asarray(path.sum(axis=0)).ravel().astype(int)
        pos_node = np.asarray(path.T @ np.asarray(y)).ravel()
    else:
        n_node = t.n_node_samples
        pos_node = None
    total = n_node[0]
    rows = []

    def visit(node, conds, depth):
        is_leaf = t.children_left[node] == _tree.TREE_LEAF
        if is_leaf or (max_depth is not None and depth == max_depth):
            value = t.value[node][0]
            weighted = value[1] / value.sum()
            n = int(n_node[node])
            rate = (pos_node[node] / n if n else np.nan) if pos_node is not None else weighted
            if n >= min_samples:
                rows.append({
                    "regra": "SE " + (" E ".join(conds) if conds else "(raiz)"),
                    "ENTÃO": "CHURN" if weighted >= 0.5 else "NÃO CHURN",
                    "taxa_churn_no": rate,
                    "n_clientes": n,
                    "suporte_%": 100 * n / total,
                    "profundidade": depth,
                })
            return
        name = names[t.feature[node]]
        thr = t.threshold[node]
        if _is_binary_name(name):  # limiar 0,5 em atributo 0/1
            left, right = f"{name} = 0", f"{name} = 1"
        else:
            left, right = f"{name} <= {thr:.2f}", f"{name} > {thr:.2f}"
        visit(t.children_left[node], conds + [left], depth + 1)
        visit(t.children_right[node], conds + [right], depth + 1)

    visit(0, [], 0)
    return (pd.DataFrame(rows)
            .sort_values(["taxa_churn_no", "n_clientes"], ascending=[False, False])
            .reset_index(drop=True))


def _is_binary_name(name: str) -> bool:
    return (name.startswith(("tipo_contrato_", "forma_pagamento_"))
            or name in {"fibra_optica", "debito_automatico", "tem_atraso"})


# ------------------------------------------------------------- base transformada
def export_transformed(pipe_prep, X_train, X_test, y_train, y_test, ids, path):
    """Exporta a base transformada (prep ajustado SÓ no treino) com partição e flags."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pipe_prep = pipe_prep.fit(X_train, y_train)
    parts = []
    for part, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        Xt = pipe_prep.transform(X).copy()
        Xt.columns = [clean_name(c) for c in Xt.columns]
        flags = X.isna().astype(int).add_prefix("imputed_").reset_index(drop=True)
        frame = pd.concat([ids[part].rename("id_cliente"), Xt.reset_index(drop=True), flags,
                           y.rename("churn").reset_index(drop=True)], axis=1)
        frame.insert(1, "partition", part)
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    result.to_csv(path, index=False)
    return result

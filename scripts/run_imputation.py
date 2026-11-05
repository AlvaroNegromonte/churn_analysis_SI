"""Compara imputadores no treino e exporta uma visão preenchida com proveniência."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .imputation import CustomerImputer, SUPPORT_COLUMNS, predictor_frame
from .preprocessing import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, prepare_data,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "base_churn_5000_VF2_aluno.xlsx - Dados_Sinteticos_VF2.csv"


@dataclass(frozen=True)
class ImputationConfig:
    random_state: int = 42
    test_size: float = 0.2
    folds: int = 3
    mask_fraction: float = 0.1
    n_neighbors: int = 5
    min_relative_improvement: float = 0.01

    def __post_init__(self):
        if not 0 < self.test_size < 1 or not 0 < self.mask_fraction < 1:
            raise ValueError("Proporções devem estar entre 0 e 1.")
        if not isinstance(self.folds, int) or self.folds < 2:
            raise ValueError("Informe pelo menos dois folds.")
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("n_neighbors deve ser inteiro positivo.")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("Margem de melhoria inválida.")


def _validation_mask(frame, numeric_targets, rng, fraction):
    mask = pd.DataFrame(False, index=frame.index, columns=FEATURE_COLUMNS)
    for column in [c for c in numeric_targets if c not in SUPPORT_COLUMNS] + list(CATEGORICAL_FEATURES):
        observed = frame.index[frame[column].notna()].to_numpy()
        if len(observed):
            chosen = rng.choice(observed, size=max(1, int(len(observed) * fraction)), replace=False)
            mask.loc[chosen, column] = True
    if any(c in numeric_targets for c in SUPPORT_COLUMNS):
        # Os históricos comprometidos reais perdem as três contagens em conjunto.
        observed = frame.index[frame.loc[:, list(SUPPORT_COLUMNS)].notna().all(axis=1)].to_numpy()
        if len(observed):
            chosen = rng.choice(observed, size=max(1, int(len(observed) * fraction)), replace=False)
            mask.loc[chosen, list(SUPPORT_COLUMNS)] = True
    return mask


def evaluate_imputers(training, config):
    """Validação interna: nenhum dado de teste ou rótulo entra no ajuste/score."""
    features = predictor_frame(training)
    targets = [c for c in NUMERIC_FEATURES if features[c].isna().any()]
    if not targets:
        raise ValueError("Não há ausências numéricas no treino para comparar estratégias.")
    records, masks, fold_scores = [], [], []
    splitter = KFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
    for fold, (fit_positions, validation_positions) in enumerate(splitter.split(features), start=1):
        fitting = features.iloc[fit_positions]
        validation = features.iloc[validation_positions]
        rng = np.random.default_rng(config.random_state + fold)
        mask = _validation_mask(validation, targets, rng, config.mask_fraction)
        masked = validation.mask(mask)
        for column in mask:
            for index in mask.index[mask[column]]:
                masks.append({"fold": fold, "id_cliente": training.at[index, "id_cliente"],
                              "field": column, "observed_value": validation.at[index, column]})
        for strategy in ("median", "knn"):
            imputer = CustomerImputer(numeric_strategy=strategy, n_neighbors=config.n_neighbors).fit(fitting)
            estimates = imputer.transform(masked)
            errors = {}
            for column in targets + list(CATEGORICAL_FEATURES):
                selected = mask[column]
                truth = validation.loc[selected, column]
                prediction = estimates.loc[selected, column]
                if truth.empty:
                    continue
                row = {"fold": fold, "strategy": strategy, "field": column, "masked_values": len(truth)}
                if column in NUMERIC_FEATURES:
                    difference = prediction.astype(float).to_numpy() - truth.astype(float).to_numpy()
                    scale = float(fitting[column].std(ddof=0))
                    scale = scale if scale > 0 else 1.0
                    mae = float(np.abs(difference).mean())
                    row.update(mae=mae, rmse=float(np.sqrt((difference ** 2).mean())),
                               training_std=scale, normalized_mae=mae / scale, accuracy=None)
                    errors[column] = mae / scale
                else:
                    row.update(mae=None, rmse=None, training_std=None, normalized_mae=None,
                               accuracy=float(prediction.eq(truth).mean()))
                records.append(row)
            # Peso igual para mensalidade, atrasos e o bloco de suporte, quando presentes.
            groups = [v for c, v in errors.items() if c not in SUPPORT_COLUMNS]
            support = [v for c, v in errors.items() if c in SUPPORT_COLUMNS]
            if support:
                groups.append(float(np.mean(support)))
            fold_scores.append({"fold": fold, "strategy": strategy, "score": float(np.mean(groups))})
    metrics = pd.DataFrame(records)
    scores = pd.DataFrame(fold_scores).groupby("strategy")["score"].mean().to_dict()
    margin = scores["median"] * (1 - config.min_relative_improvement)
    selected = "knn" if scores["knn"] < margin else "median"
    selection = {
        "selected_numeric_strategy": selected,
        "mean_internal_scores": scores,
        "fold_scores": fold_scores,
        "criterion": "Menor MAE normalizado, com peso igual por atributo/bloco; KNN exige melhoria relativa > margem configurada. Empates favorecem mediana.",
        "test_used_for_selection": False,
        "numeric_targets": targets,
    }
    return metrics, pd.DataFrame(masks), selection


def _complete(frame, partition, imputer):
    original = predictor_frame(frame)
    filled = imputer.transform(original)
    indicators = imputer.missing_indicators(original)
    result = frame.copy()
    result["partition"] = partition
    for column in FEATURE_COLUMNS:
        result[column] = filled[column]
    result = pd.concat([result, indicators], axis=1)
    issues = []
    for column in FEATURE_COLUMNS:
        observed = original[column].notna()
        if not filled.loc[observed, column].eq(original.loc[observed, column]).all():
            raise ValueError(f"Valor observado alterado em {column}.")
        for index in original.index[~observed]:
            value = filled.at[index, column]
            issues.append({
                "id_cliente": frame.at[index, "id_cliente"],
                "data_corte": frame.at[index, "data_corte"],
                "source_rows": frame.at[index, "source_rows"], "partition": partition,
                "field": column, "canonical_value": None, "imputed_value": value,
                "method": imputer.numeric_strategy if column in NUMERIC_FEATURES else "most_frequent",
                "support_status": frame.at[index, "status_suporte"] if column in SUPPORT_COLUMNS else "",
                "observed_value_recovered": False,
            })
    return result, issues


def run_imputation(source=SOURCE, output_dir=None, config=None):
    """Executa Nível B após split, sem modificar as saídas determinísticas do Nível A."""
    config = config or ImputationConfig()
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "data" / "processed" / "imputation"
    source = Path(source).resolve()
    source_hash = sha256(source.read_bytes()).hexdigest()
    prepared = prepare_data(source)
    main = prepared.main
    train_indices, test_indices = train_test_split(
        np.arange(len(main)), test_size=config.test_size,
        random_state=config.random_state, stratify=main["churn"],
    )
    training = main.iloc[np.sort(train_indices)].copy()
    testing = main.iloc[np.sort(test_indices)].copy()
    metrics, masks, selection = evaluate_imputers(training, config)
    imputer = CustomerImputer(numeric_strategy=selection["selected_numeric_strategy"],
                             n_neighbors=config.n_neighbors).fit(training)
    frames, audit = {}, []
    for name, frame, partition in [
        ("churn_imputed_train.csv", training, "train"),
        ("churn_imputed_test.csv", testing, "test"),
        ("churn_imputed_new_customers.csv", prepared.new_customers, "new_customers"),
    ]:
        completed, issues = _complete(frame, partition, imputer)
        frames[name] = completed
        audit.extend(issues)
    all_completed = pd.concat(list(frames.values()), ignore_index=True).sort_values(
        ["id_cliente", "data_corte"], kind="stable"
    ).reset_index(drop=True)
    if len(all_completed) != len(prepared.canonical) or all_completed.duplicated(["id_cliente", "data_corte"]).any():
        raise ValueError("Clientes perdidos ou duplicados na imputação.")
    if all_completed.loc[:, list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Preditores ausentes na visão preenchida.")
    frames["churn_imputed.csv"] = all_completed
    frames["imputation_audit.csv"] = pd.DataFrame(audit)
    frames["imputation_validation.csv"] = metrics
    frames["imputation_validation_masks.csv"] = masks
    before_after = {}
    for column in FEATURE_COLUMNS:
        before = prepared.canonical[column]
        after = all_completed[column]
        comparison = {"missing_before": int(before.isna().sum()), "missing_after": int(after.isna().sum()),
                      "imputed_values": int(all_completed[f"imputed_{column}"].sum())}
        if column in NUMERIC_FEATURES:
            comparison.update(mean_before=float(before.mean()), mean_after=float(after.mean()),
                              median_before=float(before.median()), median_after=float(after.median()))
        before_after[column] = comparison
    report = {
        "schema_version": 1, "config": asdict(config), "selection": selection,
        "source_sha256": source_hash,
        "runtime": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
        "counts": {"canonical": len(all_completed), "train": len(training), "test": len(testing),
                   "new_customers": len(prepared.new_customers), "imputed_cells": len(audit)},
        "before_after": before_after,
        "training_categories_modes": dict(zip(CATEGORICAL_FEATURES, imputer.categorical_imputer_.statistics_.tolist())),
        "fit_scope": "Somente treino da coorte principal; seleção por validação interna. Teste e novos recebem apenas transform.",
        "limitations": [
            "Mascaramento artificial de valores conhecidos não reproduz necessariamente o mecanismo das ausências reais.",
            "Estimativas não recuperam valores verdadeiros nem corrigem datas; flags e base canônica preservam a origem.",
            "KNN usa k=5 por padrão e distância padronizada; não houve busca pelo melhor k nem treinamento de classificador.",
            "Os 14 novos recebem transform do treino principal; isso não valida um modelo Early Churn.",
            "Ajustar novamente a imputação dentro de cada fold ao validar modelos futuros; não fazer CV diretamente sobre a visão já imputada com todo o treino externo.",
        ],
        "policies": {
            "categorical": "Moda do treino. Empates seguem a ordenação determinística do SimpleImputer.",
            "numeric": "Mediana ou KNN selecionado no treino; valores observados preservados.",
            "counts": "Arredondamento para inteiro mais próximo; empate x,5 para cima. Limites dos domínios aplicados apenas às estimativas.",
            "support": "Apenas contagens ausentes são ajustadas para 30 <= 60 <= 90, respeitando âncoras observadas.",
            "canonical": "Base canônica mantém NA e sinalizadores; a visão derivada contém estimativas identificadas.",
        },
    }
    destinations = [output_dir / name for name in [*frames, "imputation_summary.json"]]
    if source in [p.resolve() for p in destinations]:
        raise ValueError("Uma saída sobrescreveria a fonte.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8", lineterminator="\n", date_format="%Y-%m-%d")
    report["files"] = {name: sha256((output_dir / name).read_bytes()).hexdigest() for name in frames}
    (output_dir / "imputation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    if sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("A fonte foi modificada.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "imputation")
    args = parser.parse_args(argv)
    report = run_imputation(args.input, args.output_dir)
    print(f"Estratégia numérica selecionada: {report['selection']['selected_numeric_strategy']}")
    print(f"Categóricos: moda; clientes preservados: {report['counts']['canonical']}; células preenchidas: {report['counts']['imputed_cells']}")
    print("Ajuste somente no treino. Nenhum classificador ou agrupamento foi treinado; slides não foram alterados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

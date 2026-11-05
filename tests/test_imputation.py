"""Validação da imputação aprendida, dos domínios e da separação treino/teste."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from scripts.imputation import CustomerImputer, SUPPORT_COLUMNS
from scripts.ml_preprocessing import build_ml_preprocessor, select_features
from scripts.preprocessing import FEATURE_COLUMNS, NUMERIC_DOMAINS, prepare_data
from scripts.run_imputation import ImputationConfig, evaluate_imputers, run_imputation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "base_churn_5000_VF2_aluno.xlsx - Dados_Sinteticos_VF2.csv"


def rows():
    return pd.DataFrame({
        "tempo_cliente_meses": [12., 12., 12., 12.], "mensalidade_rs": [20., 40., 100., 120.],
        "atrasos_6m": [1., 1., 2., 2.],
        "chamados_30d": [0., 0., 2., 2.], "chamados_60d": [1., 1., 3., 3.],
        "chamados_90d": [2., 2., 4., 4.],
        "tipo_contrato": ["Mensal", "Mensal", "Mensal", "Anual"],
        "forma_pagamento": ["Pix", "Pix", "Boleto", "Pix"],
        "fibra_optica": [0., 0., 1., 1.], "debito_automatico": [0., 0., 1., 1.],
    })


@pytest.mark.parametrize("strategy", ["median", "knn"])
def test_fit_is_required_and_observed_values_are_never_changed(strategy):
    data = rows()
    imputer = CustomerImputer(numeric_strategy=strategy)
    with pytest.raises(NotFittedError):
        imputer.transform(data)
    fitted = clone(imputer).fit(data)
    pd.testing.assert_frame_equal(fitted.transform(data), data.loc[:, list(FEATURE_COLUMNS)])
    incoming = data.iloc[[0]].copy()
    incoming.loc[:, ["mensalidade_rs", "tipo_contrato", *SUPPORT_COLUMNS]] = np.nan
    filled = fitted.transform(incoming)
    assert not filled.isna().any().any()
    assert filled.tipo_contrato.iloc[0] == "Mensal"
    assert fitted.missing_indicators(incoming).sum(axis=1).iloc[0] == 5
    assert data.mensalidade_rs.tolist() == [20, 40, 100, 120]
    assert incoming.mensalidade_rs.isna().all()


def test_knn_uses_profile_and_ignores_target_and_metadata():
    training = rows()
    training.loc[:, "atrasos_6m"] = 1.
    training.loc[:, list(SUPPORT_COLUMNS)] = [0., 1., 2.]
    training.loc[:, "debito_automatico"] = 0.
    training.loc[:, "tipo_contrato"] = "Mensal"
    training.loc[:, "forma_pagamento"] = "Pix"
    incoming = training.iloc[[0]].copy()
    incoming.loc[:, "mensalidade_rs"] = np.nan
    incoming.loc[:, "fibra_optica"] = 1.
    fitted = CustomerImputer(numeric_strategy="knn", n_neighbors=2).fit(training)
    assert fitted.transform(incoming).mensalidade_rs.iloc[0] == pytest.approx(110)
    training["churn"] = [0, 0, 1, 1]
    training["data_ocorrencia_churn"] = "2099-01-01"
    training["id_cliente"] = "metadado"
    other = CustomerImputer(numeric_strategy="knn", n_neighbors=2).fit(training)
    pd.testing.assert_frame_equal(fitted.transform(incoming), other.transform(incoming))


def test_modes_and_medians_only_use_training_and_counts_respect_anchors():
    training = rows()
    fitted = CustomerImputer().fit(training)
    incoming = training.iloc[[0]].copy()
    incoming.loc[:, ["mensalidade_rs", "atrasos_6m", "tipo_contrato", "forma_pagamento", "chamados_30d"]] = np.nan
    incoming.loc[:, ["chamados_60d", "chamados_90d"]] = 0.
    before = fitted.numeric_imputer_.statistics_.copy()
    result = fitted.transform(incoming)
    assert result.mensalidade_rs.iloc[0] == 70
    assert result.atrasos_6m.iloc[0] == 2  # 1,5: empate arredondado para cima.
    assert result.tipo_contrato.iloc[0] == "Mensal"
    assert result.forma_pagamento.iloc[0] == "Pix"
    assert result.loc[0, list(SUPPORT_COLUMNS)].tolist() == [0., 0., 0.]
    np.testing.assert_array_equal(before, fitted.numeric_imputer_.statistics_)


def test_inconsistent_observed_support_is_rejected_not_repaired():
    incoming = rows()
    incoming.loc[0, "chamados_30d"] = 5
    with pytest.raises(ValueError, match="30 <= 60 <= 90"):
        CustomerImputer().fit(incoming)


@pytest.mark.parametrize("strategy", ["median", "knn"])
def test_imputation_can_be_used_inside_future_ml_pipelines(strategy):
    training = rows()
    incoming = training.iloc[[0]].copy()
    incoming.loc[:, ["mensalidade_rs", "forma_pagamento"]] = np.nan
    pipeline = build_ml_preprocessor(numeric_imputation=strategy, scaling="standard").fit(training)
    assert np.isfinite(pipeline.transform(incoming)).all()


def test_internal_validation_masks_only_known_values_and_support_in_blocks():
    # Dados reais somente como treino deste teste de contrato.
    training = prepare_data(SOURCE).main.iloc[:120].copy()
    training.loc[training.index[0], "mensalidade_rs"] = np.nan
    config = ImputationConfig(folds=2)
    metrics, masks, selection = evaluate_imputers(training, config)
    original = training.set_index("id_cliente")
    assert not selection["test_used_for_selection"]
    assert set(metrics.strategy) == {"median", "knn"}
    for record in masks.itertuples():
        assert pd.notna(original.at[record.id_cliente, record.field])
    support = masks.loc[masks.field.isin(SUPPORT_COLUMNS)]
    assert support.groupby(["fold", "id_cliente"]).size().eq(3).all()
    _, repeated_masks, repeated_selection = evaluate_imputers(training, config)
    pd.testing.assert_frame_equal(masks, repeated_masks)
    assert selection == repeated_selection


@pytest.mark.parametrize("settings", [{"folds": 1}, {"mask_fraction": 0},
                                      {"test_size": 1}, {"n_neighbors": 0}])
def test_invalid_experiment_configuration(settings):
    with pytest.raises(ValueError):
        ImputationConfig(**settings)


def test_real_export_is_complete_partitioned_audited_and_preserves_sources(tmp_path):
    protected = [SOURCE, ROOT / "analise_base_churn.ipynb", *sorted((ROOT / "data" / "processed").glob("*.*"))]
    before = {p: sha256(p.read_bytes()).hexdigest() for p in protected}
    summary = run_imputation(output_dir=tmp_path, config=ImputationConfig(folds=2))
    canonical = prepare_data(SOURCE).canonical.set_index("id_cliente")
    filled = pd.read_csv(tmp_path / "churn_imputed.csv").set_index("id_cliente")
    audit = pd.read_csv(tmp_path / "imputation_audit.csv")
    assert len(filled) == 4990 and filled.index.is_unique
    assert not filled.loc[:, list(FEATURE_COLUMNS)].isna().any().any()
    assert summary["counts"]["train"] == 3980
    assert summary["counts"]["test"] == 996
    assert summary["counts"]["new_customers"] == 14
    assert summary["counts"]["imputed_cells"] == len(audit) == 319
    for column in FEATURE_COLUMNS:
        observed = canonical[column].notna()
        values = filled.loc[canonical.index[observed], column]
        if column in NUMERIC_DOMAINS:
            np.testing.assert_allclose(values.astype(float), canonical.loc[observed, column].astype(float))
            low, high, integer = NUMERIC_DOMAINS[column]
            assert filled[column].between(low, high).all()
            if integer:
                assert filled[column].mod(1).eq(0).all()
        else:
            assert values.eq(canonical.loc[observed, column]).all()
    pd.testing.assert_series_equal(filled.churn.sort_index(), canonical.churn.astype("int64").sort_index(),
                                   check_dtype=False, check_index_type=False)
    assert filled.chamados_30d.le(filled.chamados_60d).all()
    assert filled.chamados_60d.le(filled.chamados_90d).all()
    assert not audit.observed_value_recovered.any()
    assert before == {p: sha256(p.read_bytes()).hexdigest() for p in protected}
    manifest = json.loads((tmp_path / "imputation_summary.json").read_text())
    assert manifest["selection"]["test_used_for_selection"] is False
    for name, digest in manifest["files"].items():
        assert sha256((tmp_path / name).read_bytes()).hexdigest() == digest

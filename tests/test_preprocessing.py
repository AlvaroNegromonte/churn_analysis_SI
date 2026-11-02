"""Contratos de negócio, auditoria e prevenção de vazamento; nenhum modelo é treinado."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from scripts.ml_preprocessing import build_ml_preprocessor, select_features, supervised_data
from scripts.preprocessing import (
    AuditLog, BINARY_FEATURES, CANONICAL_COLUMNS, CATEGORY_MAPS, FEATURE_COLUMNS,
    NUMERIC_FEATURES, PreprocessingConfig, RAW_COLUMNS, build_churn,
    clean_numeric_domains, deduplicate_records, load_data, normalize_categories,
    normalize_dates, parse_date, prepare_data, validate_result, write_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "base_churn_5000_VF2_aluno.xlsx - Dados_Sinteticos_VF2.csv"
SOURCE_HASH = "4f5d117bf1629be9c9c9d8d05617359708eeec9a0358f6c8925db6d02d08f8b5"
CONFIG = PreprocessingConfig()


def raw_row(**changes):
    values = dict(zip(RAW_COLUMNS, [
        "C1", "21/09/2024", "79,90", "", "1", "Mensal", "Pix", "1", "0", "",
    ]))
    values.update(changes)
    return values


def prepare_one(**changes):
    return prepare_data(pd.DataFrame([raw_row(**changes)]))


@pytest.fixture(scope="module")
def real_result():
    return prepare_data(SOURCE)


def test_real_input_and_population_preserved(real_result):
    assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
    assert len(real_result.audit_rows) == 5020
    assert len(real_result.canonical) == 4990
    assert len(real_result.main) == 4976
    assert len(real_result.new_customers) == 14
    assert real_result.main.churn.value_counts().to_dict() == {0: 3213, 1: 1763}
    assert real_result.new_customers.churn.value_counts().to_dict() == {1: 9, 0: 5}
    assert real_result.audit_rows.disposition.value_counts().to_dict() == {
        "principal": 4976, "duplicata_exata": 20, "novos": 14, "segregado": 10,
    }
    assert real_result.summary["source_sha256"] == SOURCE_HASH
    assert tuple(real_result.canonical.columns) == CANONICAL_COLUMNS


def test_real_quality_policies_match_eda(real_result):
    issues = real_result.audit_issues
    violations = issues.loc[issues.code.eq("numeric_domain_violation")]
    assert violations.field.value_counts().to_dict() == {"mensalidade_rs": 12, "atrasos_6m": 8}
    standardized = issues.loc[issues.code.eq("category_standardized")]
    assert standardized.field.value_counts().to_dict() == {"forma_pagamento": 90, "tipo_contrato": 80}
    assert not issues.code.eq("invalid_date").any()  # A lista do pedido não substitui a evidência.
    assert issues.code.eq("support_before_acquisition").sum() == 52
    assert issues.loc[issues.code.eq("support_before_acquisition"), "source_row"].nunique() == 33
    assert real_result.canonical.chamados_90d.isna().sum() == 33
    assert real_result.main.chamados_90d.isna().sum() == 23
    assert real_result.new_customers.chamados_90d.isna().sum() == 10
    assert real_result.canonical.mensalidade_rs.isna().sum() == 132
    assert real_result.canonical.atrasos_6m.isna().sum() == 43
    assert real_result.canonical.chamados_90d.median() == 1


def test_output_validation_detects_invalid_target_and_timeline():
    result = prepare_one(data_ocorrencia_churn="01/10/2026")
    corrupted = result.canonical.copy()
    corrupted.loc[0, "churn"] = 0
    with pytest.raises(ValueError, match="horizonte temporal"):
        validate_result(replace(result, canonical=corrupted), CONFIG)
    corrupted = result.canonical.copy()
    corrupted.loc[0, "data_aquisicao_cliente"] = pd.Timestamp("2027-01-01")
    with pytest.raises(ValueError, match="Cronologia"):
        validate_result(replace(result, canonical=corrupted), CONFIG)


def test_input_dataframe_not_mutated():
    raw = pd.DataFrame([raw_row(tipo_contrato="mensal", mensalidade_rs="9999")])
    before = raw.copy(deep=True)
    prepare_data(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_deterministic_steps_are_idempotent():
    frame = pd.DataFrame([raw_row(tipo_contrato=" mensal ", forma_pagamento="cartao", mensalidade_rs="-85")])
    audit = AuditLog(frame)
    categorical = normalize_categories(frame, audit)
    pd.testing.assert_frame_equal(categorical, normalize_categories(categorical, audit))
    numeric = clean_numeric_domains(frame, audit, ["mensalidade_rs", "atrasos_6m", *BINARY_FEATURES])
    pd.testing.assert_frame_equal(numeric, clean_numeric_domains(numeric, audit, ["mensalidade_rs", "atrasos_6m", *BINARY_FEATURES]))
    dated = normalize_dates(frame, audit)
    pd.testing.assert_frame_equal(dated, normalize_dates(dated, audit))
    duplicates = pd.concat([frame, frame], ignore_index=True)
    dedup = deduplicate_records(duplicates, AuditLog(duplicates))
    pd.testing.assert_frame_equal(dedup, deduplicate_records(dedup, AuditLog(dedup)))


def test_repeatable_outputs_and_audit_hashes(real_result, tmp_path):
    repeated = prepare_data(SOURCE)
    pd.testing.assert_frame_equal(real_result.canonical, repeated.canonical)
    pd.testing.assert_frame_equal(real_result.audit_issues, repeated.audit_issues)
    first = write_outputs(real_result, tmp_path / "first")
    second = write_outputs(repeated, tmp_path / "second")
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]
    manifest = json.loads((tmp_path / "first" / "audit_summary.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha256((tmp_path / "first" / name).read_bytes()).hexdigest() == digest


def test_refuse_output_over_original(tmp_path):
    source = tmp_path / "churn_canonical.csv"
    pd.DataFrame([raw_row()]).to_csv(source, index=False)
    before = source.read_bytes()
    result = prepare_data(source)
    with pytest.raises(ValueError, match="sobrescreveria"):
        write_outputs(result, tmp_path)
    assert source.read_bytes() == before


def test_duplicates_are_traced_and_conflicts_are_segregated():
    raw = pd.DataFrame([raw_row(), raw_row(), raw_row(id_cliente="C2"), raw_row(id_cliente="C2", mensalidade_rs="90")])
    result = prepare_data(raw)
    assert result.canonical.id_cliente.tolist() == ["C1"]
    assert json.loads(result.canonical.source_rows.iloc[0]) == [2, 3]
    assert result.audit_rows.disposition.tolist() == ["principal", "duplicata_exata", "segregado", "segregado"]
    assert result.audit_issues.code.eq("conflicting_identifier").sum() == 2


@pytest.mark.parametrize("contract,payment", [(" mensal ", "cartao"), ("ANUAL", " PIX "), ("bienal", "boleto")])
def test_explicit_category_mappings(contract, payment):
    result = prepare_one(tipo_contrato=contract, forma_pagamento=payment)
    assert result.canonical.tipo_contrato.iloc[0] in CATEGORY_MAPS["tipo_contrato"].values()
    assert result.canonical.forma_pagamento.iloc[0] in CATEGORY_MAPS["forma_pagamento"].values()


def test_unknown_categories_are_missing_and_audited():
    result = prepare_one(tipo_contrato="Trimestral", forma_pagamento="Dinheiro")
    assert result.canonical[["tipo_contrato", "forma_pagamento"]].isna().all().all()
    assert result.audit_issues.code.eq("unknown_category").sum() == 2


@pytest.mark.parametrize("field,value", [
    ("mensalidade_rs", "9.99"), ("mensalidade_rs", "500.01"), ("mensalidade_rs", "-85"),
    ("mensalidade_rs", np.inf), ("mensalidade_rs", "texto"),
    ("atrasos_6m", "99"), ("atrasos_6m", "-1"), ("atrasos_6m", "1.5"),
    ("fibra_optica", "2"), ("debito_automatico", "-1"),
])
def test_invalid_numeric_becomes_missing_without_removing_customer(field, value):
    result = prepare_one(**{field: value})
    assert len(result.canonical) == 1
    assert pd.isna(result.canonical[field].iloc[0])
    assert result.audit_issues.loc[result.audit_issues.field.eq(field), "code"].eq("numeric_domain_violation").any()


@pytest.mark.parametrize("fee,delays", [("10", "0"), ("500", "6")])
def test_numeric_domain_boundaries_are_inclusive(fee, delays):
    result = prepare_one(mensalidade_rs=fee, atrasos_6m=delays)
    assert result.canonical.mensalidade_rs.iloc[0] == float(fee)
    assert result.canonical.atrasos_6m.iloc[0] == int(delays)


@pytest.mark.parametrize("text,status", [("21/09/2024", "valid"), ("2024-09-21", "valid"),
                                         ("31/02/2024", "invalid"), ("09/21/2024", "invalid"),
                                         ("", "absent"), ("?", "unavailable")])
def test_controlled_date_formats(text, status):
    assert parse_date(text)[1] == status


@pytest.mark.parametrize("changes,reason", [
    ({"data_aquisicao_cliente": "31/02/2024"}, "unusable_acquisition_date"),
    ({"data_ocorrencia_churn": "data errada"}, "unusable_cancellation_date"),
    ({"data_ocorrencia_churn": "?"}, "unusable_cancellation_date"),
    ({"data_ocorrencia_churn": "20/09/2023"}, "cancellation_before_acquisition"),
    ({"data_aquisicao_cliente": "22/09/2026"}, "acquisition_after_cutoff"),
    ({"data_ocorrencia_churn": "21/09/2026"}, "cancelled_at_or_before_cutoff"),
    ({"id_cliente": ""}, "missing_identifier"),
    ({"data_aquisicao_cliente": "21/08/2016"}, "tenure_domain_violation"),
])
def test_unusable_rows_are_not_silently_removed(changes, reason):
    result = prepare_one(**changes)
    assert result.canonical.empty
    assert len(result.audit_rows) == 1
    assert reason in result.audit_issues.code.tolist()


@pytest.mark.parametrize("cancel,expected", [
    ("2026-09-20", None), ("2026-09-21", None), ("2026-09-22", 1),
    ("2026-11-20", 1), ("2026-11-21", 0), ("2026-12-01", 0), (None, 0),
])
def test_real_target_function_boundaries(cancel, expected):
    dates = pd.Series(pd.to_datetime([cancel]))
    result = build_churn(dates, pd.Series([True]), CONFIG).iloc[0]
    if expected is None:
        assert pd.isna(result)
    else:
        assert result == expected


def test_target_needs_eligibility_valid_date_and_future_observation():
    dates = pd.Series(pd.to_datetime([None, "2026-11-20", None]))
    short = replace(CONFIG, observation=pd.Timestamp("2026-10-21"))
    assert build_churn(dates, pd.Series([True, True, True]), short).isna().all()
    assert build_churn(dates, pd.Series([False, False, True]), CONFIG, pd.Series([False, False, True])).isna().all()
    no_coverage = replace(CONFIG, assume_full_followup=False)
    assert build_churn(pd.Series([pd.NaT]), pd.Series([True]), no_coverage).isna().all()
    assert build_churn(pd.Series(pd.to_datetime(["2026-10-01"])), pd.Series([True]), no_coverage).iloc[0] == 1


@pytest.mark.parametrize("days,cohort", [(59, "novos"), (60, "principal"), (61, "principal"), (0, "novos")])
def test_maturity_is_days_not_future_observation(days, cohort):
    acquisition = (CONFIG.cutoff - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    result = prepare_one(data_aquisicao_cliente=acquisition)
    assert result.canonical.coorte.iloc[0] == cohort
    assert result.canonical.churn.iloc[0] == 0


def test_calendar_months_are_not_days_divided_by_30():
    result = prepare_one(data_aquisicao_cliente="22/08/2026")
    assert result.canonical.tempo_cliente_dias.iloc[0] == 30
    assert result.canonical.tempo_cliente_meses.iloc[0] == 0


@pytest.mark.parametrize("window", [30, 60, 90])
def test_support_window_boundaries(window):
    dates = [CONFIG.cutoff - pd.Timedelta(days=window), CONFIG.cutoff, CONFIG.cutoff + pd.Timedelta(days=1)]
    history = ", ".join(value.strftime("%d/%m/%Y") for value in dates)
    result = prepare_one(data_chamado=history)
    assert result.canonical[f"chamados_{window}d"].iloc[0] == 1
    assert result.support_events[f"included_{window}d"].tolist() == [False, True, False]


@pytest.mark.parametrize("history,status", [("", "sem_chamados"), ("?", "informacao_indisponivel"),
                                           ("31/02/2026", "data_invalida"),
                                           ("01/01/2020, 20/09/2026", "cronologia_inconsistente")])
def test_absent_event_is_different_from_bad_history(history, status):
    result = prepare_one(data_chamado=history)
    assert result.canonical.status_suporte.iloc[0] == status
    if status == "sem_chamados":
        assert result.canonical.chamados_90d.iloc[0] == 0
    else:
        assert pd.isna(result.canonical.chamados_90d.iloc[0])


def test_known_incomplete_history_does_not_become_zero():
    result = prepare_one(historico_suporte_incompleto=True)
    assert result.canonical.status_suporte.iloc[0] == "historico_incompleto"
    assert result.canonical.chamados_90d.isna().all()


def test_unknown_completeness_does_not_become_zero():
    result = prepare_one(historico_suporte_incompleto="?")
    assert result.canonical.chamados_90d.isna().all()


def test_identifier_normalization_does_not_hide_conflicts():
    raw = pd.DataFrame([raw_row(id_cliente=" C1 "), raw_row(id_cliente="C1", mensalidade_rs="90")])
    result = prepare_data(raw)
    assert result.canonical.empty
    assert result.audit_issues.code.eq("identifier_stripped").sum() == 1
    assert result.audit_issues.code.eq("conflicting_identifier").sum() == 2


def test_support_above_domain_is_missing_not_clipped():
    result = prepare_one(data_chamado=", ".join(["20/09/2026"] * 51))
    assert result.canonical.chamados_90d.isna().all()
    assert len(result.support_events) == 51
    assert "Valor: 51" in result.audit_issues.loc[result.audit_issues.code.eq("numeric_domain_violation"), "detail"].iloc[0]


def test_future_events_and_labels_do_not_change_predictors():
    first = prepare_one(data_chamado="20/09/2026", data_ocorrencia_churn="01/10/2026")
    second = prepare_one(data_chamado="20/09/2026, 30/12/2026", data_ocorrencia_churn="01/12/2026")
    pd.testing.assert_frame_equal(select_features(first.canonical), select_features(second.canonical))
    assert first.canonical.churn.iloc[0] == 1 and second.canonical.churn.iloc[0] == 0


def test_new_raw_records_can_be_prepared_without_future_target():
    raw = pd.DataFrame([raw_row(id_cliente="NEW")]).drop(columns="data_ocorrencia_churn")
    result = prepare_data(raw, include_target=False)
    assert result.canonical.churn.isna().all()
    assert select_features(result.canonical).shape == (1, len(FEATURE_COLUMNS))
    with pytest.raises(ValueError, match="conhecido"):
        supervised_data(result.canonical)


@pytest.mark.parametrize("scaling", ["none", "standard", "minmax"])
def test_ml_pipeline_is_unfitted_and_learns_only_from_training(scaling):
    train = pd.DataFrame([raw_row(id_cliente="A", mensalidade_rs="20", tipo_contrato="Mensal"),
                          raw_row(id_cliente="B", mensalidade_rs="40", tipo_contrato="Anual"),
                          raw_row(id_cliente="C", mensalidade_rs="", tipo_contrato="Mensal")])
    training = prepare_data(train).canonical
    testing = prepare_one(id_cliente="TEST", mensalidade_rs="", tipo_contrato="Bienal").canonical
    preprocessor = build_ml_preprocessor(scaling=scaling)
    with pytest.raises(NotFittedError):
        preprocessor.transform(testing)
    fitted = clone(preprocessor).fit(training)
    columns = fitted.named_steps["columns"]
    imputer = columns.named_transformers_["numeric"].named_steps["imputer"]
    fee_position = list(NUMERIC_FEATURES).index("mensalidade_rs")
    assert imputer.statistics_[fee_position] == 30
    encoded = columns.named_transformers_["categorical"].named_steps["encoder"]
    assert "Bienal" not in encoded.categories_[0]
    before = imputer.statistics_.copy()
    transformed = fitted.transform(testing)
    assert np.isfinite(transformed).all()
    assert transformed.shape[1] == fitted.transform(training).shape[1]
    np.testing.assert_array_equal(imputer.statistics_, before)
    if scaling != "none":
        scaler = columns.named_transformers_["numeric"].named_steps["scaler"]
        if scaling == "standard":
            assert scaler.mean_[fee_position] == 30
        else:
            assert scaler.data_max_[fee_position] == 40


def test_unseen_test_values_and_target_never_enter_fit_statistics(real_result):
    training = real_result.main.iloc[:20].copy()
    a = build_ml_preprocessor().fit(training)
    changed = training.copy()
    changed["churn"] = 1 - changed["churn"]
    changed["id_cliente"] = "não é preditor"
    changed["data_ocorrencia_churn"] = "2099-01-01"
    b = build_ml_preprocessor().fit(changed)
    np.testing.assert_allclose(a.transform(training), b.transform(changed))
    assert set(select_features(real_result.canonical)) == set(FEATURE_COLUMNS)
    assert not {"churn", "id_cliente", "data_ocorrencia_churn", "coorte", "source_rows"}.intersection(FEATURE_COLUMNS)


def test_all_missing_training_feature_requires_explicit_decision():
    training = prepare_one(mensalidade_rs="").canonical
    with pytest.raises(ValueError, match="Treino sem valores"):
        build_ml_preprocessor().fit(training)


def test_undefined_category_mode_is_not_invented():
    training = prepare_one(tipo_contrato="").canonical
    with pytest.raises(ValueError, match="Treino sem valores"):
        build_ml_preprocessor(categorical_imputation="most_frequent").fit(training)
    # Marcador constante é uma política distinta, explícita e válida para categoria ausente.
    assert np.isfinite(build_ml_preprocessor().fit_transform(training)).all()


def test_category_mode_is_fitted_only_on_training():
    raw = pd.DataFrame([raw_row(id_cliente="A", tipo_contrato="Mensal"),
                        raw_row(id_cliente="B", tipo_contrato="Mensal"),
                        raw_row(id_cliente="C", tipo_contrato="Anual")])
    training = prepare_data(raw).canonical
    fitted = build_ml_preprocessor(categorical_imputation="most_frequent").fit(training)
    imputer = fitted.named_steps["columns"].named_transformers_["categorical"].named_steps["imputer"]
    assert imputer.statistics_[0] == "Mensal"
    incoming = select_features(training.iloc[:1])
    incoming["tipo_contrato"] = np.nan
    fitted.transform(incoming)
    assert imputer.statistics_[0] == "Mensal"


def test_ml_handles_missing_new_record_and_unknown_category(real_result):
    preprocessor = build_ml_preprocessor().fit(real_result.main.iloc[:100])
    incoming = select_features(real_result.main.iloc[:1])
    incoming.loc[:, list(NUMERIC_FEATURES)] = np.nan
    incoming.loc[:, "tipo_contrato"] = "Categoria nunca vista"
    incoming.loc[:, "forma_pagamento"] = np.nan
    assert np.isfinite(preprocessor.transform(incoming)).all()


def test_schema_validation_and_raw_contract():
    with pytest.raises(ValueError, match="Esquema"):
        prepare_data(pd.DataFrame({"id_cliente": ["C1"]}))
    with pytest.raises(ValueError, match="Esquema"):
        prepare_data(prepare_one().canonical)
    with pytest.raises(ValueError, match="Preditores"):
        select_features(pd.DataFrame({"churn": [1]}))


def test_multiple_or_different_cutoffs_are_not_silently_ignored():
    raw = pd.DataFrame([raw_row(data_corte="2026-09-20")])
    with pytest.raises(ValueError, match="outro corte"):
        prepare_data(raw)


def test_future_target_cannot_affect_raw_deduplication():
    raw = pd.DataFrame([raw_row(churn=0), raw_row(churn=1)])
    with pytest.raises(ValueError, match="não esperadas"):
        prepare_data(raw)


@pytest.mark.parametrize("config", [{"horizon_days": 0}, {"horizon_days": 60.5},
                                   {"maturity_days": -1}, {"observation": "2026-09-20"},
                                   {"cutoff": "2026-09-21 12:00"}])
def test_invalid_temporal_configuration(config):
    with pytest.raises(ValueError):
        PreprocessingConfig(**config)


def test_cli_runs_from_other_directory_without_notebook(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame([raw_row()]).to_csv(raw, index=False)
    output = tmp_path / "processed"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_preprocessing.py"), "--input", str(raw), "--output-dir", str(output)],
        cwd=tmp_path, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert len(pd.read_csv(output / "churn_main.csv")) == 1
    assert not list(tmp_path.glob("*.ipynb"))


def test_invalid_ml_options_are_rejected():
    with pytest.raises(ValueError):
        build_ml_preprocessor(scaling="arbitrary")
    with pytest.raises(ValueError):
        build_ml_preprocessor(categorical_imputation="arbitrary")

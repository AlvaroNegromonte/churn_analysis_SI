"""Nível A: preparação determinística, sem ajuste estatístico ou dependência do notebook."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


RAW_COLUMNS = (
    "id_cliente", "data_aquisicao_cliente", "mensalidade_rs", "data_chamado",
    "atrasos_6m", "tipo_contrato", "forma_pagamento", "fibra_optica",
    "debito_automatico", "data_ocorrencia_churn",
)
NUMERIC_FEATURES = (
    "tempo_cliente_meses", "mensalidade_rs", "atrasos_6m",
    "chamados_30d", "chamados_60d", "chamados_90d",
)
CATEGORICAL_FEATURES = ("tipo_contrato", "forma_pagamento")
BINARY_FEATURES = ("fibra_optica", "debito_automatico")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
WINDOWS = (30, 60, 90)
MISSING_MARKERS = frozenset({
    "na", "n/a", "n.a.", "nan", "null", "none", "missing", "-", "--", "?",
    "sem informacao", "nao informado", "nao disponivel",
})
CATEGORY_MAPS = {
    "tipo_contrato": {"mensal": "Mensal", "anual": "Anual", "bienal": "Bienal"},
    "forma_pagamento": {"boleto": "Boleto", "cartao": "Cartão", "pix": "Pix"},
}
NUMERIC_DOMAINS = {
    "mensalidade_rs": (10, 500, False), "atrasos_6m": (0, 6, True),
    "fibra_optica": (0, 1, True), "debito_automatico": (0, 1, True),
    "tempo_cliente_meses": (0, 120, True),
    **{f"chamados_{days}d": (0, 50, True) for days in WINDOWS},
}
ISSUE_COLUMNS = (
    "source_row", "id_cliente", "stage", "field", "code", "action", "before", "after", "detail",
)
EVENT_COLUMNS = (
    "source_row", "id_cliente", "event_number", "original_date", "parsed_date",
    "status", "included_30d", "included_60d", "included_90d",
)
META_COLUMNS = (
    "id_cliente", "data_corte", "data_aquisicao_cliente", "data_ocorrencia_churn",
    "tempo_cliente_dias", "coorte", "churn", "status_suporte",
    "exposicao_completa_30d", "exposicao_completa_60d", "exposicao_completa_90d",
    "source_rows",
)
CANONICAL_COLUMNS = META_COLUMNS + FEATURE_COLUMNS


@dataclass(frozen=True)
class PreprocessingConfig:
    cutoff: pd.Timestamp = pd.Timestamp("2026-09-21")
    observation: pd.Timestamp = pd.Timestamp("2026-11-21")
    horizon_days: int = 60
    maturity_days: int = 60
    assume_full_followup: bool = True

    def __post_init__(self):
        for name in ("cutoff", "observation"):
            value = pd.Timestamp(getattr(self, name))
            if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
                raise ValueError(f"{name}: informe uma data válida, sem horário ou fuso.")
            object.__setattr__(self, name, value)
        if (not isinstance(self.horizon_days, int) or not isinstance(self.maturity_days, int)
                or self.horizon_days <= 0 or self.maturity_days < 0 or self.observation < self.cutoff):
            raise ValueError("Configuração temporal inválida.")

    @property
    def horizon_end(self):
        return self.cutoff + pd.Timedelta(days=self.horizon_days)

    def as_dict(self):
        return {
            "cutoff": self.cutoff.strftime("%Y-%m-%d"),
            "observation": self.observation.strftime("%Y-%m-%d"),
            "horizon_end": self.horizon_end.strftime("%Y-%m-%d"),
            "horizon_days": self.horizon_days, "maturity_days": self.maturity_days,
            "assume_full_followup": self.assume_full_followup,
        }


@dataclass
class PreparationResult:
    canonical: pd.DataFrame
    main: pd.DataFrame
    new_customers: pd.DataFrame
    audit_rows: pd.DataFrame
    audit_issues: pd.DataFrame
    support_events: pd.DataFrame
    summary: dict
    source_path: Path | None = None


def normalize_text(value):
    """Chave de comparação; não inventa categorias nem remove significado do identificador."""
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def is_missing(value):
    return pd.isna(value) or normalize_text(value) in MISSING_MARKERS or str(value).strip() == ""


def _json_value(value):
    if isinstance(value, (pd.Timestamp, datetime, date)):
        value = value.isoformat()
    elif pd.isna(value):
        value = None
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        value = str(value)
    elif isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class AuditLog:
    """Problemas e mudanças por linha original; a fonte bruta fica no ledger separado."""

    def __init__(self, raw):
        self.raw = raw
        self.issues = []
        self.disposition = pd.Series("pendente", index=raw.index, dtype="string")

    def add(self, index, stage, field, code, action, after=None, detail=""):
        self.issues.append({
            "source_row": int(index) + 2,
            "id_cliente": self.raw.at[index, "id_cliente"],
            "stage": stage, "field": field, "code": code, "action": action,
            "before": _json_value(self.raw.at[index, field]) if field in self.raw else "null",
            "after": _json_value(after), "detail": detail,
        })

    def mask(self, mask, stage, field, code, action, detail=""):
        for index in mask.index[mask.fillna(False)]:
            self.add(index, stage, field, code, action, detail=detail)

    def exclude(self, frame, mask, code, *, disposition="segregado", field=""):
        self.mask(mask, "eligibility", field, code, disposition)
        self.disposition.loc[mask.index[mask.fillna(False)]] = disposition
        return frame.loc[~mask.fillna(False)].copy()


def load_data(source, *, include_target=True):
    """Lê CSV bruto preservando vazio literal, necessário à semântica dos eventos."""
    if isinstance(source, pd.DataFrame):
        frame = source.copy(deep=True)
    else:
        frame = pd.read_csv(source, sep=",", encoding="utf-8", dtype=str, keep_default_na=False)
    required = set(RAW_COLUMNS)
    if not include_target:
        required.remove("data_ocorrencia_churn")
    missing = required.difference(frame.columns)
    allowed = set(RAW_COLUMNS) | {"historico_suporte_incompleto", "data_corte"}
    unexpected = set(frame.columns).difference(allowed)
    if missing or unexpected or not frame.columns.is_unique:
        raise ValueError(f"Esquema bruto inválido. Colunas ausentes: {sorted(missing)}; não esperadas: {sorted(unexpected)}")
    if "data_ocorrencia_churn" not in frame:
        frame["data_ocorrencia_churn"] = ""
    return frame.reset_index(drop=True)


def deduplicate_records(raw, audit):
    """Deduplica apenas linhas idênticas, antes de normalizar valores ou identificadores."""
    duplicated = raw.duplicated(keep="first")
    audit.mask(duplicated, "deduplication", "", "exact_duplicate", "duplicata_exata")
    audit.disposition.loc[duplicated] = "duplicata_exata"
    frame = raw.loc[~duplicated].copy()
    ids = frame["id_cliente"].map(lambda value: pd.NA if is_missing(value) else str(value).strip())
    invalid_id = ids.isna()
    frame = audit.exclude(frame, invalid_id, "missing_identifier", field="id_cliente")
    for index in frame.index:
        if str(frame.at[index, "id_cliente"]) != ids.loc[index]:
            audit.add(index, "identifiers", "id_cliente", "identifier_stripped", "remover_espacos_externos", ids.loc[index])
    frame["id_cliente"] = ids.loc[frame.index].astype("string")
    conflicts = frame["id_cliente"].duplicated(keep=False)
    return audit.exclude(frame, conflicts, "conflicting_identifier", field="id_cliente")


def parse_date(value):
    """Retorna (data, estado): vazio de evento é distinto de informação indisponível."""
    if isinstance(value, str) and value == "":
        return pd.NaT, "absent"
    if is_missing(value):
        return pd.NaT, "unavailable"
    if isinstance(value, (pd.Timestamp, datetime, date)):
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None and parsed == parsed.normalize():
            return parsed, "valid"
        return pd.NaT, "invalid"
    text = str(value).strip()
    for pattern, fmt in ((r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y"), (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d")):
        if re.fullmatch(pattern, text):
            parsed = pd.to_datetime(text, format=fmt, errors="coerce")
            if pd.notna(parsed):
                return parsed, "valid"
    return pd.NaT, "invalid"


def normalize_dates(frame, audit):
    result = frame.copy()
    for column in ("data_aquisicao_cliente", "data_ocorrencia_churn"):
        parsed = frame[column].map(parse_date)
        result[column] = pd.to_datetime(parsed.map(lambda item: item[0]))
        states = parsed.map(lambda item: item[1])
        status_column = f"{column}_status"
        # Reaplicar a normalização não pode transformar data inválida em evento ausente.
        if status_column in frame:
            previous = frame[status_column].where(frame[status_column].isin(["valid", "absent", "invalid", "unavailable"]))
            states = states.where(frame[column].notna(), previous.fillna(states))
        result[status_column] = states
        for state in ("invalid", "unavailable"):
            audit.mask(result[f"{column}_status"].eq(state), "dates", column,
                       f"{state}_date", "segregar_se_essencial")
    return result


def normalize_categories(frame, audit):
    result = frame.copy()
    for column, mapping in CATEGORY_MAPS.items():
        values = []
        for index, value in frame[column].items():
            if is_missing(value):
                normalized = pd.NA
                audit.add(index, "categories", column, "missing_category", "preservar_ausencia")
            else:
                normalized = mapping.get(normalize_text(value), pd.NA)
                if pd.isna(normalized):
                    audit.add(index, "categories", column, "unknown_category", "converter_para_ausencia")
                elif str(value) != normalized:
                    audit.add(index, "categories", column, "category_standardized", "padronizar", normalized)
            values.append(normalized)
        result[column] = pd.Series(values, index=frame.index, dtype="string")
    return result


def clean_numeric_domains(frame, audit, columns):
    """Violações viram ausência; não há medianas, modas, arredondamento ou winsorização."""
    result = frame.copy()
    for column in columns:
        minimum, maximum, integer = NUMERIC_DOMAINS[column]
        original = frame[column]
        missing = original.map(is_missing)
        numeric = pd.to_numeric(original.astype("string").str.replace(",", ".", regex=False), errors="coerce")
        invalid = ~missing & (numeric.isna() | ~numeric.between(minimum, maximum))
        if integer:
            invalid |= ~missing & numeric.notna() & numeric.mod(1).ne(0)
        audit.mask(missing, "domains", column, "missing_numeric", "preservar_ausencia")
        for index in invalid.index[invalid.fillna(False)]:
            audit.add(index, "domains", column, "numeric_domain_violation", "converter_para_ausencia",
                      detail=f"Valor: {original.loc[index]}; domínio: {minimum}..{maximum}; inteiro={integer}")
        result[column] = numeric.mask(missing | invalid).astype("Int64" if integer else "Float64")
    return result


def select_eligible(frame, audit, config):
    """Datas essenciais desconhecidas são segregadas; clientes fora do risco são documentados."""
    result = frame.copy()
    rules = [
        (result["data_aquisicao_cliente"].isna(), "unusable_acquisition_date", "segregado", "data_aquisicao_cliente"),
        (result["data_ocorrencia_churn_status"].isin(["invalid", "unavailable"]), "unusable_cancellation_date", "segregado", "data_ocorrencia_churn"),
        (result["data_ocorrencia_churn"].lt(result["data_aquisicao_cliente"]), "cancellation_before_acquisition", "segregado", "data_ocorrencia_churn"),
        (result["data_aquisicao_cliente"].gt(config.cutoff), "acquisition_after_cutoff", "inelegivel", "data_aquisicao_cliente"),
        (result["data_ocorrencia_churn"].le(config.cutoff), "cancelled_at_or_before_cutoff", "inelegivel", "data_ocorrencia_churn"),
    ]
    for mask, code, disposition, field in rules:
        result = audit.exclude(result, mask.reindex(result.index), code, disposition=disposition, field=field)
    return result


def derive_tenure(frame, config):
    result = frame.copy()
    acquisition = result["data_aquisicao_cliente"]
    result["data_corte"] = config.cutoff
    result["tempo_cliente_dias"] = (config.cutoff - acquisition).dt.days.astype("Int64")
    result["tempo_cliente_meses"] = (
        (config.cutoff.year - acquisition.dt.year) * 12 + config.cutoff.month - acquisition.dt.month
        - acquisition.dt.day.gt(config.cutoff.day).astype(int)
    ).astype("Int64")
    mature = result["tempo_cliente_dias"].ge(config.maturity_days)
    result["coorte"] = pd.Series(np.where(mature, "principal", "novos"), index=result.index, dtype="string")
    return result


def _support_history(index, row, config, audit):
    value = row["data_chamado"]
    if isinstance(value, str) and value == "":
        audit.add(index, "support", "data_chamado", "empty_support", "zero_eventos_registrados", 0)
        return [], "sem_chamados"
    if is_missing(value):
        audit.add(index, "support", "data_chamado", "unavailable_support", "contagem_desconhecida")
        return [], "informacao_indisponivel"
    events = []
    status = "registrado"
    for number, text in enumerate(str(value).split(","), start=1):
        parsed, state = parse_date(text.strip())
        event_status = state
        if state != "valid":
            status = "data_invalida"
            audit.add(index, "support", "data_chamado", "invalid_support_date", "contagem_desconhecida", detail=text)
        elif parsed < row["data_aquisicao_cliente"]:
            event_status = "before_acquisition"
            status = "cronologia_inconsistente"
            audit.add(index, "support", "data_chamado", "support_before_acquisition", "contagem_desconhecida", detail=text)
        elif parsed > config.cutoff:
            event_status = "after_cutoff"
            audit.add(index, "support", "data_chamado", "support_after_cutoff", "ignorar_no_preditor", detail=text)
        event = {"source_row": index + 2, "id_cliente": row["id_cliente"], "event_number": number,
                 "original_date": text.strip(), "parsed_date": parsed, "status": event_status}
        for days in WINDOWS:
            event[f"included_{days}d"] = bool(state == "valid" and parsed >= row["data_aquisicao_cliente"]
                                              and config.cutoff - pd.Timedelta(days=days) < parsed <= config.cutoff)
        events.append(event)
    return events, status


def derive_support_features(frame, audit, config):
    result = frame.copy()
    all_events = []
    statuses = {}
    counts = {days: {} for days in WINDOWS}
    for index, row in frame.iterrows():
        events, status = _support_history(index, row, config, audit)
        if "historico_suporte_incompleto" in row:
            flag = normalize_text(row["historico_suporte_incompleto"])
            if flag in {"true", "1", "sim"}:
                status = "historico_incompleto"
                audit.add(index, "support", "data_chamado", "incomplete_support_history", "contagem_desconhecida")
            elif flag not in {"false", "0", "nao"}:
                status = "informacao_indisponivel"
                audit.add(index, "support", "data_chamado", "unknown_history_completeness", "contagem_desconhecida")
        usable = status in {"sem_chamados", "registrado"}
        for days in WINDOWS:
            counts[days][index] = sum(event[f"included_{days}d"] for event in events) if usable else pd.NA
        statuses[index] = status
        all_events.extend(events)
    result["status_suporte"] = pd.Series(statuses, dtype="string")
    for days in WINDOWS:
        result[f"chamados_{days}d"] = pd.Series(counts[days], dtype="Int64")
        result[f"exposicao_completa_{days}d"] = result["tempo_cliente_dias"].ge(days)
    return result, pd.DataFrame(all_events, columns=EVENT_COLUMNS)


def build_churn(cancellations, eligible, config, invalid_dates=None):
    """Alvo (corte, fim]; data no corte implica inelegibilidade, não classe zero."""
    if invalid_dates is None:
        invalid_dates = pd.Series(False, index=cancellations.index)
    usable = eligible.fillna(False) & ~invalid_dates.fillna(True) & (cancellations.isna() | cancellations.gt(config.cutoff))
    target = pd.Series(pd.NA, index=cancellations.index, dtype="Int64")
    positive = usable & cancellations.gt(config.cutoff) & cancellations.le(config.horizon_end) & cancellations.le(config.observation)
    observed_negative = usable & cancellations.gt(config.horizon_end) & cancellations.le(config.observation)
    target.loc[positive] = 1
    target.loc[observed_negative] = 0
    if config.assume_full_followup and config.observation >= config.horizon_end:
        target.loc[usable & (cancellations.isna() | cancellations.gt(config.horizon_end))] = 0
    return target


def _audit_tables(raw, audit):
    issues = pd.DataFrame(audit.issues, columns=ISSUE_COLUMNS)
    if len(issues):
        issues = issues.sort_values(["source_row", "stage", "field", "code"], kind="stable").reset_index(drop=True)
    rows = raw.copy()
    rows.insert(0, "source_row", np.arange(len(raw)) + 2)
    rows["disposition"] = audit.disposition
    codes = issues.groupby("source_row")["code"].agg(lambda s: json.dumps(sorted(set(s)), ensure_ascii=False))
    rows["audit_codes"] = rows["source_row"].map(codes).fillna("[]")
    return rows, issues


def validate_result(result, config, *, include_target=True):
    canonical = result.canonical
    if tuple(canonical.columns) != CANONICAL_COLUMNS or canonical.duplicated(["id_cliente", "data_corte"]).any():
        raise ValueError("Esquema canônico ou granularidade inválidos.")
    for column in FEATURE_COLUMNS:
        values = canonical[column].dropna()
        if column in NUMERIC_DOMAINS:
            low, high, integer = NUMERIC_DOMAINS[column]
            if not values.between(low, high).all() or (integer and values.mod(1).ne(0).any()):
                raise ValueError(f"Violação introduzida em {column}.")
        elif not values.isin(CATEGORY_MAPS[column].values()).all():
            raise ValueError(f"Categoria não canônica em {column}.")
    if not canonical["churn"].dropna().isin([0, 1]).all():
        raise ValueError("Rótulo fora de {0,1}.")
    acquisition = canonical["data_aquisicao_cliente"]
    cancellation = canonical["data_ocorrencia_churn"]
    if (acquisition.isna().any() or acquisition.gt(config.cutoff).any()
            or not canonical["data_corte"].eq(config.cutoff).all()
            or cancellation.le(config.cutoff).any() or cancellation.lt(acquisition).any()):
        raise ValueError("Cronologia incompatível na base canônica.")
    in_window = cancellation.gt(config.cutoff) & cancellation.le(config.horizon_end) & cancellation.le(config.observation)
    if ((canonical["churn"].eq(1).fillna(False) & ~in_window).any()
            or (canonical["churn"].eq(0).fillna(False) & in_window).any()):
        raise ValueError("Rótulo incompatível com o horizonte temporal.")
    if include_target and config.assume_full_followup and config.observation >= config.horizon_end and canonical["churn"].isna().any():
        raise ValueError("Desfecho desconhecido apesar da premissa de cobertura completa.")
    if len(result.main) + len(result.new_customers) != len(canonical):
        raise ValueError("Partição de coortes inconsistente.")
    if not result.main["tempo_cliente_dias"].ge(config.maturity_days).all():
        raise ValueError("Cliente novo na coorte principal.")
    if not result.new_customers["tempo_cliente_dias"].lt(config.maturity_days).all():
        raise ValueError("Cliente maturado na coorte de novos.")
    if result.audit_rows["disposition"].eq("pendente").any():
        raise ValueError("Há linhas sem destino auditado.")


def prepare_data(source, config=None, *, include_target=True):
    """Coordena o Nível A a partir do esquema bruto. Não escreve arquivos nem faz fit."""
    config = config or PreprocessingConfig()
    raw = load_data(source, include_target=include_target)
    if "data_corte" in raw:
        supplied_cuts = raw["data_corte"].map(parse_date)
        if not supplied_cuts.map(lambda item: item[1] == "valid" and item[0] == config.cutoff).all():
            raise ValueError("Entrada contém outro corte ou vários cortes: prepare uma fotografia por execução.")
    raw_fingerprint = sha256(raw.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    audit = AuditLog(raw)
    frame = deduplicate_records(raw, audit)
    frame = normalize_dates(frame, audit)
    frame = normalize_categories(frame, audit)
    frame = clean_numeric_domains(frame, audit, ["mensalidade_rs", "atrasos_6m", *BINARY_FEATURES])
    frame = select_eligible(frame, audit, config)
    frame = derive_tenure(frame, config)
    invalid_tenure = ~frame["tempo_cliente_meses"].between(0, 120)
    frame = audit.exclude(frame, invalid_tenure, "tenure_domain_violation", field="data_aquisicao_cliente")
    frame, events = derive_support_features(frame, audit, config)
    frame = clean_numeric_domains(frame, audit, [f"chamados_{days}d" for days in WINDOWS])
    if include_target:
        frame["churn"] = build_churn(frame["data_ocorrencia_churn"], pd.Series(True, index=frame.index), config)
    else:
        frame["churn"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    raw_ids = raw["id_cliente"].map(lambda value: "" if is_missing(value) else str(value).strip())
    lineage = (raw.index.to_series() + 2).groupby(raw_ids).agg(lambda s: json.dumps(s.tolist()))
    frame["source_rows"] = frame["id_cliente"].map(lineage)
    audit.disposition.loc[frame.index] = frame["coorte"]
    canonical = frame.loc[:, list(CANONICAL_COLUMNS)].sort_values(["id_cliente", "data_corte"], kind="stable").reset_index(drop=True)
    main = canonical.loc[canonical["coorte"].eq("principal")].reset_index(drop=True)
    new_customers = canonical.loc[canonical["coorte"].eq("novos")].reset_index(drop=True)
    rows, issues = _audit_tables(raw, audit)
    summary = {
        "schema_version": 1, "config": config.as_dict(), "include_target": include_target,
        "source_table_sha256": raw_fingerprint,
        "source_sha256": None if isinstance(source, pd.DataFrame) else sha256(Path(source).read_bytes()).hexdigest(),
        "counts": {"input_rows": len(raw), "canonical": len(canonical), "main": len(main), "new_customers": len(new_customers),
                   **{str(k): int(v) for k, v in rows["disposition"].value_counts().sort_index().items()}},
        "issues": {str(k): {"occurrences": int(len(g)), "affected_rows": int(g["source_row"].nunique())}
                   for k, g in issues.groupby("code", sort=True)},
        "missing_canonical": {column: int(canonical[column].isna().sum()) for column in FEATURE_COLUMNS},
        "target_counts": {str(k): int(v) for k, v in canonical["churn"].value_counts().sort_index().items()},
        "provenance": feature_provenance(),
        "assumptions": ["Cobertura até a observação é premissa didática, não prova independente no CSV.",
                        "Mensalidade, contrato, pagamento, atrasos e indicadores representam o corte por premissa.",
                        "Vazio literal de suporte é zero eventos registrados; informação comprometida permanece ausente."],
    }
    source_path = None if isinstance(source, pd.DataFrame) else Path(source).resolve()
    result = PreparationResult(canonical, main, new_customers, rows, issues, events, summary, source_path)
    validate_result(result, config, include_target=include_target)
    return result


def feature_provenance():
    return {
        "tempo_cliente_meses": "Meses completos de calendário entre aquisição válida e corte; não divide dias por 30.",
        **{f"chamados_{days}d": f"Contagem em (corte-{days} dias, corte], com data >= aquisição; vazio literal=0, histórico comprometido=NA."
           for days in WINDOWS},
        **{column: f"{column} na fotografia do corte, com domínio validado; violação=NA."
           for column in ("mensalidade_rs", "atrasos_6m", *BINARY_FEATURES)},
        **{column: f"Mapeamento explícito de {column}; desconhecido=NA, sem ajuste estatístico."
           for column in CATEGORICAL_FEATURES},
        "churn": "Cancelamento em (corte, fim_horizonte] conforme config.horizon_days; zero somente com desfecho suficientemente observado; não é preditor.",
        "trace": "source_rows remete às linhas originais de audit_rows.csv; eventos e problemas têm source_row, começando em 2.",
    }


def write_outputs(result, output_dir, *, source_path=None):
    """Exporta apenas resultados validados. Nomes fixos são substituídos em reexecuções."""
    output_dir = Path(output_dir)
    frames = {
        "churn_canonical.csv": result.canonical,
        "churn_main.csv": result.main,
        "churn_new_customers.csv": result.new_customers,
        "audit_rows.csv": result.audit_rows,
        "audit_issues.csv": result.audit_issues,
        "audit_support_events.csv": result.support_events,
    }
    destinations = [output_dir / name for name in (*frames, "audit_summary.json")]
    protected = [Path(p).resolve() for p in (result.source_path, source_path) if p is not None]
    if set(protected).intersection(p.resolve() for p in destinations):
        raise ValueError("Uma saída sobrescreveria a base de entrada.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8", lineterminator="\n", date_format="%Y-%m-%d", na_rep="")
    manifest = dict(result.summary)
    manifest["runtime"] = {name: version(name) for name in ("pandas", "numpy")}
    manifest["implementation_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["files"] = {name: sha256((output_dir / name).read_bytes()).hexdigest() for name in frames}
    (output_dir / "audit_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return destinations

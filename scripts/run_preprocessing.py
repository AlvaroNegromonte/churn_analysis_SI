"""Executa apenas a preparação determinística e exporta suas saídas auditáveis."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preprocessing import PreprocessingConfig, prepare_data, write_outputs


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "base_churn_5000_VF2_aluno.xlsx - Dados_Sinteticos_VF2.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--cutoff", default="2026-09-21")
    parser.add_argument("--observation", default="2026-11-21")
    parser.add_argument("--no-full-followup", action="store_true", help="Não assumir acompanhamento completo dos campos de churn vazios.")
    args = parser.parse_args(argv)
    before = sha256(args.input.read_bytes()).hexdigest()
    config = PreprocessingConfig(cutoff=args.cutoff, observation=args.observation,
                                 assume_full_followup=not args.no_full_followup)
    result = prepare_data(args.input, config)
    paths = write_outputs(result, args.output_dir, source_path=args.input)
    if sha256(args.input.read_bytes()).hexdigest() != before:
        raise RuntimeError("A base original foi modificada.")
    counts = result.summary["counts"]
    print(f"Entrada: {counts['input_rows']}; canônica: {counts['canonical']}; principal: {counts['main']}; novos: {counts['new_customers']}")
    print(f"Auditoria: {len(result.audit_issues)} ocorrências; {len(result.audit_rows)} linhas de entrada rastreadas.")
    print(f"Saídas geradas: {len(paths)}; SHA-256 da fonte preservado: {before}")
    print("Nenhum pré-processador estatístico foi ajustado; nenhum modelo foi treinado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

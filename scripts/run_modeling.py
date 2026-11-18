"""Exporta a base transformada do miniprojeto 2 sem precisar abrir o notebook.

Uso (na raiz do projeto):
    .venv/Scripts/python -m scripts.run_modeling          # Windows
    .venv/bin/python -m scripts.run_modeling              # Linux/macOS

Split 70/30 estratificado (semente 42) sobre a coorte principal; o pré-processamento
(KNN nos numéricos, moda nos categóricos, one-hot) é ajustado SOMENTE no treino.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .ml_preprocessing import build_ml_preprocessor
from .modeling import MAIN_PATH, MODELING_DIR, export_transformed, load_main, split_70_30


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=MAIN_PATH)
    parser.add_argument("--output", type=Path, default=MODELING_DIR / "churn_transformed_70_30.csv")
    parser.add_argument("--imputation", choices=["knn", "median"], default="knn")
    args = parser.parse_args(argv)

    X_train, X_test, y_train, y_test, ids = split_70_30(load_main(args.input))
    prep = build_ml_preprocessor(numeric_imputation=args.imputation).set_output(transform="pandas")
    result = export_transformed(prep, X_train, X_test, y_train, y_test, ids, args.output)
    counts = result.groupby("partition")["churn"].agg(["size", "mean"])
    print(f"Base transformada salva em {args.output}")
    print(counts.to_string())


if __name__ == "__main__":
    main()

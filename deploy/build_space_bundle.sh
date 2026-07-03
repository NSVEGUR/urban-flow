#!/usr/bin/env bash
# Assembles the minimal file set the Streamlit demo needs into a standalone
# bundle for Hugging Face Spaces (which expects requirements.txt + README.md
# frontmatter at the repo root, and doesn't need our heavier baseline/SOTA
# dependencies — pmdarima, pytorch-forecasting, statsmodels, etc.).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$(mktemp -d)}"

echo "Building Space bundle in $OUT_DIR"

mkdir -p "$OUT_DIR/app/uncertainty" "$OUT_DIR/app/demo" "$OUT_DIR/data"

cp "$ROOT_DIR/app/__init__.py" "$OUT_DIR/app/"
cp "$ROOT_DIR/app/config.py" "$OUT_DIR/app/"
cp "$ROOT_DIR/app/data_pipeline.py" "$OUT_DIR/app/"
cp "$ROOT_DIR/app/evaluation.py" "$OUT_DIR/app/"
cp "$ROOT_DIR/app/utils.py" "$OUT_DIR/app/"
cp "$ROOT_DIR/app/uncertainty/__init__.py" "$OUT_DIR/app/uncertainty/"
cp "$ROOT_DIR/app/uncertainty/quantile_xgboost.py" "$OUT_DIR/app/uncertainty/"
cp "$ROOT_DIR/app/demo/__init__.py" "$OUT_DIR/app/demo/"
cp "$ROOT_DIR/app/demo/streamlit_app.py" "$OUT_DIR/app/demo/"
cp "$ROOT_DIR/data/traffic.csv" "$OUT_DIR/data/"
cp "$ROOT_DIR/deploy/space_requirements.txt" "$OUT_DIR/requirements.txt"
cp "$ROOT_DIR/deploy/space_readme.md" "$OUT_DIR/README.md"

echo "Bundle ready: $OUT_DIR"
echo "Deploy with: hf upload <username>/<space-name> $OUT_DIR . --repo-type space"

#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

SKIP_EMBEDDINGS=0
SKIP_TRAINING=0
SKIP_EXPORT=0
WITH_TRENDS=0
WITH_TRENDYOL=0

usage() {
  cat <<'EOF'
Usage: scripts/run_production_pipeline.sh [options]

Options:
  --skip-embeddings      Skip item2vec embedding generation
  --skip-training        Skip leakage-safe ranker training
  --skip-export          Skip dashboard artifact export
  --with-trends          Run google trends generation
  --with-trendyol        Run trendyol category scrape
  -h, --help             Show this help

Env:
  PYTHON_BIN=python3.12  Override python executable (default: python)
EOF
}

log() {
  printf '[pipeline] %s\n' "$1"
}

run() {
  log "RUN: $*"
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-embeddings)
      SKIP_EMBEDDINGS=1
      shift
      ;;
    --skip-training)
      SKIP_TRAINING=1
      shift
      ;;
    --skip-export)
      SKIP_EXPORT=1
      shift
      ;;
    --with-trends)
      WITH_TRENDS=1
      shift
      ;;
    --with-trendyol)
      WITH_TRENDYOL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log "Project root: $ROOT_DIR"
run "$PYTHON_BIN" -c "import sys; print('python:', sys.executable)"

run "$PYTHON_BIN" src/data_processing/build_baskets_tables.py
run "$PYTHON_BIN" src/data_generation/build_synthetic_customers.py

if [[ "$SKIP_EMBEDDINGS" -eq 0 ]]; then
  run "$PYTHON_BIN" src/data_generation/build_product_embeddings_item2vec.py --max-anchor-items 0 --max-neighbors-per-item 20
else
  log "Skip embeddings step"
fi

run "$PYTHON_BIN" src/data_generation/build_category_tree_from_marketsales.py

if [[ "$WITH_TRENDS" -eq 1 ]]; then
  run "$PYTHON_BIN" src/data_generation/google_trends_from_marketsales.py
fi

if [[ "$WITH_TRENDYOL" -eq 1 ]]; then
  run "$PYTHON_BIN" src/data_generation/scrape_trendyol_category_tree.py
fi

if [[ "$SKIP_TRAINING" -eq 0 ]]; then
  run "$PYTHON_BIN" scripts/train_ranker_leakage_safe.py
else
  log "Skip training step"
fi

if [[ "$SKIP_EXPORT" -eq 0 ]]; then
  run "$PYTHON_BIN" scripts/export_dashboard_artifacts.py
else
  log "Skip dashboard export step"
fi

log "Pipeline completed"

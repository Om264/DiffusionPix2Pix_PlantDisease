#!/usr/bin/env bash
# Prepare REAL data: validate -> split -> masks -> severity.   Usage: bash scripts/prepare.sh ./data/raw ./data/processed
set -euo pipefail
RAW=${1:-./data/raw}; PROC=${2:-./data/processed}
python -m preprocessing.prepare_dataset --data_dir "$RAW" --output_dir "$PROC"
python -m preprocessing.prepare_masks --processed_dir "$PROC" --method hsv      # or: --method existing --masks_dir ...
python -m preprocessing.estimate_severity --processed_dir "$PROC"                # skip if you have REAL severity labels

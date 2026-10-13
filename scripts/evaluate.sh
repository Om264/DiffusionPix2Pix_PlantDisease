#!/usr/bin/env bash
# Generation metrics + combined datasets + downstream classifiers.
set -euo pipefail
PROC=${1:-./outputs/processed}
python -m evaluation.evaluate_generation --real_dir "$PROC/test" --synthetic_dir ./outputs/filtered --output_dir ./outputs/metrics
python -m preprocessing.build_dataset --processed_dir "$PROC" --synthetic_dir ./outputs/filtered --output_dir ./outputs/combined
python -m evaluation.evaluate_downstream --real_dir "$PROC" --synthetic_dir ./outputs/filtered --seeds 0 1 2
python -m evaluation.visualization grid --synthetic_dir ./outputs/filtered

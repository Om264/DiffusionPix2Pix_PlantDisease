#!/usr/bin/env bash
# Generate >=10k images, filter, deduplicate. Usage: bash scripts/generate_10k.sh [TOTAL=10000]
# NOTE: generates TOTAL*1.3 images so that ~TOTAL survive quality control; run_pipeline.py does the same plus extra rounds.
set -euo pipefail
TOTAL=${1:-10000}
python -m generation.generate --checkpoint ./outputs/checkpoints/best.pt --total_images "$TOTAL" --oversample 1.3 \
    --balance_mode uniform --output_dir ./outputs/generated --resume
python -m quality.quality_control --input_dir ./outputs/generated --output_dir ./outputs/filtered

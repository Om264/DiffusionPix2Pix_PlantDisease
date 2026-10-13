#!/usr/bin/env bash
# Usage: bash scripts/train.sh [extra --set overrides...]    (resumes automatically if outputs/checkpoints/latest.pt exists)
set -euo pipefail
python -m training.train --config configs/training.yaml --resume --set "$@"

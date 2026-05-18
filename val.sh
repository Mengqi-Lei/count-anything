#!/usr/bin/env bash
set -euo pipefail

# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash val.sh

CONFIG_PATH="${CONFIG_PATH:-config/count_anything_val_cloc.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
python -m count_anything.train.train \
  -c "${CONFIG_PATH}" \
  --use-cluster 0 \
  --num-gpus "${NUM_GPUS}"

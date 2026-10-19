#!/usr/bin/env bash
set -euo pipefail
# Run the five folds sequentially on the selected GPU; existing runs resume.
WORKSPACE="$1"
RAW_DATASET="$2"
GPU="${3:-0}"
VARIANT="${4:-adaptive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAME="$($PYTHON_BIN -c 'import json,sys,pathlib; print(json.loads((pathlib.Path(sys.argv[1])/"prepared.json").read_text())["dataset_name"])' "$WORKSPACE")"
EPOCHS="${5:-700}"
SEED="${6:-20261017}"
for fold in 0 1 2 3 4; do
  extra=()
  case "$VARIANT" in
    adaptive) trainer=nnUNetTrainerPolarityDAdaptive ;;
    fixed) trainer=nnUNetTrainerPolarityDFixed ;;
    baseline) trainer=nnUNetTrainerPolarityDBaseline ;;
    *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
  esac
  if [[ -f "$WORKSPACE/nnUNet_results/${DATASET_NAME}/${trainer}__nnUNetPlans__3d_fullres/fold_${fold}/provenance.json" ]]; then
    extra=(--resume)
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/train.py" --workspace "$WORKSPACE" --raw "$RAW_DATASET" \
    --variant "$VARIANT" --fold "$fold" --gpu "$GPU" --epochs "$EPOCHS" --seed "$SEED" "${extra[@]}"
done

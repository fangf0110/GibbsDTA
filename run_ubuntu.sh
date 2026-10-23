#!/bin/bash

# 1. Read parameters (with defaults)
DATASET=${1:-kiba}
EPOCHS=${2:-100}
BATCH_SIZE=${3:-256}
LR=${4:-0.001}
SMI_ENCODER=${5:-chemberta}
EXTRA_ARGS="${@:6}"

LOG_FILE="phynet_5fold_${DATASET}_${SMI_ENCODER}_nohup.log"

echo "=================================================="
echo "   PhyNet-DTA 5-Fold Cross Validation Launcher"
echo "   Dataset:    $DATASET"
echo "   Epochs:     $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Learning R: $LR"
echo "   Encoder:    $SMI_ENCODER"
echo "   Extra Args: $EXTRA_ARGS"
echo "   Log File:   $LOG_FILE"
echo "=================================================="

# 2. Run the 5-fold CV in the background
nohup bash -c '
# Activate virtual environment if it exists
source .venv/bin/activate 2>/dev/null || true

for i in {0..4}
do
    echo ""
    echo "==================================="
    echo "Running Fold $i..."
    echo "==================================="
    python -u run.py --dataset '"$DATASET"' --epochs '"$EPOCHS"' --batch_size '"$BATCH_SIZE"' --lr '"$LR"' --smi_encoder '"$SMI_ENCODER"' --val_fold $i '"$EXTRA_ARGS"'
done

echo ""
echo "All folds completed! Results saved to results/ directory."
' > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started in background. PID: $PID"
echo "To watch progress, run:"
echo "  tail -f $LOG_FILE"
echo "To stop, run:"
echo "  kill $PID"
echo "=================================================="

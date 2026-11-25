#!/usr/bin/env bash
# Example training pipeline for Qwen3-TTS
# This script demonstrates a complete training workflow

set -e

# Configuration
DEVICE="cuda:0"
PROJECT_NAME="my_speaker"
SPEAKER_NAME="speaker_${PROJECT_NAME}"

RAW_JSONL="data/${PROJECT_NAME}/train_raw.jsonl"
VAL_JSONL="data/${PROJECT_NAME}/val_raw.jsonl"
PREPARED_JSONL="data/${PROJECT_NAME}/train_with_codes.jsonl"
VAL_PREPARED_JSONL="data/${PROJECT_NAME}/val_with_codes.jsonl"

OUTPUT_DIR="output/${PROJECT_NAME}"
CONFIG_FILE="train_config.yaml"

# Test text for inference
TEST_TEXT="She said she would be here by noon."

echo "=================================="
echo "Qwen3-TTS Training Pipeline"
echo "=================================="
echo "Project: $PROJECT_NAME"
echo "Speaker: $SPEAKER_NAME"
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo "=================================="

# Step 1: Check data exists
if [ ! -f "$RAW_JSONL" ]; then
    echo "Error: Training data not found at $RAW_JSONL"
    echo "Please prepare your data in JSONL format first."
    exit 1
fi

echo ""
echo "Step 1: Preparing training data..."
python train_pipeline.py \
    --config $CONFIG_FILE \
    --mode prepare \
    --raw_data "$RAW_JSONL" \
    --device $DEVICE

# Step 2: Prepare validation data if it exists
if [ -f "$VAL_JSONL" ]; then
    echo ""
    echo "Step 2: Preparing validation data..."
    python prepare_data.py \
        --device $DEVICE \
        --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
        --input_jsonl "$VAL_JSONL" \
        --output_jsonl "$VAL_PREPARED_JSONL"
    VAL_FLAG="--val_data $VAL_PREPARED_JSONL"
else
    echo ""
    echo "Step 2: No validation data provided (optional)"
    VAL_FLAG=""
fi

# Step 3: Update config with current project paths
echo ""
echo "Step 3: Updating configuration..."
cat > "${OUTPUT_DIR}_config.yaml" <<EOF
init_model_path: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
output_model_path: "$OUTPUT_DIR"
train_jsonl: "$PREPARED_JSONL"
val_jsonl: $([ -f "$VAL_PREPARED_JSONL" ] && echo "\"$VAL_PREPARED_JSONL\"" || echo "null")

batch_size: 2
learning_rate: 2e-5
num_epochs: 3
speaker_name: "$SPEAKER_NAME"

warmup_ratio: 0.1
max_grad_norm: 1.0
weight_decay: 0.01
gradient_accumulation_steps: 4

save_steps: 500
eval_steps: 500

mixed_precision: "bf16"
attn_implementation: "flash_attention_2"
seed: 42
EOF

# Step 4: Train model
echo ""
echo "Step 4: Starting training..."
python train_pipeline.py \
    --config "${OUTPUT_DIR}_config.yaml" \
    --mode train \
    --train_data "$PREPARED_JSONL" \
    $VAL_FLAG

# Step 5: Test inference
echo ""
echo "Step 5: Testing inference..."
BEST_CHECKPOINT="${OUTPUT_DIR}/best_model"
if [ ! -d "$BEST_CHECKPOINT" ]; then
    BEST_CHECKPOINT="${OUTPUT_DIR}/checkpoint-epoch-2"
fi

python train_pipeline.py \
    --config "${OUTPUT_DIR}_config.yaml" \
    --mode infer \
    --checkpoint "$BEST_CHECKPOINT" \
    --text "$TEST_TEXT" \
    --output_wav "${OUTPUT_DIR}/test_output.wav"

echo ""
echo "=================================="
echo "Training completed successfully!"
echo "=================================="
echo "Best model: $BEST_CHECKPOINT"
echo "Test output: ${OUTPUT_DIR}/test_output.wav"
echo "Training log: ${OUTPUT_DIR}/training.log"
echo "=================================="

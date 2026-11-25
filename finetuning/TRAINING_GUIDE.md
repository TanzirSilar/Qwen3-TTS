# Qwen3-TTS Complete Training Guide

This guide explains how to use the new comprehensive training pipeline for Qwen3-TTS.

## Quick Start

### 1. Setup

Ensure you have the editable install in place:
```bash
cd /speech/tanzir/Qwen3-TTS
pip install -e .
```

### 2. Prepare Your Data

Create a raw training JSONL file (`train_raw.jsonl`) with the following format:

```jsonl
{"audio":"./data/utt0001.wav","text":"Your text here","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"More text here","ref_audio":"./data/ref.wav"}
```

### 3. Run Full Pipeline

```bash
cd finetuning
python train_pipeline.py \
  --config train_config.yaml \
  --mode full \
  --raw_data train_raw.jsonl \
  --test_text "She said she would be here by noon."
```

This will:
1. ✅ Prepare data (extract audio codes)
2. ✅ Train the model
3. ✅ Test inference on the trained model

## Detailed Usage

### Configuration

Edit `train_config.yaml` to customize training:

```yaml
# Model
init_model_path: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
output_model_path: "output"

# Data
train_jsonl: "train_with_codes.jsonl"
val_jsonl: null  # Optional validation set

# Training
batch_size: 2
learning_rate: 2e-5
num_epochs: 3
speaker_name: "speaker_test"

# Optimization
warmup_ratio: 0.1
max_grad_norm: 1.0
gradient_accumulation_steps: 4

# Checkpointing
save_steps: 500
eval_steps: 500
```

### Pipeline Modes

#### Mode 1: Prepare Data Only

Extract audio codes from raw JSONL:

```bash
python train_pipeline.py \
  --config train_config.yaml \
  --mode prepare \
  --raw_data train_raw.jsonl \
  --device cuda:0
```

Output: `train_raw_with_codes.jsonl`

#### Mode 2: Train Only

Train with already-prepared data:

```bash
python train_pipeline.py \
  --config train_config.yaml \
  --mode train \
  --train_data train_with_codes.jsonl \
  --val_data val_with_codes.jsonl  # Optional
```

Output: `output/checkpoint-epoch-X`, `output/best_model/`

#### Mode 3: Inference Only

Generate audio with a trained model:

```bash
python train_pipeline.py \
  --config train_config.yaml \
  --mode infer \
  --checkpoint output/best_model \
  --text "Your text here" \
  --output_wav result.wav
```

#### Mode 4: Full Pipeline

Run the complete pipeline in one command:

```bash
python train_pipeline.py \
  --config train_config.yaml \
  --mode full \
  --raw_data train_raw.jsonl \
  --val_data val_raw.jsonl \
  --test_text "Test text after training" \
  --device cuda:0
```

## Training Script Details

### Advanced Training Features

The `train_qwen3_tts.py` script includes:

- **Mixed Precision Training**: BF16 for memory efficiency
- **Gradient Accumulation**: Effective larger batch sizes
- **Learning Rate Scheduling**: Warmup + Cosine Annealing
- **Gradient Clipping**: Prevents training instability
- **Checkpointing**: Save per-epoch and per-step
- **Validation**: Optional validation set evaluation
- **Detailed Logging**: Training metrics to file and console

### Training Loop

```python
from train_qwen3_tts import Qwen3TTSTrainer

trainer = Qwen3TTSTrainer(
    init_model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    output_dir="output",
    train_jsonl="train_with_codes.jsonl",
    val_jsonl="val_with_codes.jsonl",  # Optional
    batch_size=2,
    lr=2e-5,
    num_epochs=3,
    speaker_name="speaker_test",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    save_steps=500,
    eval_steps=500,
    gradient_accumulation_steps=4,
)

trainer.train()
```

## Output Structure

After training, your output directory will contain:

```
output/
├── checkpoint-epoch-0/
│   ├── config.json
│   ├── model.safetensors
│   └── other model files...
├── checkpoint-epoch-1/
├── checkpoint-epoch-2/
├── best_model/  # Symlink to best checkpoint
├── logs/        # TensorBoard logs
└── training.log # Training log file
```

## Inference

### Using Trained Model

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/best_model",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="speaker_test",
)
sf.write("output.wav", wavs[0], sr)
```

## Troubleshooting

### Out of Memory (OOM)

Reduce `batch_size` or increase `gradient_accumulation_steps`:

```yaml
batch_size: 1
gradient_accumulation_steps: 8
```

### Slow Training

- Increase `batch_size` if you have VRAM
- Use `save_steps: 0` and `eval_steps: 0` to disable intermediate checkpointing
- Reduce `eval_steps` frequency

### Poor Audio Quality

- Increase `num_epochs`
- Reduce `learning_rate` (try 1e-5)
- Use more training data
- Ensure `ref_audio` is consistent and high-quality

### GPU Errors

Ensure you have:
- CUDA >= 11.8
- PyTorch with CUDA support
- Flash Attention 2 installed: `pip install flash-attn --no-build-isolation`

## Comparison: Old vs New Training

### Old Script (sft_12hz.py)

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_test
```

### New Script (train_qwen3_tts.py)

```bash
python train_qwen3_tts.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name speaker_test \
  --warmup_ratio 0.1 \
  --save_steps 500 \
  --eval_steps 500 \
  --gradient_accumulation_steps 4
```

**New Features:**
- ✅ Learning rate scheduling (warmup + cosine annealing)
- ✅ Validation set support
- ✅ Per-step checkpointing (not just per-epoch)
- ✅ Detailed metric tracking
- ✅ Better error handling and logging
- ✅ Configurable checkpointing frequency
- ✅ Gradient accumulation for effective larger batches

## Best Practices

1. **Data Preparation**
   - Use consistent reference audio across all samples
   - Ensure audio is at 24kHz (the script will check this)
   - Use high-quality wav files
   - Mix different speakers in validation set for generalization

2. **Training**
   - Start with small `batch_size` and increase if VRAM allows
   - Use `warmup_ratio=0.1` for stability
   - Monitor validation loss if available
   - Save checkpoints frequently with `save_steps`

3. **Hyperparameter Tuning**
   - `lr`: 1e-6 to 5e-5 (try 2e-5 first)
   - `num_epochs`: 3-10 (more data = fewer epochs)
   - `batch_size`: Limited by VRAM
   - `warmup_ratio`: 0.05-0.2

4. **Hardware**
   - Requires GPU with >= 20GB VRAM for 1.7B model
   - Use mixed precision (BF16) for memory efficiency
   - Use Flash Attention 2 for speed

## For More Information

- **Original Model**: https://github.com/QwenLM/Qwen3-TTS
- **Training Scripts**: See `train_qwen3_tts.py` and `train_pipeline.py`
- **Dataset Format**: See README.md in this directory

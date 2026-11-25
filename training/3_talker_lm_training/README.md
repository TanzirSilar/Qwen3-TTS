# Phase 3: Talker LLM Training

Train the main transformer LLM that generates speech codes from text and speaker embeddings.

## Overview

The Talker LLM:
- **Input**: Text tokens + previous audio codes + speaker embedding
- **Output**: Next code predictions (16 codebooks)
- **Architecture**: Causal transformer decoder (1.7B parameters)
- **Purpose**: Auto-regressive speech generation

## Architecture

```
Text Input
    ↓
Text Embedding (from tokenizer)
    ↓
+
+─→ Previous Audio Codes
    ↓
+
+─→ Speaker Embedding (from speaker encoder)
    ↓
Input Embeddings Combined
    ↓
Transformer Decoder (32 layers)
    ├── Self-Attention with RoPE
    ├── Feed-Forward (4x expansion)
    └── RMSNorm
    ↓
Output Projection
    ↓
16x Codec Head (one per codebook)
    ↓
Codec Code Predictions [0-2047]
```

## Training

### Configuration
Edit `talker_config.yaml`:
```yaml
# Architecture
vocab_size: 151936           # Text vocabulary
hidden_size: 4096
intermediate_size: 22016
num_hidden_layers: 32
num_attention_heads: 32
num_key_value_heads: 32
head_dim: 128

# Input/Output
max_position_embeddings: 2048
num_codebooks: 16
codebook_size: 2048

# Training
batch_size: 32               # Per GPU
learning_rate: 2e-5
num_epochs: 10
warmup_ratio: 0.1            # 10% of steps
max_grad_norm: 1.0
gradient_accumulation_steps: 4

# Losses
codec_0_weight: 1.0          # Main codec loss
sub_codec_weight: 0.3        # Auxiliary codecs
attention_loss_weight: 0.0   # Optional

# Model
mixed_precision: "bf16"
attn_implementation: "flash_attention_2"
gradient_checkpointing: true

# Data
max_seq_length: 2048
min_audio_length: 1.0        # seconds
max_audio_length: 20.0
```

### Prepare Data

Before training, pre-process data:

```bash
# 1. Extract speaker embeddings
python extract_speaker_embeddings.py \
  --speaker_encoder_checkpoint ../../checkpoints/speaker_encoder/best_model.pt \
  --train_manifest ../../datasets/talker_lm_data/manifests/train_manifest.jsonl \
  --output_dir ../../datasets/talker_lm_data/processed/speaker_embeddings

# 2. Tokenize audio with trained tokenizer
python encode_audio.py \
  --tokenizer_checkpoint ../../checkpoints/tokenizer/best_model.pt \
  --train_manifest ../../datasets/talker_lm_data/manifests/train_manifest.jsonl \
  --output_dir ../../datasets/talker_lm_data/processed/audio_codes
```

### Run Training

```bash
# Single GPU
python train_talker_lm.py \
  --config talker_config.yaml \
  --train_manifest ../../datasets/talker_lm_data/manifests/train_manifest.jsonl \
  --val_manifest ../../datasets/talker_lm_data/manifests/val_manifest.jsonl \
  --speaker_embeddings ../../datasets/talker_lm_data/processed/speaker_embeddings \
  --audio_codes ../../datasets/talker_lm_data/processed/audio_codes \
  --output_dir ../../checkpoints/talker_lm

# Multi-GPU with Accelerate (recommended)
accelerate launch train_talker_lm.py \
  --config talker_config.yaml \
  --train_manifest ../../datasets/talker_lm_data/manifests/train_manifest.jsonl \
  --val_manifest ../../datasets/talker_lm_data/manifests/val_manifest.jsonl \
  --speaker_embeddings ../../datasets/talker_lm_data/processed/speaker_embeddings \
  --audio_codes ../../datasets/talker_lm_data/processed/audio_codes \
  --output_dir ../../checkpoints/talker_lm
```

## Key Concepts

### Auto-Regressive Generation
Predicts codes sequentially:
```
Step 1: P(codec_0[t] | text, previous_codes) → codec_0[t]
Step 2: P(codec_0[t+1] | text, previous_codes + codec_0[t])
...
```

### Multi-Codebook Prediction
Predicts all 16 codebooks simultaneously:
```
Main Loss: Predict codec_0 (primary acoustic content)
Aux Loss:  Predict codecs 1-15 (fine details, tone, prosody)

Total Loss = L_codec_0 + 0.3 * L_codecs_1-15
```

### Learning Rate Scheduling
```
Warmup phase (10% of steps): LR increases linearly
Main phase: LR decreases with cosine annealing
```

## Monitoring

### TensorBoard
```bash
tensorboard --logdir ../../logs/talker_lm/runs/
```

Watch for:
- **loss/total**: Should steadily decrease
- **loss/codec_0**: Main codec prediction loss
- **loss/codec_aux**: Auxiliary codec prediction loss
- **learning_rate**: Warmup then cosine decay
- **validation_loss**: Track generalization

### Logs
```bash
tail -f ../../logs/talker_lm/training.log
```

Expected output:
```
Epoch 2 | Step 1000 | Loss: 2.1234
├── Codec-0 Loss: 1.8234
├── Codec-Aux Loss: 0.3000
├── Learning Rate: 1.5e-5
└── Val Loss: 2.3456
```

## Validation Metrics

The training script evaluates generation quality:

1. **Loss Metrics**
   - Codec-0 loss (main acoustic content)
   - Auxiliary codec loss
   - Total loss

2. **Generation Quality** (optional, slow)
   - Decode generated codes → audio
   - Compute WER using ASR model
   - Compute speaker similarity

3. **Diversity**
   - Entropy of predicted code distributions
   - Temperature analysis

## Checkpointing

Saves at:
- Every step: `checkpoint-step-{N}/` (every 500 steps by default)
- Best model (lowest val loss): `best_model/`
- Latest: `latest/`

Each checkpoint contains:
- Model weights (safetensors format)
- Optimizer state
- Learning rate scheduler
- Training statistics
- Configuration

Resume training:
```bash
python train_talker_lm.py \
  --config talker_config.yaml \
  --resume_from ../../checkpoints/talker_lm/checkpoint-step-5000/
```

## Expected Results

After training on 100 hours with 2000+ speakers (~2 weeks on 8x A100):

| Metric | Expected Value |
|--------|---|
| Validation Loss | 1.8-2.2 |
| Codec-0 Loss | 1.5-1.8 |
| Codec-Aux Loss | 0.3-0.4 |
| Generation WER | <5% |
| Speaker Similarity | >0.80 |

## Common Issues

**Problem: Loss Plateaus**
- Loss stops decreasing
- **Solution**:
  - Increase learning rate
  - Check data quality/diversity
  - Longer training

**Problem: Instability**
- Loss spikes randomly
- **Solution**:
  - Reduce learning rate
  - Increase warmup steps
  - Reduce gradient_accumulation_steps

**Problem: Poor Generation Quality**
- Generated audio doesn't match text
- **Solution**:
  - Train longer
  - More diverse data
  - Adjust loss weights
  - Check text-audio alignment

**Problem: OOM Errors**
- Out of GPU memory
- **Solution**:
  - Reduce `batch_size`
  - Reduce `max_seq_length`
  - Enable gradient_checkpointing
  - Use flash_attention_2

## Advanced Features

### Gradient Checkpointing
Trade compute for memory:
```yaml
gradient_checkpointing: true
```

### Flash Attention 2
Faster attention with less memory:
```yaml
attn_implementation: "flash_attention_2"
```

### Distributed Training
Use multiple GPUs/nodes:
```bash
accelerate config  # Configure your setup
accelerate launch train_talker_lm.py ...
```

## Inference / Testing

After training:

1. **Validate the LLM**:
   ```bash
   python ../../evaluation/eval_talker_lm.py \
     --checkpoint ../../checkpoints/talker_lm/best_model/
   ```

2. **Generate Speech**:
   ```python
   from train_talker_lm import TalkerLLM
   from qwen_tts import Qwen3TTSModel
   
   # Load trained model
   model = Qwen3TTSModel.from_pretrained(
       "checkpoints/talker_lm/best_model/",
       tokenizer_checkpoint="checkpoints/tokenizer/best_model.pt",
       speaker_encoder_checkpoint="checkpoints/speaker_encoder/best_model.pt"
   )
   
   # Generate
   wavs, sr = model.generate_voice_clone(
       text="Hello world",
       language="English",
       ref_audio="reference.wav"
   )
   ```

3. **Full Integration**:
   - Combine tokenizer + speaker encoder + talker LLM
   - Package as `Qwen3TTSModel`
   - Run inference tests

---

**Time to complete**: 2-4 weeks on 8x A100 GPUs
**Next Phase**: Integration & Testing

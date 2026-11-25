# Phase 2: Speaker Encoder Training

Train the ECAPA-TDNN speaker embedding extractor for speaker-aware TTS.

## Overview

The speaker encoder:
- **Input**: Mel-spectrogram from reference speech
- **Output**: 1024-dimensional speaker embedding
- **Architecture**: ECAPA-TDNN with Res2Net blocks
- **Purpose**: Extract speaker identity for voice cloning

## Architecture

```
Reference Audio (24kHz)
    ↓
Mel-Spectrogram (128 dims)
    ↓
TDNN Layer
    ↓
4x Res2Net Blocks
    ↓
Attentive Statistics Pooling
    ↓
Classification / Metric Head
    ↓
Speaker Embedding (1024-dim)
```

## Training

### Configuration
Edit `speaker_config.yaml`:
```yaml
# Architecture
mel_dim: 128                 # Mel-spectrogram dimension
enc_dim: 1024                # Embedding dimension
enc_channels: [512, 512, 512, 512, 1536]
enc_kernel_sizes: [5, 3, 3, 3, 1]
enc_dilations: [1, 2, 3, 4, 1]
enc_attention_channels: 128
enc_res2net_scale: 8

# Training
batch_size: 128              # Total batch size (nsamples * utterances_per_speaker)
learning_rate: 1e-3
num_epochs: 50
warmup_steps: 5000
max_grad_norm: 1.0

# Metric Learning
num_speakers: 1000           # In training dataset
samples_per_speaker: 2       # Utterances per speaker per batch
metric: "arcface"            # or "cosface"
arcface_margin: 0.3
arcface_scale: 30.0

# Loss
ce_weight: 1.0               # Classification loss weight
metric_weight: 1.0           # Metric loss weight
```

### Run Training
```bash
# Single GPU
python train_speaker_encoder.py \
  --config speaker_config.yaml \
  --train_manifest ../../datasets/speaker_encoder_data/manifests/train_manifest.jsonl \
  --val_manifest ../../datasets/speaker_encoder_data/manifests/val_manifest.jsonl \
  --output_dir ../../checkpoints/speaker_encoder

# Multi-GPU with Accelerate
accelerate launch train_speaker_encoder.py \
  --config speaker_config.yaml \
  --train_manifest ../../datasets/speaker_encoder_data/manifests/train_manifest.jsonl \
  --val_manifest ../../datasets/speaker_encoder_data/manifests/val_manifest.jsonl \
  --output_dir ../../checkpoints/speaker_encoder
```

## Key Concepts

### ECAPA-TDNN Architecture
- **TDNN**: Time Delay Neural Network for temporal modeling
- **Res2Net**: Multi-scale residual blocks
- **Attentive Statistics Pooling**: Weighted averaging of frame embeddings
- **Classification Head**: Softmax over speaker IDs

### Metric Learning
Learns embeddings where same speakers are close and different speakers are far apart.

**ArcFace Loss**:
```
Loss = cross_entropy(softmax(s*cos(θ + m)), speaker_id)
where:
  s = scale factor (30)
  m = margin (0.3)
  θ = angle between embedding and class center
```

### Batch Sampling
Important for metric learning:
```
Per-batch speaker diversity: num_speakers * samples_per_speaker
Example: 32 speakers × 4 utterances = 128 batch size
```

## Monitoring

### TensorBoard
```bash
tensorboard --logdir ../../logs/speaker_encoder/runs/
```

Watch for:
- **classification_loss**: Should decrease
- **arcface_loss**: Should decrease
- **train_accuracy**: Should increase to >99%
- **val_accuracy**: Should increase

### Logs
```bash
tail -f ../../logs/speaker_encoder/training.log
```

Expected output:
```
Epoch 5 | Step 100 | Loss: 1.2345
├── CE Loss: 0.5234
├── ArcFace Loss: 0.7111
├── Train Accuracy: 98.5%
└── Val EER: 0.85%
```

## Validation Metrics

The training script evaluates speaker verification performance:

1. **Classification Accuracy**
   - % of correct speaker predictions on validation set

2. **Speaker Verification**
   - **EER** (Equal Error Rate): Lower is better (target: <1%)
   - **minDCF**: Minimum Detection Cost Function
   - **Cosine Similarity**: Distribution of speaker similarity scores

3. **Embedding Quality**
   - Intra-speaker variance (should be low)
   - Inter-speaker distance (should be high)
   - Embedding norm stability

## Checkpointing

Saves at:
- Every epoch: `checkpoint-epoch-{N}.pt`
- Best model (lowest EER): `best_model.pt`

Each checkpoint contains:
- Model weights
- Optimizer state
- Best validation EER
- Configuration

Resume training:
```bash
python train_speaker_encoder.py \
  --config speaker_config.yaml \
  --resume_from ../../checkpoints/speaker_encoder/checkpoint-epoch-10.pt
```

## Expected Results

After training on 100k speaker utterances (~1 week on 4x A100):

| Metric | Expected Value |
|--------|---|
| Train Accuracy | >99% |
| Val Accuracy | >95% |
| EER (Equal Error Rate) | <2% |
| Speaker Verification | >0.85 similarity |

## Common Issues

**Problem: Poor Speaker Separation**
- EER too high (>5%)
- **Solution**:
  - Increase training duration
  - Add data augmentation
  - Use larger model

**Problem: Overfitting**
- Train accuracy high but val accuracy low
- **Solution**:
  - Add regularization (weight decay)
  - Use dropout
  - Add data augmentation

**Problem: Slow Convergence**
- Loss not decreasing
- **Solution**:
  - Increase learning rate
  - Adjust batch size
  - Reduce warmup steps

## Data Augmentation

The training script supports:
- SpecAugment: Time and frequency masking
- Mixup: Blend multiple speakers
- TimeStretch: Speed perturbation

Enable in config:
```yaml
augmentation:
  spec_augment: true
  mixup: true
  time_stretch: false
```

## After Training

Once speaker encoder training is complete:

1. **Validate the encoder**:
   ```bash
   python ../../evaluation/eval_speaker_encoder.py \
     --checkpoint ../../checkpoints/speaker_encoder/best_model.pt
   ```

2. **Extract speaker embeddings**:
   ```python
   from train_speaker_encoder import SpeakerEncoder
   encoder = SpeakerEncoder.from_pretrained("checkpoints/speaker_encoder/best_model.pt")
   
   # Get embedding from reference audio
   mel_spec = ...  # Load mel-spectrogram
   embedding = encoder(mel_spec)
   ```

3. **Use for Talker LLM training**:
   - Pre-compute all speaker embeddings
   - Use in Talker LLM as conditioning

---

**Time to complete**: 3-5 days on 4x A100 GPUs
**Next Phase**: Talker LLM Training

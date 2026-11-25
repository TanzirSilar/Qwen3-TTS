# Phase 1: Tokenizer Training

Train the VQ-VAE codec that compresses 24kHz speech into discrete multi-codebook tokens.

## Overview

The tokenizer is a Vector-Quantized Variational Autoencoder (VQ-VAE) that:
- **Input**: Waveform at 24kHz (continuous signal)
- **Output**: 16 discrete codebooks with 2048 tokens each
- **Compression**: 24kHz → 12.5 FPS (80ms frames), 16 codes per frame
- **Purpose**: Encode speech into discrete tokens for LLM training

## Architecture

```
Input Audio (24kHz)
    ↓
Mel-Spectrogram (128 dims)
    ↓
Encoder (Conv1d layers)
    ↓
16x VQ Layers (Vector Quantization)
    ↓
Decoder (ConvTranspose1d layers)
    ↓
Reconstructed Mel-Spectrogram
    ↓
Loss: Reconstruction + VQ + Commitment
```

## Training

### Configuration
Edit `tokenizer_config.yaml`:
```yaml
# Architecture
sample_rate: 24000           # Input audio sample rate
num_mels: 128                # Mel-spectrogram dimensions
num_codebooks: 16            # Number of VQ codebooks
codebook_size: 2048          # Tokens per codebook
embedding_dim: 64            # Embedding dimension

# Training
batch_size: 32               # Per GPU
learning_rate: 1e-4
num_epochs: 100
warmup_steps: 10000
max_grad_norm: 1.0

# Losses
reconstruction_weight: 1.0
commitment_weight: 0.25
adversarial_weight: 0.1
```

### Run Training
```bash
# Single GPU
python train_tokenizer.py \
  --config tokenizer_config.yaml \
  --train_manifest ../../datasets/tokenizer_data/manifests/train_manifest.jsonl \
  --val_manifest ../../datasets/tokenizer_data/manifests/val_manifest.jsonl \
  --output_dir ../../checkpoints/tokenizer

# Multi-GPU with Accelerate
accelerate launch train_tokenizer.py \
  --config tokenizer_config.yaml \
  --train_manifest ../../datasets/tokenizer_data/manifests/train_manifest.jsonl \
  --val_manifest ../../datasets/tokenizer_data/manifests/val_manifest.jsonl \
  --output_dir ../../checkpoints/tokenizer
```

## Key Concepts

### Vector Quantization (VQ)
Maps continuous embeddings to nearest discrete tokens:
```
Continuous feature → Find nearest codebook vector → Discrete token
```

This creates a bottleneck that forces the model to learn efficient representations.

### Loss Function
```
L_total = L_reconstruction + 0.25 * L_commitment + 0.1 * L_adversarial

L_reconstruction = MSE(input, reconstructed)
L_commitment = ||e - sg[z]||²  (commitment loss)
L_adversarial = adversarial loss from discriminator
```

### Training Stability Tips
1. Start with low learning rate (1e-4)
2. Monitor codebook usage (all codes should be used)
3. If codebook collapse occurs, increase codebook_size or batch_size
4. Use gradient clipping (max_grad_norm=1.0)
5. Validate frequently (every N steps)

## Monitoring

### TensorBoard
```bash
tensorboard --logdir ../../logs/tokenizer/runs/
```

Watch for:
- **reconstruction_loss**: Should decrease over time
- **codebook_usage**: % of codebook tokens used (should be >90%)
- **perplexity**: Effective codebook size (higher = better usage)

### Logs
```bash
tail -f ../../logs/tokenizer/training.log
```

Expected output:
```
Epoch 0 | Step 100 | Loss: 2.3451
├── Reconstruction: 2.1234
├── VQ Loss: 0.1234
├── Codebook Usage: 85.3%
└── Perplexity: 1847.3
```

## Validation Metrics

The training script automatically evaluates:

1. **Reconstruction Quality**
   - MSE Loss
   - L1 Loss
   - Perceptual Loss (optional)

2. **Codebook Efficiency**
   - % of active codes
   - Perplexity (measure of effective codebook size)
   - Code frequency distribution

3. **Reconstruction Audio Quality**
   - PESQ score (perceptual quality)
   - STOI score (intelligibility)
   - SNR (signal-to-noise ratio)

## Checkpointing

Saves at:
- Every epoch: `checkpoint-epoch-{N}.pt`
- Best model (lowest val loss): `best_model.pt`
- Latest: `latest.pt`

Each checkpoint contains:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Training step counter
- Configuration

Resume training:
```bash
python train_tokenizer.py \
  --config tokenizer_config.yaml \
  --resume_from ../../checkpoints/tokenizer/checkpoint-epoch-5.pt
```

## Expected Results

After training on 100 hours of data (~2 weeks on 8x A100):

| Metric | Expected Value |
|--------|---|
| Reconstruction PESQ | 2.8-3.2 |
| STOI Score | 0.90-0.94 |
| Codebook Usage | >90% |
| Perplexity | >1500 |

## Common Issues

**Problem: Codebook Collapse**
- All tokens map to same code
- **Solution**: 
  - Increase `codebook_size`
  - Reduce learning rate
  - Increase batch size

**Problem: High Reconstruction Loss**
- Poor audio reconstruction
- **Solution**:
  - Train longer
  - Increase model capacity
  - Check audio quality in dataset

**Problem: OOM Errors**
- Out of memory on GPU
- **Solution**:
  - Reduce `batch_size`
  - Reduce `sample_rate` or use shorter clips
  - Enable gradient checkpointing

## After Training

Once tokenizer training is complete:

1. **Validate the tokenizer**:
   ```bash
   python ../../evaluation/eval_tokenizer.py \
     --checkpoint ../../checkpoints/tokenizer/best_model.pt
   ```

2. **Use it for next phase**:
   - Speaker encoder training uses pre-extracted mel-spectrograms
   - Talker LLM uses pre-tokenized audio codes
   
3. **Export for inference**:
   ```python
   from train_tokenizer import TokenizerModel
   model = TokenizerModel.from_pretrained("checkpoints/tokenizer/best_model.pt")
   model.eval()
   # Use for inference
   ```

---

**Time to complete**: 1-2 weeks on 8x A100 GPUs
**Next Phase**: Speaker Encoder Training

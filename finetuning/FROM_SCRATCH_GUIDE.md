# Qwen3-TTS From-Scratch Training Guide

Training Qwen3-TTS from scratch requires training multiple components in sequence. This guide breaks down what's needed.

## Architecture Overview

Qwen3-TTS consists of three main components:

```
Input Audio → [Tokenizer] → Discrete Codes → [Talker LLM] → Audio Codes
                                      ↑
                                   Text + Speaker Embedding
                                      ↑
                              [Speaker Encoder]
                                      ↑
                              Reference Audio
```

### Component 1: Speech Tokenizer (Qwen3-TTS-Tokenizer-12Hz)
- **Purpose**: Encode continuous speech into discrete multi-codebook tokens
- **Architecture**: VQ-VAE with 16 codebooks
- **Input**: Waveform at 24kHz
- **Output**: 16 codebook indices (12.5 FPS / 80ms frame rate)
- **Training Data**: 1000s of hours of diverse speech
- **Compute**: ~8-16 GPU-days on A100

### Component 2: Speaker Encoder (ECAPA-TDNN)
- **Purpose**: Extract speaker embeddings from speech
- **Architecture**: ECAPA-TDNN with Res2Net blocks
- **Input**: Mel-spectrogram (128 dims, 24kHz)
- **Output**: Speaker embedding (1024-dim)
- **Training Data**: Speaker verification dataset (100k+ speakers)
- **Compute**: ~4-8 GPU-days on A100

### Component 3: Talker LLM (Transformer)
- **Purpose**: Predict acoustic codes from text + speaker embedding
- **Architecture**: Causal transformer decoder (1.7B parameters)
- **Input**: Text tokens + previous audio codes + speaker embedding
- **Output**: Next code predictions (16 codebooks)
- **Training Data**: Text-audio pairs with speakers (1000s of hours)
- **Compute**: ~100-500 GPU-days on A100

## Data Requirements

### For Tokenizer Training
- **Minimum**: 100 hours of diverse speech
- **Recommended**: 1000+ hours
- **Format**: WAV at 24kHz, mono or stereo
- **Languages**: Multilingual ideally
- **Source**: LibriSpeech, Common Voice, internal datasets

### For Speaker Encoder Training
- **Minimum**: 50k speakers with 1-10 utterances each
- **Recommended**: 100k+ speakers
- **Format**: WAV at 16-24kHz
- **Task**: Speaker classification or verification
- **Source**: VoxCeleb, TIMIT, speaker datasets

### For Talker LLM Training
- **Minimum**: 100 hours of text-audio pairs
- **Recommended**: 1000+ hours with diverse speakers
- **Format**: 
  - Audio: WAV at 24kHz
  - Text: Transcriptions (UTF-8)
  - Speaker IDs: Consistent speaker labels
- **Pairing**: Each audio must have corresponding text and speaker ID
- **Source**: LibriSpeech, Common Voice, TTS datasets

## Training Pipeline

```
Phase 1: Tokenizer Training (1-2 weeks)
├── Stage 1: VQ Learning (5-7 days)
├── Stage 2: Adversarial Refinement (2-3 days)
└── Outputs: Tokenizer checkpoint

Phase 2: Speaker Encoder Training (3-5 days)
├── Speaker Classifier Training
├── Metric Learning (optional)
└── Outputs: Speaker encoder checkpoint

Phase 3: Talker LLM Training (2-4 weeks)
├── Stage 1: Codec Prediction (10-14 days)
├── Stage 2: Fine-tuning with Instruction (5-7 days)
└── Outputs: Talker LLM checkpoint

Phase 4: Integration & Testing (3-5 days)
├── Combine components
├── Validation on benchmarks
└── Release & Demo
```

## Component-Specific Training Details

### 1. Tokenizer Training

**What it learns:**
- Compress 24kHz speech into 16 discrete codebooks
- Each codebook has 2048 vocabulary tokens
- Achieves ~12.5x compression (24kHz → 12.5 FPS, 16 codes per frame)

**Training process:**
1. Encode audio with learnable codecs
2. Apply VQ bottleneck
3. Decode back to audio
4. Minimize reconstruction loss
5. Adversarial loss to match real speech distribution

**Key hyperparameters:**
- Codebook size: 2048 per codebook
- Number of codebooks: 16
- Embedding dim: Varies per codebook
- Reconstruction loss weight: 1.0
- Adversarial loss weight: 0.1-0.5
- Learning rate: 1e-4 to 1e-3
- Batch size: 32-128 on 8 GPUs

**Validation:**
- Reconstruction PESQ score
- Codebook usage (% active codes)
- ASR WER on reconstructed audio

### 2. Speaker Encoder Training

**What it learns:**
- Extract speaker identity from speech
- 1024-dim speaker embeddings
- Speaker-invariant to content/emotion

**Training process:**
1. Load speaker classification dataset
2. Extract mel-spectrograms (128-dim, 24kHz)
3. Pass through ECAPA-TDNN
4. Classification head with softmax
5. Metric loss (ArcFace, CosFace recommended)

**Key hyperparameters:**
- Embedding dimension: 1024
- Number of speakers in batch: 32
- Samples per speaker: 2-4
- Metric: ArcFace with margin=0.3
- Learning rate: 1e-3
- Batch size: 128-256

**Validation:**
- Speaker verification EER (Equal Error Rate)
- Speaker classification accuracy

### 3. Talker LLM Training

**What it learns:**
- Predict next audio code from text + previous codes + speaker
- Multi-head prediction (16 codebooks)
- Context-aware generation

**Training process:**
1. Tokenize text to IDs
2. Encode audio to codes (using trained tokenizer)
3. Create input sequences: `[text_tokens, audio_codes_1..15, speaker_embedding]`
4. Predict next code (codec_0)
5. Auto-regressive decoding during inference

**Key hyperparameters:**
- Model size: 1.7B parameters
- Hidden dim: 4096
- Num layers: 32
- Num heads: 32
- Learning rate: 2e-5 (warmup + cosine)
- Batch size: 32-64 (effective with grad accumulation)
- Warmup steps: 10% of total
- Max seq length: 2048 tokens

**Loss function:**
```python
main_loss = CrossEntropyLoss(codec_0_predictions, codec_0_labels)
sub_talker_loss = MSELoss(codec_1..15_predictions, codec_1..15_targets)
total_loss = main_loss + 0.3 * sub_talker_loss
```

## Practical Setup

### Hardware Requirements
- **Minimum**: 4x A100 GPUs (40GB VRAM each)
- **Recommended**: 8x A100 GPUs for faster training
- **Storage**: 2-4 TB for datasets + checkpoints

### Software Stack
```bash
pip install torch==2.4.0 transformers==4.42.0 accelerate==0.33.0
pip install librosa soundfile julius encodec
pip install tensorboard wandb
```

### Data Preparation Pipeline
```python
# Step 1: Collect raw audio + text + speaker metadata
raw_data/
├── audio_files/
│   ├── speaker_001/
│   │   ├── utt001.wav
│   │   └── utt002.wav
│   └── speaker_002/
└── metadata.csv
    # Columns: audio_path, text, speaker_id, duration, language

# Step 2: Convert to training format
prepared_data/
├── train/
│   ├── audio_codes.pt  # Tokenized audio
│   ├── text_ids.pt     # Tokenized text
│   ├── speaker_ids.pt
│   └── manifest.json   # {audio_path, text, speaker_id, ...}
├── val/
└── test/
```

### Distributed Training
Use Accelerate for multi-GPU training:

```bash
accelerate config  # Configure your setup
accelerate launch train_tokenizer.py --config config.yaml
accelerate launch train_speaker_encoder.py --config config.yaml
accelerate launch train_talker_lm.py --config config.yaml
```

## Component Training Scripts

What we'll create:

1. **train_tokenizer.py** - Train VQ codec from scratch
2. **train_speaker_encoder.py** - Train speaker embedding extractor
3. **train_talker_lm.py** - Train the main LLM component
4. **data_processor.py** - Prepare datasets
5. **combined_train.py** - Full pipeline coordinator

## Realistic Timeline

| Phase | Duration | GPU-Days | Cost (A100) |
|-------|----------|----------|------------|
| Data Collection | 2-4 weeks | 0 | $0 |
| Tokenizer Training | 1-2 weeks | 100-150 | $20-30K |
| Speaker Encoder | 3-5 days | 50-80 | $10-15K |
| Talker LLM | 2-4 weeks | 300-500 | $60-100K |
| **Total** | **1-3 months** | **450-730** | **$90-150K** |

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Data quality varies | Implement data filtering, QA pipeline |
| Long training times | Use distributed training, gradient accumulation |
| OOM errors | Reduce batch size, use gradient checkpointing |
| Mode collapse (Tokenizer) | Use diverse data, increase codebook size |
| Speaker leakage | Proper data split by speaker, genre |
| Generation quality low | Train longer, use better data, tune loss weights |

## References

- **Tokenizer Architecture**: Based on SpeechTokenizer, EnCodec
- **Speaker Encoder**: ECAPA-TDNN from SpeakerNet
- **LLM Architecture**: Based on Qwen LLM
- **Paper**: https://arxiv.org/abs/2601.15621

## Next Steps

1. Collect and prepare training data
2. Create tokenizer training pipeline
3. Create speaker encoder training pipeline
4. Create talker LLM training pipeline
5. Integration and testing
6. Evaluation on benchmarks
7. Release and documentation

This is a significant undertaking requiring months of work and substantial compute resources. Consider whether fine-tuning or using the released models better serves your use case first.

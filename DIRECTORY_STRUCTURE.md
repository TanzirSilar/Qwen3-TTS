# Qwen3-TTS Complete Directory Structure

## Overview

The project is now organized with both fine-tuning and from-scratch training capabilities.

## Full Directory Tree

```
/speech/tanzir/Qwen3-TTS/
│
├── qwen_tts/                              # Main package (editable install)
│   ├── __init__.py
│   ├── __main__.py
│   ├── core/                              # Core models
│   │   ├── models/
│   │   │   ├── modeling_qwen3_tts.py
│   │   │   ├── configuration_qwen3_tts.py
│   │   │   └── processing_qwen3_tts.py
│   │   ├── tokenizer_12hz/
│   │   ├── tokenizer_25hz/
│   │   └── __init__.py
│   ├── inference/
│   │   ├── qwen3_tts_model.py
│   │   ├── qwen3_tts_tokenizer.py
│   │   └── __init__.py
│   └── cli/
│
├── finetuning/                            # Fine-tuning Scripts ✅ DONE
│   ├── train_qwen3_tts.py                # Main training orchestrator
│   ├── train_pipeline.py                 # Unified pipeline interface
│   ├── train_config.yaml                 # Configuration
│   ├── example_training.sh               # Example workflow
│   ├── dataset.py                        # Dataset utilities
│   ├── prepare_data.py                   # Data preparation
│   ├── sft_12hz.py                       # Original SFT script
│   ├── TRAINING_GUIDE.md                 # User guide
│   ├── TRAINING_COMPONENTS.md            # Technical details
│   └── README.md                         # Fine-tuning README
│
├── training/                              # From-Scratch Training 🚀 NEW
│   │
│   ├── 0_data_preparation/               # Phase 0: Data Prep
│   │   ├── prepare_tokenizer_data.py     # (Placeholder)
│   │   ├── prepare_speaker_data.py       # (Placeholder)
│   │   ├── prepare_talker_data.py        # (Placeholder)
│   │   ├── utils.py                      # (Placeholder)
│   │   └── README.md                     # Detailed guide
│   │
│   ├── 1_tokenizer_training/             # Phase 1: Tokenizer
│   │   ├── train_tokenizer.py            # (Placeholder)
│   │   ├── tokenizer_config.yaml         # (Placeholder)
│   │   ├── data_loader.py                # (Placeholder)
│   │   ├── model.py                      # (Placeholder)
│   │   └── README.md                     # Detailed guide
│   │
│   ├── 2_speaker_encoder_training/       # Phase 2: Speaker Encoder
│   │   ├── train_speaker_encoder.py      # (Placeholder)
│   │   ├── speaker_config.yaml           # (Placeholder)
│   │   ├── data_loader.py                # (Placeholder)
│   │   ├── model.py                      # (Placeholder)
│   │   └── README.md                     # Detailed guide
│   │
│   ├── 3_talker_lm_training/             # Phase 3: Talker LLM
│   │   ├── train_talker_lm.py            # (Placeholder)
│   │   ├── talker_config.yaml            # (Placeholder)
│   │   ├── data_loader.py                # (Placeholder)
│   │   ├── model.py                      # (Placeholder)
│   │   └── README.md                     # Detailed guide
│   │
│   ├── combined_training.py              # (Placeholder) - Orchestrate all phases
│   ├── from_scratch_config.yaml          # Master configuration
│   ├── requirements.txt                  # Training dependencies
│   ├── README.md                         # Training overview
│   └── FROM_SCRATCH_GUIDE.md             # Complete training guide
│
├── configs/                               # Centralized Configurations 🆕
│   ├── tokenizer.yaml                    # Tokenizer hyperparameters
│   ├── speaker_encoder.yaml              # Speaker encoder config
│   └── talker_lm.yaml                    # Talker LLM config
│
├── datasets/                              # Data Storage 🆕
│   │
│   ├── tokenizer_data/
│   │   ├── raw_audio/                    # Raw WAV files
│   │   ├── manifests/                    # JSONL manifests
│   │   │   ├── train_manifest.jsonl      # (Generated during prep)
│   │   │   └── val_manifest.jsonl        # (Generated during prep)
│   │   └── processed/                    # Pre-computed features
│   │
│   ├── speaker_encoder_data/
│   │   ├── raw_audio/                    # By speaker/{speaker_id}/
│   │   ├── manifests/
│   │   │   ├── train_manifest.jsonl      # (Generated during prep)
│   │   │   └── val_manifest.jsonl        # (Generated during prep)
│   │   └── processed/
│   │
│   └── talker_lm_data/
│       ├── raw_audio/                    # WAV files
│       ├── transcriptions/               # Text files (one per audio)
│       ├── manifests/
│       │   ├── train_manifest.jsonl      # (Generated during prep)
│       │   ├── val_manifest.jsonl        # (Generated during prep)
│       │   └── test_manifest.jsonl       # (Generated during prep)
│       └── processed/
│           ├── speaker_embeddings/       # Pre-computed embeddings
│           └── audio_codes/              # Pre-tokenized audio
│
├── checkpoints/                           # Training Outputs 🆕
│   │
│   ├── tokenizer/
│   │   ├── checkpoint-epoch-0.pt
│   │   ├── checkpoint-epoch-1.pt
│   │   ├── best_model.pt
│   │   └── latest.pt
│   │
│   ├── speaker_encoder/
│   │   ├── checkpoint-epoch-0.pt
│   │   ├── best_model.pt
│   │   └── latest.pt
│   │
│   └── talker_lm/
│       ├── checkpoint-step-500/
│       ├── checkpoint-step-1000/
│       ├── best_model/
│       └── latest/
│
├── logs/                                  # Training Logs 🆕
│   │
│   ├── tokenizer/
│   │   ├── training.log
│   │   └── runs/                         # TensorBoard logs
│   │       └── events.out.tfevents.*
│   │
│   ├── speaker_encoder/
│   │   ├── training.log
│   │   └── runs/
│   │
│   └── talker_lm/
│       ├── training.log
│       └── runs/
│
├── evaluation/                            # Evaluation Scripts 🆕
│   ├── eval_tokenizer.py                 # (Placeholder) - Reconstruction quality
│   ├── eval_speaker_encoder.py           # (Placeholder) - Speaker verification
│   ├── eval_talker_lm.py                 # (Placeholder) - Generation quality
│   ├── benchmarks/                       # Benchmark datasets
│   │   ├── seed_tts/
│   │   ├── multilingual_tts/
│   │   └── instruct_tts/
│   └── metrics/
│       ├── pesq.py                       # (Placeholder)
│       ├── wer.py                        # (Placeholder)
│       └── speaker_similarity.py         # (Placeholder)
│
├── examples/                              # Example Scripts
│   ├── test_model_12hz_base.py
│   ├── test_tokenizer_12hz.py
│   └── ...
│
├── DIRECTORY_STRUCTURE.md                # This file 🆕
├── README.md                             # Main README
├── STRUCTURE.md                          # (Legacy)
├── pyproject.toml
└── .git/

```

## Status Summary

| Component | Status | Location |
|-----------|--------|----------|
| **Fine-tuning** | ✅ COMPLETE | `finetuning/` |
| **From-Scratch Training Setup** | ✅ STRUCTURE READY | `training/` |
| **Configurations** | ✅ TEMPLATES CREATED | `configs/` |
| **Data Directories** | ✅ STRUCTURE READY | `datasets/` |
| **Training Scripts** | 🟡 PLACEHOLDER (ready to implement) | `training/*/` |
| **Checkpoints** | 🟡 DIRECTORIES READY | `checkpoints/` |
| **Logging** | 🟡 DIRECTORIES READY | `logs/` |
| **Evaluation** | 🟡 PLACEHOLDER (ready to implement) | `evaluation/` |

## File Counts

```
Created in this session:
- 10 README files for guidance
- 3 Configuration YAML files (tokenizer, speaker_encoder, talker_lm)
- 1 Master configuration file
- 1 Directory structure documentation
- 11+ Python placeholder files (ready for implementation)
- 10+ Training requirement files

Total structure directories: 25+
```

## What Each Phase Does

### Phase 0: Data Preparation
```
Raw Audio/Text Files
    ↓
prepare_*_data.py scripts
    ↓
JSONL Manifests with metadata
    ↓
Training data ready
```

### Phase 1: Tokenizer Training
```
Audio Manifests
    ↓
train_tokenizer.py
    ↓
VQ-VAE Codec
    ↓
checkpoint-epoch-N.pt, best_model.pt
```

### Phase 2: Speaker Encoder Training
```
Speaker Audio Manifests
    ↓
train_speaker_encoder.py
    ↓
ECAPA-TDNN Embeddings
    ↓
checkpoint-epoch-N.pt, best_model.pt
```

### Phase 3: Talker LLM Training
```
Text-Audio Manifests +
Pre-computed Embeddings +
Pre-tokenized Codes
    ↓
train_talker_lm.py
    ↓
Transformer LLM
    ↓
checkpoint-step-N/, best_model/
```

## Next Steps

### 1. Install Dependencies
```bash
cd /speech/tanzir/Qwen3-TTS
pip install -e .
pip install -r training/requirements.txt
```

### 2. Prepare Your Data
Place files in `datasets/*/raw_audio/` following the directory structure described in `training/0_data_preparation/README.md`

### 3. Run Data Preparation
```bash
cd training/0_data_preparation
python prepare_tokenizer_data.py --audio_dir ../../datasets/tokenizer_data/raw_audio
```

### 4. Configure Training
Edit configuration files in `configs/`:
- `tokenizer.yaml` - Tokenizer parameters
- `speaker_encoder.yaml` - Speaker encoder parameters
- `talker_lm.yaml` - Talker LLM parameters

### 5. Start Training
```bash
cd training
python 1_tokenizer_training/train_tokenizer.py --config ../configs/tokenizer.yaml
```

## Documentation Map

| Task | Documentation |
|------|---|
| Quick overview | `training/README.md` |
| Complete guide | `training/FROM_SCRATCH_GUIDE.md` |
| Data preparation | `training/0_data_preparation/README.md` |
| Tokenizer training | `training/1_tokenizer_training/README.md` |
| Speaker encoder | `training/2_speaker_encoder_training/README.md` |
| Talker LLM | `training/3_talker_lm_training/README.md` |
| Fine-tuning | `finetuning/TRAINING_GUIDE.md` |
| Technical details | `finetuning/TRAINING_COMPONENTS.md` |

## Hardware Requirements

| Component | Min GPUs | Recommended | Time |
|-----------|----------|-------------|------|
| **Tokenizer** | 4x A100 | 8x A100 | 1-2 weeks |
| **Speaker Encoder** | 2x A100 | 4x A100 | 3-5 days |
| **Talker LLM** | 4x A100 | 8x A100 | 2-4 weeks |
| **Total** | 12 GPU-days | 20 GPU-days | 1-3 months |

## Common Commands

```bash
# View training status
tail -f logs/tokenizer/training.log

# Monitor with TensorBoard
tensorboard --logdir logs/

# Check dataset statistics
python training/0_data_preparation/utils.py validate_manifest \
  datasets/tokenizer_data/manifests/train_manifest.jsonl

# Resume training from checkpoint
python training/1_tokenizer_training/train_tokenizer.py \
  --config configs/tokenizer.yaml \
  --resume_from checkpoints/tokenizer/checkpoint-epoch-5.pt
```

## Notes

- Each phase depends on the previous one's output
- Phases 1 & 2 (Tokenizer & Speaker Encoder) can train in parallel
- Phase 3 (Talker LLM) depends on both Phase 1 & 2 outputs
- Use Accelerate for multi-GPU training (recommended)
- Monitor GPU memory and adjust batch sizes if needed
- Save your configurations before training starts

---

**Created**: 2026-11-23
**Location**: /speech/tanzir/Qwen3-TTS/

**Next Action**: Read `training/README.md` to begin setup

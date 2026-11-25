# Qwen3-TTS From-Scratch Training

Complete training pipeline for training Qwen3-TTS models from scratch. This directory contains everything needed to train all three components of the system.

## 📋 Overview

Qwen3-TTS consists of three trainable components that must be trained sequentially:

```
Phase 0: Data Preparation
         ↓
Phase 1: Tokenizer Training (VQ-VAE codec)
         ↓
Phase 2: Speaker Encoder Training (ECAPA-TDNN)
         ↓
Phase 3: Talker LLM Training (Transformer)
         ↓
Final: Integration & Testing
```

## 📁 Directory Structure

```
training/
├── 0_data_preparation/      # Data preparation scripts
│   ├── prepare_tokenizer_data.py
│   ├── prepare_speaker_data.py
│   ├── prepare_talker_data.py
│   └── utils.py
│
├── 1_tokenizer_training/    # VQ-VAE tokenizer training
│   ├── train_tokenizer.py
│   ├── tokenizer_config.yaml
│   └── data_loader.py
│
├── 2_speaker_encoder_training/  # Speaker embedding extraction
│   ├── train_speaker_encoder.py
│   ├── speaker_config.yaml
│   └── data_loader.py
│
├── 3_talker_lm_training/    # Main LLM training
│   ├── train_talker_lm.py
│   ├── talker_config.yaml
│   ├── model.py
│   └── data_loader.py
│
├── combined_training.py     # Orchestrate all phases
├── requirements.txt         # Training dependencies
└── FROM_SCRATCH_GUIDE.md    # Detailed guide
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /speech/tanzir/Qwen3-TTS
pip install -e .
pip install -r training/requirements.txt
```

### 2. Prepare Your Data

Place your data in:
- **Tokenizer data**: `datasets/tokenizer_data/raw_audio/*.wav`
- **Speaker data**: `datasets/speaker_encoder_data/raw_audio/{speaker_id}/*.wav`
- **Talker data**: `datasets/talker_lm_data/raw_audio/*.wav` + `datasets/talker_lm_data/transcriptions/{id}.txt`

### 3. Run Data Preparation
```bash
cd training/0_data_preparation
python prepare_tokenizer_data.py --audio_dir ../../datasets/tokenizer_data/raw_audio
python prepare_speaker_data.py --audio_dir ../../datasets/speaker_encoder_data/raw_audio
python prepare_talker_data.py --audio_dir ../../datasets/talker_lm_data/raw_audio
```

### 4. Run Training

**Option A: Train each component separately**
```bash
cd /speech/tanzir/Qwen3-TTS/training

# Phase 1
python 1_tokenizer_training/train_tokenizer.py --config configs/tokenizer.yaml

# Phase 2
python 2_speaker_encoder_training/train_speaker_encoder.py --config configs/speaker_encoder.yaml

# Phase 3
python 3_talker_lm_training/train_talker_lm.py --config configs/talker_lm.yaml
```

**Option B: Orchestrate all phases**
```bash
python combined_training.py --config from_scratch_config.yaml --mode full
```

## 📊 Data Requirements

### Minimum Setup
| Component | Min Data | Recommended | Duration |
|-----------|----------|-------------|----------|
| Tokenizer | 10 hours | 100+ hours | 1-2 weeks |
| Speaker Encoder | 1k speakers | 10k+ speakers | 3-5 days |
| Talker LLM | 10 hours | 100+ hours | 1-2 weeks |

### Data Format

**Tokenizer**: Raw audio files
```
datasets/tokenizer_data/raw_audio/
├── sample_001.wav
├── sample_002.wav
└── ...
```

**Speaker Encoder**: Audio grouped by speaker
```
datasets/speaker_encoder_data/raw_audio/
├── speaker_001/
│   ├── utt_001.wav
│   └── utt_002.wav
├── speaker_002/
│   └── utt_001.wav
└── ...
```

**Talker LLM**: Paired audio + transcriptions
```
datasets/talker_lm_data/
├── raw_audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── transcriptions/
    ├── sample_001.txt  (content: "text here")
    ├── sample_002.txt
    └── ...
```

## ⚙️ Configuration Files

Each phase has its own configuration file in `configs/`:

- `tokenizer.yaml` - Tokenizer hyperparameters
- `speaker_encoder.yaml` - Speaker encoder config
- `talker_lm.yaml` - Talker LLM config

Edit these before training to customize:
- Learning rates
- Batch sizes
- Number of epochs
- Architecture parameters
- Loss weights

## 🔍 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Logs
```bash
tail -f logs/tokenizer/training.log
tail -f logs/speaker_encoder/training.log
tail -f logs/talker_lm/training.log
```

## 📈 Training Progress

Expected timelines on 8x A100 GPUs:

| Phase | Duration | GPU-Days | Estimated Cost |
|-------|----------|----------|---|
| Tokenizer | 1-2 weeks | 100-150 | $20-30K |
| Speaker Encoder | 3-5 days | 50-80 | $10-15K |
| Talker LLM | 2-4 weeks | 300-500 | $60-100K |
| **Total** | **1-3 months** | **450-730** | **$90-150K** |

(Costs based on on-demand A100 at ~$3.06/hour)

## 🧪 Evaluation

After training, evaluate each component:

```bash
python ../evaluation/eval_tokenizer.py --checkpoint checkpoints/tokenizer/best_model.pt
python ../evaluation/eval_speaker_encoder.py --checkpoint checkpoints/speaker_encoder/best_model.pt
python ../evaluation/eval_talker_lm.py --checkpoint checkpoints/talker_lm/best_model/
```

## 📚 Detailed Guides

- [FROM_SCRATCH_GUIDE.md](FROM_SCRATCH_GUIDE.md) - Complete training guide
- [0_data_preparation/README.md](0_data_preparation/README.md) - Data prep details
- [1_tokenizer_training/README.md](1_tokenizer_training/README.md) - Tokenizer training
- [2_speaker_encoder_training/README.md](2_speaker_encoder_training/README.md) - Speaker encoder training
- [3_talker_lm_training/README.md](3_talker_lm_training/README.md) - Talker LLM training

## 🛠️ Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`

### Slow Training
- Use more GPUs with Accelerate
- Reduce number of validation steps
- Increase learning rate carefully

### Poor Quality
- Train longer (more epochs)
- Use more diverse training data
- Adjust loss weights in config
- Check data quality

## 📖 References

- **Paper**: https://arxiv.org/abs/2601.15621
- **Original Repo**: https://github.com/QwenLM/Qwen3-TTS
- **Components**:
  - Tokenizer: Based on SpeechTokenizer, EnCodec
  - Speaker Encoder: ECAPA-TDNN from SpeakerNet
  - Talker LLM: Based on Qwen LLM architecture

## 📝 Notes

- Each phase depends on the previous one's output
- You can interrupt and resume training (checkpoints auto-save)
- Use different GPUs/nodes for each phase if needed
- Monitor GPU memory and adjust batch sizes accordingly
- Keep training logs for debugging and analysis

---

**Next Steps**: 
1. Read [FROM_SCRATCH_GUIDE.md](FROM_SCRATCH_GUIDE.md) for detailed information
2. Prepare your dataset
3. Configure training parameters in `configs/`
4. Run data preparation scripts
5. Start Phase 1 training

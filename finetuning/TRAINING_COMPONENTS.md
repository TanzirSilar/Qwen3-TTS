# Training Components Overview

## New Training Infrastructure for Qwen3-TTS

This document describes the complete training infrastructure added to the Qwen3-TTS project.

## Components

### 1. `train_qwen3_tts.py` - Main Training Script

**Purpose**: Comprehensive training loop with production-ready features.

**Key Features**:
- Full training orchestration using Accelerate
- Validation support with best model tracking
- Learning rate scheduling (warmup + cosine annealing)
- Per-step and per-epoch checkpointing
- Detailed logging to file and console
- Gradient accumulation for larger effective batch sizes
- BF16 mixed precision training

**Classes**:
- `TrainingMetrics`: Tracks loss metrics across training
- `Qwen3TTSTrainer`: Main orchestrator class

**Example Usage**:
```python
from train_qwen3_tts import Qwen3TTSTrainer

trainer = Qwen3TTSTrainer(
    init_model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    output_dir="output",
    train_jsonl="train_with_codes.jsonl",
    val_jsonl="val_with_codes.jsonl",
    batch_size=2,
    lr=2e-5,
    num_epochs=3,
    speaker_name="my_speaker",
)
trainer.train()
```

**Command Line**:
```bash
python train_qwen3_tts.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --val_jsonl val_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name my_speaker
```

### 2. `train_pipeline.py` - Unified Pipeline Interface

**Purpose**: Single entry point for all training operations (prepare, train, infer).

**Key Features**:
- Loads configuration from YAML
- Supports multiple modes: prepare, train, infer, full
- Encapsulates data preparation, training, and inference
- Error handling and logging

**Class**:
- `Qwen3TTSPipeline`: Orchestrates all pipeline stages

**Supported Modes**:

1. **prepare**: Extract audio codes from raw data
   ```bash
   python train_pipeline.py --mode prepare --raw_data train_raw.jsonl
   ```

2. **train**: Train with prepared data
   ```bash
   python train_pipeline.py --mode train --train_data train_with_codes.jsonl
   ```

3. **infer**: Generate audio with trained model
   ```bash
   python train_pipeline.py --mode infer \
     --checkpoint output/best_model \
     --text "Your text here"
   ```

4. **full**: Complete pipeline in one command
   ```bash
   python train_pipeline.py --mode full \
     --raw_data train_raw.jsonl \
     --test_text "Test text"
   ```

### 3. `train_config.yaml` - Configuration File

**Purpose**: Centralized configuration for all training hyperparameters.

**Sections**:
- **Model & Data Paths**: Model location, data files
- **Training Parameters**: Batch size, learning rate, epochs, speaker name
- **Optimization**: Learning rate scheduling, gradient clipping
- **Checkpointing**: Save and evaluation frequency
- **Advanced Settings**: Precision, attention implementation, seed

**Example**:
```yaml
init_model_path: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
batch_size: 2
learning_rate: 2e-5
num_epochs: 3
speaker_name: "speaker_test"
```

### 4. `example_training.sh` - Complete Example Script

**Purpose**: Demonstrates a full training workflow from raw data to inference.

**What It Does**:
1. ✅ Validates input data exists
2. ✅ Prepares training data (extract audio codes)
3. ✅ Prepares validation data (if available)
4. ✅ Updates configuration for the project
5. ✅ Runs training
6. ✅ Tests inference on trained model

**Usage**:
```bash
chmod +x example_training.sh
./example_training.sh
```

### 5. `TRAINING_GUIDE.md` - User Documentation

**Contents**:
- Quick start guide
- Detailed usage for each mode
- Configuration reference
- Inference examples
- Troubleshooting
- Best practices

## Architecture

```
User Input
    ↓
train_pipeline.py (Entry Point)
    ├── Mode: prepare
    │   └── Calls prepare_data.py
    │       └── Tokenizer.encode(audio)
    │
    ├── Mode: train
    │   └── train_qwen3_tts.py::Qwen3TTSTrainer
    │       ├── Load Model
    │       ├── Load Data
    │       ├── Setup Optimizer & Scheduler
    │       ├── Training Loop
    │       │   ├── Forward Pass
    │       │   ├── Loss Computation
    │       │   ├── Backward Pass
    │       │   ├── Gradient Clipping
    │       │   ├── Optimizer Step
    │       │   └── Logging
    │       └── Save Checkpoints
    │
    ├── Mode: infer
    │   └── Load Model & Generate Audio
    │
    └── Mode: full
        └── All of the above in sequence
```

## Training Loop Details

### Forward Pass
1. Extract speaker embedding from reference audio
2. Embed text and audio codes
3. Add all embeddings together
4. Pass through transformer
5. Compute main loss + sub_talker loss

### Backward Pass
1. Compute combined loss: `loss = main_loss + 0.3 * sub_talker_loss`
2. Accumulate gradients (every N steps)
3. Clip gradients to max_norm
4. Update optimizer
5. Update learning rate scheduler

### Checkpointing
- **Per-epoch**: Always saved
- **Per-step**: Saved every `save_steps` steps (if > 0)
- **Best model**: Tracked via validation loss if available
- **Config update**: Speaker ID and embedding saved in checkpoint config

## Key Improvements Over Original

| Feature | Original | New |
|---------|----------|-----|
| Learning Rate Scheduling | ❌ | ✅ Warmup + Cosine Annealing |
| Validation Support | ❌ | ✅ Optional validation set |
| Per-step Checkpointing | ❌ | ✅ Configurable frequency |
| Best Model Tracking | ❌ | ✅ Via validation loss |
| Detailed Metrics | ⚠️ Limited | ✅ Extensive logging |
| Gradient Accumulation | ❌ | ✅ For larger effective batch |
| Configuration File | ❌ | ✅ YAML-based |
| Pipeline Interface | ❌ | ✅ Unified entry point |
| Data Prep Integration | Separate | ✅ Integrated |

## File Structure After Training

```
output/
├── checkpoint-epoch-0/
│   ├── config.json              # Updated config with speaker_id
│   ├── model.safetensors        # Model weights
│   ├── generation_config.json
│   ├── preprocessor_config.json
│   └── ...
├── checkpoint-epoch-1/
├── checkpoint-epoch-2/
├── checkpoint-step-500/         # Per-step checkpoints
├── checkpoint-step-1000/
├── best_model/                  # Symlink to best checkpoint
├── logs/
│   └── [tensorboard files]
└── training.log
```

## Important Design Decisions

1. **Editable Install Support**: All imports use `qwen_tts` module directly (works with pip install -e .)

2. **Accelerate Framework**: Uses Accelerate for distributed training, mixed precision, and gradient accumulation

3. **BF16 Mixed Precision**: Faster training + lower memory without quality loss (better than FP16 for LLMs)

4. **Gradient Accumulation**: Allows larger effective batches without OOM

5. **Learning Rate Scheduling**: Warmup prevents training instability; cosine annealing provides smooth decay

6. **Validation is Optional**: Users can train without validation; best model found via step-based checkpointing otherwise

7. **Configuration-Driven**: YAML config enables reproducible experiments

8. **Modular Design**: Each component is reusable:
   - `Qwen3TTSTrainer` can be imported for custom training loops
   - `Qwen3TTSPipeline` provides CLI interface
   - Components work independently

## Running Training

### Minimal Example
```bash
cd /speech/tanzir/Qwen3-TTS/finetuning
python train_qwen3_tts.py --train_jsonl train_with_codes.jsonl
```

### Full-Featured Example
```bash
python train_pipeline.py \
  --config train_config.yaml \
  --mode full \
  --raw_data train_raw.jsonl \
  --val_data val_raw.jsonl \
  --test_text "She said she would be here by noon."
```

### Custom Training Loop
```python
from train_qwen3_tts import Qwen3TTSTrainer

trainer = Qwen3TTSTrainer(...)
trainer.train()

# Access results
print(f"Best validation loss: {trainer.best_val_loss}")
print(f"Total training steps: {trainer.global_step}")
```

## Extending the Training System

### Adding Custom Loss Terms
Modify `_training_step()` in `Qwen3TTSTrainer`:
```python
def _training_step(self, batch):
    # ... existing code ...
    main_loss = outputs.loss
    sub_talker_loss = ...
    custom_loss = ...  # Your custom loss
    
    total_loss = main_loss + 0.3 * sub_talker_loss + 0.1 * custom_loss
    return total_loss, main_loss, sub_talker_loss
```

### Adding Custom Metrics
Create subclass of `TrainingMetrics`:
```python
class CustomMetrics(TrainingMetrics):
    def __init__(self):
        super().__init__()
        self.custom_values = []
    
    def update_custom(self, value):
        self.custom_values.append(value)
```

### Adding Custom Callbacks
Extend `Qwen3TTSTrainer`:
```python
class CustomTrainer(Qwen3TTSTrainer):
    def _training_step(self, batch):
        loss, main, sub = super()._training_step(batch)
        self._on_step_end(loss)  # Your callback
        return loss, main, sub
```

## References

- Original Model: https://github.com/QwenLM/Qwen3-TTS
- Accelerate: https://huggingface.co/docs/accelerate/
- Qwen Models: https://huggingface.co/Qwen

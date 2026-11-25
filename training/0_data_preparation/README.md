# Phase 0: Data Preparation

Convert raw audio and metadata into training-ready manifests for each training phase.

## Overview

This phase creates JSONL manifest files that point to your data with necessary metadata. The manifests are used by the data loaders in subsequent phases.

## Scripts

### 1. `prepare_tokenizer_data.py`
Prepares data for tokenizer training.

**Input**: Raw WAV files in a directory
**Output**: `datasets/tokenizer_data/manifests/{train,val}_manifest.jsonl`

```bash
python prepare_tokenizer_data.py \
  --audio_dir ../../datasets/tokenizer_data/raw_audio \
  --output_dir ../../datasets/tokenizer_data/manifests \
  --train_split 0.9 \
  --sample_rate 24000
```

**Output manifest format**:
```jsonl
{"audio_path": "/path/to/audio.wav", "duration": 5.23, "language": "en"}
{"audio_path": "/path/to/audio2.wav", "duration": 3.15, "language": "en"}
```

### 2. `prepare_speaker_data.py`
Prepares data for speaker encoder training.

**Input**: WAV files organized by speaker ID
**Directory structure**:
```
raw_audio/
├── speaker_001/
│   ├── utt_001.wav
│   ├── utt_002.wav
│   └── ...
├── speaker_002/
│   └── ...
└── ...
```

**Output**: `datasets/speaker_encoder_data/manifests/{train,val}_manifest.jsonl`

```bash
python prepare_speaker_data.py \
  --audio_dir ../../datasets/speaker_encoder_data/raw_audio \
  --output_dir ../../datasets/speaker_encoder_data/manifests \
  --train_split 0.8 \
  --min_speakers 1000
```

**Output manifest format**:
```jsonl
{"audio_path": "/path/to/speaker_001/utt_001.wav", "speaker_id": "speaker_001", "duration": 4.5}
{"audio_path": "/path/to/speaker_002/utt_001.wav", "speaker_id": "speaker_002", "duration": 3.2}
```

### 3. `prepare_talker_data.py`
Prepares data for talker LLM training.

**Input**: 
- Audio files in `raw_audio/`
- Text transcriptions in `transcriptions/`

**Directory structure**:
```
raw_audio/
├── sample_001.wav
├── sample_002.wav
└── ...

transcriptions/
├── sample_001.txt  (content: "text here")
├── sample_002.txt  (content: "more text")
└── ...
```

**Output**: `datasets/talker_lm_data/manifests/{train,val,test}_manifest.jsonl`

```bash
python prepare_talker_data.py \
  --audio_dir ../../datasets/talker_lm_data/raw_audio \
  --text_dir ../../datasets/talker_lm_data/transcriptions \
  --output_dir ../../datasets/talker_lm_data/manifests \
  --train_split 0.8 \
  --val_split 0.1 \
  --speaker_id_file speaker_mapping.txt
```

**Output manifest format**:
```jsonl
{"audio_path": "/path/to/sample_001.wav", "text": "transcription here", "speaker_id": "speaker_001", "duration": 5.0}
{"audio_path": "/path/to/sample_002.wav", "text": "more text", "speaker_id": "speaker_002", "duration": 3.5}
```

## Utilities

### `utils.py`
Common utility functions:
- `load_audio(path, sr)` - Load WAV files
- `get_duration(path)` - Get audio duration
- `validate_manifest()` - Check manifest integrity
- `split_manifest()` - Split into train/val/test

## Data Quality Checks

Before and after preparation, run quality checks:

```python
from utils import validate_manifest, check_data_quality

# Validate manifest files
validate_manifest("path/to/manifest.jsonl")

# Check dataset statistics
check_data_quality("datasets/tokenizer_data/manifests/train_manifest.jsonl")
```

Output:
```
Dataset: tokenizer_data
├── Total samples: 1000
├── Total duration: 50 hours
├── Avg duration: 180 seconds
├── Min/Max duration: 10s / 600s
├── Sample rate: 24000 Hz
└── Languages: en, zh, ja
```

## Tips

### For Tokenizer Data
- Use diverse speakers and acoustic conditions
- Include different languages if training multilingual
- Minimum: 10 hours, Recommended: 100+ hours
- Mix indoor/outdoor, clean/noisy audio

### For Speaker Encoder Data
- Need 1000+ different speakers
- 1-10 utterances per speaker
- Speaker IDs must be consistent
- Remove speakers with very few samples

### For Talker LLM Data
- Text must match audio content (high quality transcriptions)
- Include diverse speakers, emotions, speaking styles
- Balance across languages and domains
- Use same audio/text pairs if from public datasets

## Common Issues

**Problem**: "Audio file not found"
```
Solution: Check paths in manifest, ensure audio_dir is correct
```

**Problem**: "Duration mismatch"
```
Solution: Audio file might be corrupted. Re-encode to WAV 24kHz
```

**Problem**: "No speakers found"
```
Solution: Check directory structure matches requirements
```

**Problem**: "Text doesn't match audio"
```
Solution: Verify transcriptions are aligned with audio files
```

## Output Structure After Preparation

```
datasets/
├── tokenizer_data/
│   └── manifests/
│       ├── train_manifest.jsonl    (900 samples)
│       └── val_manifest.jsonl      (100 samples)
│
├── speaker_encoder_data/
│   └── manifests/
│       ├── train_manifest.jsonl    (1000 speakers, 5000 samples)
│       └── val_manifest.jsonl      (200 speakers, 1000 samples)
│
└── talker_lm_data/
    └── manifests/
        ├── train_manifest.jsonl    (800 samples)
        ├── val_manifest.jsonl      (100 samples)
        └── test_manifest.jsonl     (100 samples)
```

## Next Steps

After preparation:
1. Verify manifests are created and valid
2. Check dataset statistics look reasonable
3. Proceed to Phase 1: Tokenizer Training

---

**Time to complete**: 1-2 hours (depending on data size and disk I/O)

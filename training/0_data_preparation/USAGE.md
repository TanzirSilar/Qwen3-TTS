# Data Preparation Scripts - Usage Guide

Complete guide for preparing training data for each Qwen3-TTS training component.

## Quick Start

### 1. For Tokenizer Training

```bash
python prepare_tokenizer_data.py \
  --audio_dir /path/to/raw/audio \
  --output_dir ../../../datasets/tokenizer_data/manifests \
  --train_split 0.9 \
  --language en
```

**What it does:**
- Finds all audio files in a directory
- Validates duration (default: 1-20 seconds)
- Creates train/val manifests
- Output: `train_manifest.jsonl`, `val_manifest.jsonl`

### 2. For Speaker Encoder Training

```bash
python prepare_speaker_data.py \
  --audio_dir /path/to/speakers \
  --output_dir ../../../datasets/speaker_encoder_data/manifests \
  --min_speakers 1000
```

**Expected directory structure:**
```
speakers/
├── speaker_001/
│   ├── utt_001.wav
│   ├── utt_002.wav
│   └── ...
├── speaker_002/
│   ├── utt_001.wav
│   └── ...
└── ...
```

**What it does:**
- Finds speakers (subdirectories)
- Validates audio files for each speaker
- Splits by speakers (not utterances) for better generalization
- Output: `train_manifest.jsonl`, `val_manifest.jsonl`

### 3. For Talker LLM Training

```bash
python prepare_talker_data.py \
  --audio_dir /path/to/audio \
  --text_dir /path/to/text \
  --output_dir ../../../datasets/talker_lm_data/manifests
```

**Expected directory structure:**
```
audio/
├── sample_001.wav
├── sample_002.wav
└── ...

text/
├── sample_001.txt
├── sample_002.txt
└── ...
```

**Text file format:** Plain text, one file per audio

**What it does:**
- Pairs audio with text files
- Validates both audio and text
- Splits into train/val/test manifests
- Output: `train_manifest.jsonl`, `val_manifest.jsonl`, `test_manifest.jsonl`

---

## Detailed Options

### Tokenizer Data Preparation

```bash
python prepare_tokenizer_data.py \
  --audio_dir ./raw_audio \                    # Required: audio directory
  --output_dir ./manifests \                   # Optional: output directory
  --train_split 0.9 \                          # Optional: train proportion
  --val_split 0.1 \                            # Optional: val proportion
  --sample_rate 24000 \                        # Optional: target sample rate
  --min_duration 1.0 \                         # Optional: minimum duration
  --max_duration 20.0 \                        # Optional: maximum duration
  --no_shuffle \                               # Optional: disable shuffling
  --seed 42 \                                  # Optional: random seed
  --language en                                # Optional: language code
```

### Speaker Encoder Data Preparation

```bash
python prepare_speaker_data.py \
  --audio_dir ./raw_audio \                    # Required: speaker directory
  --output_dir ./manifests \                   # Optional: output directory
  --train_split 0.8 \                          # Optional: speaker train proportion
  --val_split 0.2 \                            # Optional: speaker val proportion
  --sample_rate 24000 \                        # Optional: target sample rate
  --min_duration 1.0 \                         # Optional: minimum duration
  --max_duration 10.0 \                        # Optional: maximum duration
  --min_speakers 100 \                         # Optional: minimum speakers required
  --min_utterances 1 \                         # Optional: min utterances per speaker
  --no_shuffle \                               # Optional: disable shuffling
  --seed 42                                    # Optional: random seed
```

### Talker LLM Data Preparation

```bash
python prepare_talker_data.py \
  --audio_dir ./raw_audio \                    # Required: audio directory
  --text_dir ./transcriptions \                # Required: text directory
  --output_dir ./manifests \                   # Optional: output directory
  --speaker_id_file speaker_mapping.txt \      # Optional: speaker ID mapping
  --train_split 0.8 \                          # Optional: train proportion
  --val_split 0.1 \                            # Optional: val proportion
  --test_split 0.1 \                           # Optional: test proportion
  --sample_rate 24000 \                        # Optional: target sample rate
  --min_duration 1.0 \                         # Optional: minimum duration
  --max_duration 20.0 \                        # Optional: maximum duration
  --min_text_length 1 \                        # Optional: minimum text length
  --max_text_length 500 \                      # Optional: maximum text length
  --no_shuffle \                               # Optional: disable shuffling
  --seed 42 \                                  # Optional: random seed
  --default_speaker speaker_001                # Optional: default speaker ID
```

---

## Speaker Mapping File Format

For Talker LLM, you can optionally provide a speaker mapping file to assign speakers to audio files.

**Format:** One mapping per line
```
sample_001 speaker_A
sample_002 speaker_B
sample_003 speaker_A
...
```

**Example:**
```bash
python prepare_talker_data.py \
  --audio_dir ./raw_audio \
  --text_dir ./transcriptions \
  --speaker_id_file speaker_mapping.txt
```

If no mapping file is provided, all samples use the `--default_speaker` (default: `speaker_001`).

---

## Manifest File Format

All scripts generate JSONL manifest files. Each line is a valid JSON object.

### Tokenizer Manifest
```jsonl
{"audio_path": "/absolute/path/to/audio.wav", "duration": 5.23, "language": "en"}
{"audio_path": "/absolute/path/to/audio2.wav", "duration": 3.15, "language": "en"}
```

### Speaker Encoder Manifest
```jsonl
{"audio_path": "/absolute/path/to/speaker_001/utt_001.wav", "duration": 4.5, "speaker_id": "speaker_001"}
{"audio_path": "/absolute/path/to/speaker_002/utt_001.wav", "duration": 3.2, "speaker_id": "speaker_002"}
```

### Talker LLM Manifest
```jsonl
{"audio_path": "/absolute/path/to/sample_001.wav", "duration": 5.0, "text": "transcription here", "speaker_id": "speaker_001", "text_length": 25}
{"audio_path": "/absolute/path/to/sample_002.wav", "duration": 3.5, "text": "more text", "speaker_id": "speaker_002", "text_length": 10}
```

---

## Utility Functions

The `utils.py` module provides reusable functions:

```python
from utils import *

# Get audio duration
duration = get_audio_duration("audio.wav", sr=24000)

# Load audio
waveform, sr = load_audio("audio.wav", sr=24000)

# Validate audio
is_valid, message = validate_audio("audio.wav", min_duration=1.0, max_duration=20.0)

# Create manifest entry
entry = create_manifest_entry("audio.wav", speaker_id="speaker_001")

# Save manifest
save_manifest(entries, "manifest.jsonl")

# Load manifest
entries = load_manifest("manifest.jsonl")

# Validate manifest
stats = validate_manifest("manifest.jsonl")
print_manifest_stats("manifest.jsonl")

# Split manifest
train, val, test = split_manifest(entries, train_ratio=0.8, val_ratio=0.1)

# Filter entries
valid, removed = filter_entries(entries, min_duration=1.0, max_duration=20.0)
```

---

## Examples

### Example 1: Simple Tokenizer Preparation

```bash
# Assuming you have audio files in ./my_audio/
python prepare_tokenizer_data.py \
  --audio_dir ./my_audio \
  --output_dir ./manifests \
  --language en
```

Output:
```
./manifests/
├── train_manifest.jsonl    (90% of samples)
└── val_manifest.jsonl      (10% of samples)
```

### Example 2: Speaker Encoder with Multiple Speakers

```bash
# Speaker directory structure:
# speakers/
# ├── alice/
# │   ├── hello.wav
# │   └── goodbye.wav
# ├── bob/
# │   ├── hello.wav
# │   └── goodbye.wav
# └── charlie/
#     └── hello.wav

python prepare_speaker_data.py \
  --audio_dir ./speakers \
  --output_dir ./manifests \
  --min_speakers 10 \
  --train_split 0.8 \
  --val_split 0.2
```

### Example 3: Talker LLM with Speaker Mapping

Create `speaker_mapping.txt`:
```
utt_001 alice
utt_002 alice
utt_003 bob
utt_004 bob
utt_005 charlie
```

Create transcriptions:
```
transcriptions/
├── utt_001.txt  (content: "Hello, how are you?")
├── utt_002.txt  (content: "I'm doing great!")
├── utt_003.txt  (content: "What's your name?")
└── ...
```

Run:
```bash
python prepare_talker_data.py \
  --audio_dir ./audio \
  --text_dir ./transcriptions \
  --speaker_id_file speaker_mapping.txt \
  --train_split 0.8 \
  --val_split 0.1 \
  --test_split 0.1
```

---

## Troubleshooting

### Error: "No audio files found"
- Check the directory path is correct
- Ensure files have extensions: `.wav`, `.mp3`, or `.flac`
- Files are case-sensitive on Linux

### Error: "No text found for audio"
- For Talker LLM: audio and text files must have the same basename
- Example: `sample_001.wav` ↔ `sample_001.txt`

### Error: "Invalid text length"
- Text must be between `--min_text_length` and `--max_text_length`
- Default: 1-500 characters
- Increase `--max_text_length` for longer texts

### Error: "Only X speakers found, minimum is Y"
- You need more speakers in your dataset
- Decrease `--min_speakers` or collect more data

### Warning: "Duration mismatch"
- Audio files might be corrupted
- Resample to 24kHz using FFmpeg:
  ```bash
  ffmpeg -i input.wav -acodec pcm_s16le -ar 24000 output.wav
  ```

---

## Checking Results

After running data preparation, check the manifests:

```bash
# Print statistics
python -c "
from utils import print_manifest_stats
print_manifest_stats('manifests/train_manifest.jsonl')
print_manifest_stats('manifests/val_manifest.jsonl')
"

# Validate manifests
python -c "
from utils import validate_manifest
stats = validate_manifest('manifests/train_manifest.jsonl')
print(stats)
"
```

---

## Performance Tips

1. **Parallel Processing**: For large datasets, run on multiple machines:
   ```bash
   # On machine 1: process first half
   # On machine 2: process second half
   # Then merge manifests
   ```

2. **Incremental Processing**: Process in batches:
   ```bash
   # Collect audio in batches
   # Run prepare_*_data.py for each batch
   # Combine manifests at the end
   ```

3. **Caching**: Utils functions cache audio durations, so running again is faster

---

## Next Steps

After data preparation:

1. **Verify manifests are created:**
   ```bash
   ls -la datasets/*/manifests/
   ```

2. **Check dataset statistics:**
   ```bash
   python -c "from utils import print_manifest_stats; print_manifest_stats('datasets/tokenizer_data/manifests/train_manifest.jsonl')"
   ```

3. **Proceed to Phase 1 training:**
   ```bash
   cd ../1_tokenizer_training/
   python train_tokenizer.py --config ../../configs/tokenizer.yaml
   ```

---

**Time to complete:** 1-2 hours (depending on dataset size)
**Next Phase:** Phase 1 - Tokenizer Training

# coding=utf-8
"""Prepare data for Talker LLM training.

Pairs audio files with text transcriptions and speaker IDs.
Expected directory structure:
  raw_audio/
  ├── sample_001.wav
  ├── sample_002.wav
  └── ...

  transcriptions/
  ├── sample_001.txt
  ├── sample_002.txt
  └── ...
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

from utils import (
    create_manifest_entry,
    filter_entries,
    get_audio_files,
    print_manifest_stats,
    save_manifest,
    split_manifest,
    validate_audio,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_speaker_mapping(mapping_file: str) -> Dict[str, str]:
    """Load speaker ID mapping from file.

    File format: one mapping per line, format: basename speaker_id
    Example:
      sample_001 speaker_001
      sample_002 speaker_002

    Args:
        mapping_file: Path to mapping file

    Returns:
        Dictionary mapping basename to speaker_id
    """
    mapping = {}

    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    basename = parts[0]
                    speaker_id = parts[1]
                    mapping[basename] = speaker_id

        logger.info(f"Loaded speaker mapping for {len(mapping)} samples")
    except FileNotFoundError:
        logger.warning(f"Speaker mapping file not found: {mapping_file}")

    return mapping


def load_text_file(text_path: str) -> str:
    """Load text from file.

    Args:
        text_path: Path to text file

    Returns:
        Text content
    """
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text
    except Exception as e:
        logger.warning(f"Failed to load text from {text_path}: {e}")
        return ""


def prepare_talker_data(
    audio_dir: str,
    text_dir: str,
    output_dir: str,
    speaker_id_file: Optional[str] = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    sample_rate: int = 24000,
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    min_text_length: int = 1,
    max_text_length: int = 500,
    shuffle: bool = True,
    seed: int = 42,
    default_speaker: str = "speaker_001",
) -> None:
    """Prepare Talker LLM training data.

    Args:
        audio_dir: Directory containing audio files
        text_dir: Directory containing text transcriptions
        output_dir: Directory to save manifests
        speaker_id_file: Optional file mapping audio names to speaker IDs
        train_split: Proportion for training
        val_split: Proportion for validation
        test_split: Proportion for testing
        sample_rate: Expected sample rate (Hz)
        min_duration: Minimum audio duration (seconds)
        max_duration: Maximum audio duration (seconds)
        min_text_length: Minimum text length (characters)
        max_text_length: Maximum text length (characters)
        shuffle: Whether to shuffle before splitting
        seed: Random seed
        default_speaker: Default speaker ID if not specified
    """
    logger.info(f"Preparing Talker LLM data from audio_dir={audio_dir}, text_dir={text_dir}")

    # Load speaker mapping if provided
    speaker_mapping = {}
    if speaker_id_file:
        speaker_mapping = load_speaker_mapping(speaker_id_file)

    # Get all audio files
    audio_files = get_audio_files(audio_dir)

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files")

    # Create manifest entries
    logger.info("Creating manifest entries...")
    entries = []
    invalid_count = 0

    for i, audio_path in enumerate(audio_files):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(audio_files)} files")

        # Get basename without extension
        basename = Path(audio_path).stem

        # Find corresponding text file
        text_path = Path(text_dir) / f"{basename}.txt"

        if not text_path.exists():
            invalid_count += 1
            if invalid_count <= 10:
                logger.debug(f"No text found for {audio_path}")
            continue

        # Load text
        text = load_text_file(str(text_path))

        # Validate text
        if not text or len(text) < min_text_length or len(text) > max_text_length:
            invalid_count += 1
            if invalid_count <= 10:
                logger.debug(
                    f"Invalid text for {audio_path}: "
                    f"length={len(text)}, "
                    f"expected {min_text_length}-{max_text_length}"
                )
            continue

        # Validate audio
        is_valid, message = validate_audio(
            audio_path,
            sr=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        if not is_valid:
            invalid_count += 1
            if invalid_count <= 10:
                logger.debug(f"Invalid audio: {audio_path} - {message}")
            continue

        # Get speaker ID
        speaker_id = speaker_mapping.get(basename, default_speaker)

        # Create entry
        entry = create_manifest_entry(
            audio_path,
            text=text,
            speaker_id=speaker_id,
            text_length=len(text),
        )
        entries.append(entry)

    logger.info(f"Valid entries: {len(entries)}, Invalid: {invalid_count}")

    if not entries:
        logger.error("No valid entries found!")
        return

    # Filter by duration
    logger.info("Filtering by duration...")
    entries, removed = filter_entries(
        entries,
        min_duration=min_duration,
        max_duration=max_duration,
        remove_invalid=False,
    )

    # Split into train/val/test
    logger.info("Splitting into train/val/test sets...")
    train_entries, temp_entries = split_manifest(
        entries,
        train_ratio=train_split,
        val_ratio=val_split + test_split,
        shuffle=shuffle,
        seed=seed,
    )

    # Further split temp into val/test
    val_ratio_adjusted = val_split / (val_split + test_split)
    val_entries, test_entries = split_manifest(
        temp_entries,
        train_ratio=val_ratio_adjusted,
        val_ratio=1 - val_ratio_adjusted,
        shuffle=False,
        seed=seed,
    )

    # Save manifests
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_path = Path(output_dir) / "train_manifest.jsonl"
    val_path = Path(output_dir) / "val_manifest.jsonl"
    test_path = Path(output_dir) / "test_manifest.jsonl"

    save_manifest(train_entries, str(train_path))
    save_manifest(val_entries, str(val_path))
    save_manifest(test_entries, str(test_path))

    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("TALKER LLM DATA PREPARATION COMPLETE")
    logger.info("=" * 60)

    # Count unique speakers
    train_speakers = len(set(e["speaker_id"] for e in train_entries))
    val_speakers = len(set(e["speaker_id"] for e in val_entries))
    test_speakers = len(set(e["speaker_id"] for e in test_entries))

    logger.info(f"Train split: {len(train_entries)} samples from {train_speakers} speakers")
    logger.info(f"Val split:   {len(val_entries)} samples from {val_speakers} speakers")
    logger.info(f"Test split:  {len(test_entries)} samples from {test_speakers} speakers")

    print_manifest_stats(str(train_path))
    print_manifest_stats(str(val_path))
    print_manifest_stats(str(test_path))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for Talker LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected directory structure:
  raw_audio/
  ├── sample_001.wav
  ├── sample_002.wav
  └── ...

  transcriptions/
  ├── sample_001.txt
  ├── sample_002.txt
  └── ...

Speaker mapping file format (optional):
  sample_001 speaker_001
  sample_002 speaker_002
  ...

Examples:
  python prepare_talker_data.py \\
    --audio_dir ./raw_audio \\
    --text_dir ./transcriptions

  python prepare_talker_data.py \\
    --audio_dir ./raw_audio \\
    --text_dir ./transcriptions \\
    --speaker_id_file speaker_mapping.txt \\
    --train_split 0.8 \\
    --val_split 0.1
        """,
    )

    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        required=True,
        help="Directory containing text transcriptions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="manifests",
        help="Directory to save manifests (default: manifests)",
    )
    parser.add_argument(
        "--speaker_id_file",
        type=str,
        default=None,
        help="Optional file mapping audio names to speaker IDs",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Proportion for training (default: 0.8)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proportion for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Proportion for testing (default: 0.1)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Expected sample rate in Hz (default: 24000)",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="Minimum audio duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=20.0,
        help="Maximum audio duration in seconds (default: 20.0)",
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=1,
        help="Minimum text length in characters (default: 1)",
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=500,
        help="Maximum text length in characters (default: 500)",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable shuffling before split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--default_speaker",
        type=str,
        default="speaker_001",
        help="Default speaker ID (default: speaker_001)",
    )

    args = parser.parse_args()

    prepare_talker_data(
        audio_dir=args.audio_dir,
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        speaker_id_file=args.speaker_id_file,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        default_speaker=args.default_speaker,
    )


if __name__ == "__main__":
    main()

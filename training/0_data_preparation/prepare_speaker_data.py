# coding=utf-8
"""Prepare data for Speaker Encoder training.

Converts speaker-organized audio files into training manifests.
Expected directory structure:
  raw_audio/
  ├── speaker_001/
  │   ├── utt_001.wav
  │   └── utt_002.wav
  ├── speaker_002/
  │   └── utt_001.wav
  └── ...
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from utils import (
    create_manifest_entry,
    filter_entries,
    get_speaker_dirs,
    print_manifest_stats,
    save_manifest,
    split_manifest,
    validate_audio,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_speaker_data(
    audio_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.2,
    sample_rate: int = 24000,
    min_duration: float = 1.0,
    max_duration: float = 10.0,
    min_speakers: int = 100,
    min_utterances_per_speaker: int = 1,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    """Prepare speaker encoder training data.

    Args:
        audio_dir: Directory containing speaker subdirectories
        output_dir: Directory to save manifests
        train_split: Proportion of speakers for training
        val_split: Proportion of speakers for validation
        sample_rate: Expected sample rate (Hz)
        min_duration: Minimum audio duration (seconds)
        max_duration: Maximum audio duration (seconds)
        min_speakers: Minimum number of speakers required
        min_utterances_per_speaker: Minimum utterances per speaker
        shuffle: Whether to shuffle before splitting
        seed: Random seed
    """
    logger.info(f"Preparing speaker encoder data from {audio_dir}")

    # Get speaker directories
    speaker_dirs = get_speaker_dirs(audio_dir)

    if not speaker_dirs:
        logger.error(f"No speaker directories found in {audio_dir}")
        return

    logger.info(f"Found {len(speaker_dirs)} speakers")

    # Create manifest entries
    logger.info("Creating manifest entries...")
    entries = []
    speakers_with_valid_utts = 0
    total_valid_utts = 0

    for speaker_id, audio_files in sorted(speaker_dirs.items()):
        speaker_entries = []

        for audio_path in audio_files:
            # Validate audio
            is_valid, message = validate_audio(
                audio_path,
                sr=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            if not is_valid:
                logger.debug(f"Invalid: {audio_path} - {message}")
                continue

            # Create entry
            entry = create_manifest_entry(
                audio_path,
                speaker_id=speaker_id,
            )
            speaker_entries.append(entry)

        # Check minimum utterances per speaker
        if len(speaker_entries) >= min_utterances_per_speaker:
            entries.extend(speaker_entries)
            speakers_with_valid_utts += 1
            total_valid_utts += len(speaker_entries)

    logger.info(
        f"Valid speakers: {speakers_with_valid_utts}, "
        f"Total utterances: {total_valid_utts}"
    )

    # Check minimum speakers
    if speakers_with_valid_utts < min_speakers:
        logger.warning(
            f"Only {speakers_with_valid_utts} speakers found, "
            f"but minimum required is {min_speakers}"
        )

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

    # Split by speakers (not utterances) for better generalization
    logger.info("Splitting by speakers into train/val sets...")
    train_entries, val_entries, _ = split_by_speakers(
        entries,
        train_ratio=train_split,
        val_ratio=val_split,
        shuffle=shuffle,
        seed=seed,
    )

    # Save manifests
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_path = Path(output_dir) / "train_manifest.jsonl"
    val_path = Path(output_dir) / "val_manifest.jsonl"

    save_manifest(train_entries, str(train_path))
    save_manifest(val_entries, str(val_path))

    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("SPEAKER ENCODER DATA PREPARATION COMPLETE")
    logger.info("=" * 60)

    # Count unique speakers per split
    train_speakers = len(set(e["speaker_id"] for e in train_entries))
    val_speakers = len(set(e["speaker_id"] for e in val_entries))

    logger.info(f"Train split: {len(train_entries)} utterances from {train_speakers} speakers")
    logger.info(f"Val split:   {len(val_entries)} utterances from {val_speakers} speakers")

    print_manifest_stats(str(train_path))
    print_manifest_stats(str(val_path))


def split_by_speakers(
    entries: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple:
    """Split entries by speakers (not utterances).

    Ensures each speaker appears only in train or val, not both.

    Args:
        entries: List of manifest entries with speaker_id
        train_ratio: Proportion of speakers for training
        val_ratio: Proportion of speakers for validation
        shuffle: Whether to shuffle speaker order
        seed: Random seed

    Returns:
        Tuple of (train_entries, val_entries, test_entries)
    """
    import numpy as np

    # Group by speaker
    speakers_to_entries = defaultdict(list)
    for entry in entries:
        speaker_id = entry.get("speaker_id", "unknown")
        speakers_to_entries[speaker_id].append(entry)

    speakers = list(speakers_to_entries.keys())

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(speakers)

    # Split speakers
    n_train = int(len(speakers) * train_ratio)
    n_val = int(len(speakers) * val_ratio)

    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train : n_train + n_val]
    test_speakers = speakers[n_train + n_val :]

    # Gather entries
    train_entries = [
        e for speaker in train_speakers for e in speakers_to_entries[speaker]
    ]
    val_entries = [e for speaker in val_speakers for e in speakers_to_entries[speaker]]
    test_entries = [
        e for speaker in test_speakers for e in speakers_to_entries[speaker]
    ]

    logger.info(
        f"Split by speakers: "
        f"train={len(train_speakers)}, "
        f"val={len(val_speakers)}, "
        f"test={len(test_speakers)}"
    )

    return train_entries, val_entries, test_entries


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for Speaker Encoder training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected directory structure:
  raw_audio/
  ├── speaker_001/
  │   ├── utt_001.wav
  │   └── utt_002.wav
  ├── speaker_002/
  │   └── utt_001.wav
  └── ...

Examples:
  python prepare_speaker_data.py --audio_dir ./raw_audio

  python prepare_speaker_data.py \\
    --audio_dir ./raw_audio \\
    --output_dir ./manifests \\
    --min_speakers 1000
        """,
    )

    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing speaker subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="manifests",
        help="Directory to save manifests (default: manifests)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Proportion of speakers for training (default: 0.8)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Proportion of speakers for validation (default: 0.2)",
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
        default=10.0,
        help="Maximum audio duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=100,
        help="Minimum number of speakers required (default: 100)",
    )
    parser.add_argument(
        "--min_utterances",
        type=int,
        default=1,
        help="Minimum utterances per speaker (default: 1)",
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

    args = parser.parse_args()

    prepare_speaker_data(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_speakers=args.min_speakers,
        min_utterances_per_speaker=args.min_utterances,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

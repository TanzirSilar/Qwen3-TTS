# coding=utf-8
"""Prepare data for Tokenizer training.

Converts raw audio files into training manifests with metadata.
"""

import argparse
import logging
from pathlib import Path
from typing import List

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


def prepare_tokenizer_data(
    audio_dir: str,
    output_dir: str,
    train_split: float = 0.9,
    val_split: float = 0.1,
    sample_rate: int = 24000,
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    shuffle: bool = True,
    seed: int = 42,
    language: str = "unknown",
) -> None:
    """Prepare tokenizer training data.

    Args:
        audio_dir: Directory containing raw audio files
        output_dir: Directory to save manifests
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        sample_rate: Expected sample rate (Hz)
        min_duration: Minimum audio duration (seconds)
        max_duration: Maximum audio duration (seconds)
        shuffle: Whether to shuffle before splitting
        seed: Random seed
        language: Language code (default: unknown)
    """
    logger.info(f"Preparing tokenizer data from {audio_dir}")

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

        # Validate audio
        is_valid, message = validate_audio(
            audio_path,
            sr=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        if not is_valid:
            invalid_count += 1
            if invalid_count <= 10:  # Log first 10 invalid files
                logger.debug(f"Invalid: {audio_path} - {message}")
            continue

        # Create entry
        entry = create_manifest_entry(
            audio_path,
            language=language,
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

    # Split into train/val
    logger.info("Splitting into train/val sets...")
    train_entries, val_entries, _ = split_manifest(
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
    logger.info("TOKENIZER DATA PREPARATION COMPLETE")
    logger.info("=" * 60)
    print_manifest_stats(str(train_path))
    print_manifest_stats(str(val_path))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for Tokenizer training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_tokenizer_data.py --audio_dir ./raw_audio

  python prepare_tokenizer_data.py \\
    --audio_dir ./raw_audio \\
    --output_dir ./manifests \\
    --train_split 0.9 \\
    --language en
        """,
    )

    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing raw audio files",
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
        default=0.9,
        help="Proportion for training (default: 0.9)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proportion for validation (default: 0.1)",
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
        "--language",
        type=str,
        default="unknown",
        help="Language code (default: unknown)",
    )

    args = parser.parse_args()

    prepare_tokenizer_data(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        language=args.language,
    )


if __name__ == "__main__":
    main()

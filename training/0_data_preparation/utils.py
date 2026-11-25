# coding=utf-8
"""Utility functions for data preparation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str, sr: int = 24000) -> float:
    """Get audio duration in seconds.

    Args:
        audio_path: Path to audio file
        sr: Sample rate (default 24000 Hz)

    Returns:
        Duration in seconds
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        return float(duration)
    except Exception as e:
        logger.warning(f"Failed to get duration for {audio_path}: {e}")
        return 0.0


def load_audio(audio_path: str, sr: int = 24000) -> Tuple[np.ndarray, int]:
    """Load audio file.

    Args:
        audio_path: Path to audio file
        sr: Target sample rate

    Returns:
        Tuple of (waveform, sample_rate)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        return y.astype(np.float32), sr
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        raise


def validate_audio(audio_path: str, sr: int = 24000,
                   min_duration: float = 0.5,
                   max_duration: float = 60.0) -> Tuple[bool, str]:
    """Validate audio file.

    Args:
        audio_path: Path to audio file
        sr: Expected sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds

    Returns:
        Tuple of (is_valid, message)
    """
    if not Path(audio_path).exists():
        return False, f"File not found: {audio_path}"

    try:
        duration = get_audio_duration(audio_path, sr)

        if duration < min_duration:
            return False, f"Duration {duration:.2f}s < minimum {min_duration}s"

        if duration > max_duration:
            return False, f"Duration {duration:.2f}s > maximum {max_duration}s"

        return True, f"Valid (duration: {duration:.2f}s)"
    except Exception as e:
        return False, f"Error: {str(e)}"


def create_manifest_entry(audio_path: str, **kwargs) -> Dict:
    """Create a manifest entry for an audio file.

    Args:
        audio_path: Path to audio file
        **kwargs: Additional fields (speaker_id, text, language, etc.)

    Returns:
        Dictionary with manifest entry
    """
    duration = get_audio_duration(audio_path)

    entry = {
        "audio_path": str(Path(audio_path).absolute()),
        "duration": duration,
    }

    # Add optional fields
    entry.update(kwargs)

    return entry


def save_manifest(entries: List[Dict], output_path: str) -> None:
    """Save manifest to JSONL file.

    Args:
        entries: List of manifest entries
        output_path: Path to output JSONL file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(entries)} entries to {output_path}")


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load manifest from JSONL file.

    Args:
        manifest_path: Path to JSONL manifest

    Returns:
        List of manifest entries
    """
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))

    logger.info(f"Loaded {len(entries)} entries from {manifest_path}")
    return entries


def validate_manifest(manifest_path: str) -> Dict:
    """Validate a manifest file and return statistics.

    Args:
        manifest_path: Path to JSONL manifest

    Returns:
        Dictionary with validation statistics
    """
    entries = load_manifest(manifest_path)

    if not entries:
        return {"valid": False, "error": "No entries found"}

    durations = [e.get('duration', 0) for e in entries]

    stats = {
        "valid": True,
        "total_entries": len(entries),
        "total_duration_hours": sum(durations) / 3600,
        "avg_duration_seconds": np.mean(durations),
        "min_duration_seconds": np.min(durations),
        "max_duration_seconds": np.max(durations),
        "std_duration_seconds": np.std(durations),
    }

    # Check for fields
    if entries:
        stats["fields"] = list(entries[0].keys())

    return stats


def print_manifest_stats(manifest_path: str) -> None:
    """Print manifest statistics.

    Args:
        manifest_path: Path to JSONL manifest
    """
    stats = validate_manifest(manifest_path)

    if not stats.get("valid"):
        print(f"❌ Invalid manifest: {stats.get('error')}")
        return

    print(f"\n{'='*60}")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*60}")
    print(f"Total entries:        {stats['total_entries']}")
    print(f"Total duration:       {stats['total_duration_hours']:.2f} hours")
    print(f"Average duration:     {stats['avg_duration_seconds']:.2f} seconds")
    print(f"Min/Max duration:     {stats['min_duration_seconds']:.2f}s / {stats['max_duration_seconds']:.2f}s")
    print(f"Std deviation:        {stats['std_duration_seconds']:.2f} seconds")
    if "fields" in stats:
        print(f"Fields:               {', '.join(stats['fields'])}")
    print(f"{'='*60}\n")


def split_manifest(entries: List[Dict],
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   shuffle: bool = True,
                   seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split manifest into train/val/test sets.

    Args:
        entries: List of manifest entries
        train_ratio: Proportion for training (0.0-1.0)
        val_ratio: Proportion for validation (0.0-1.0)
        shuffle: Whether to shuffle before splitting
        seed: Random seed

    Returns:
        Tuple of (train_entries, val_entries, test_entries)
    """
    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(len(entries))
        entries = [entries[i] for i in indices]

    test_ratio = 1.0 - train_ratio - val_ratio

    n_train = int(len(entries) * train_ratio)
    n_val = int(len(entries) * val_ratio)

    train_entries = entries[:n_train]
    val_entries = entries[n_train:n_train + n_val]
    test_entries = entries[n_train + n_val:]

    logger.info(f"Split: train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}")

    return train_entries, val_entries, test_entries


def get_audio_files(directory: str, extensions: Tuple[str, ...] = ('.wav', '.mp3', '.flac')) -> List[str]:
    """Get all audio files from directory.

    Args:
        directory: Path to directory
        extensions: Tuple of audio file extensions

    Returns:
        List of audio file paths
    """
    directory = Path(directory)
    audio_files = []

    for ext in extensions:
        audio_files.extend(directory.glob(f'**/*{ext}'))
        audio_files.extend(directory.glob(f'**/*{ext.upper()}'))

    audio_files = list(set(audio_files))  # Remove duplicates
    audio_files = [str(f) for f in sorted(audio_files)]

    logger.info(f"Found {len(audio_files)} audio files in {directory}")

    return audio_files


def get_speaker_dirs(directory: str) -> Dict[str, List[str]]:
    """Get speaker directories and their audio files.

    Args:
        directory: Path to parent directory

    Returns:
        Dictionary mapping speaker_id to list of audio files
    """
    directory = Path(directory)
    speaker_dirs = {}

    for speaker_dir in sorted(directory.iterdir()):
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            audio_files = get_audio_files(str(speaker_dir))

            if audio_files:
                speaker_dirs[speaker_id] = audio_files

    logger.info(f"Found {len(speaker_dirs)} speakers in {directory}")

    return speaker_dirs


def filter_entries(entries: List[Dict],
                   min_duration: float = 0.5,
                   max_duration: float = 60.0,
                   remove_invalid: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """Filter manifest entries by duration and validity.

    Args:
        entries: List of manifest entries
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        remove_invalid: Whether to remove invalid files

    Returns:
        Tuple of (valid_entries, removed_entries)
    """
    valid_entries = []
    removed_entries = []

    for entry in entries:
        duration = entry.get('duration', 0)

        # Check duration
        if duration < min_duration or duration > max_duration:
            removed_entries.append(entry)
            continue

        # Check file exists if remove_invalid
        if remove_invalid:
            if not Path(entry['audio_path']).exists():
                removed_entries.append(entry)
                continue

        valid_entries.append(entry)

    logger.info(f"Filtered: {len(valid_entries)} valid, {len(removed_entries)} removed")

    return valid_entries, removed_entries


def merge_manifests(manifest_paths: List[str]) -> List[Dict]:
    """Merge multiple manifest files.

    Args:
        manifest_paths: List of JSONL manifest paths

    Returns:
        Combined list of entries
    """
    all_entries = []

    for path in manifest_paths:
        entries = load_manifest(path)
        all_entries.extend(entries)

    logger.info(f"Merged {len(manifest_paths)} manifests into {len(all_entries)} entries")

    return all_entries

# coding=utf-8
"""Data loader for Speaker Encoder training."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def mel_spectrogram(
    y: np.ndarray,
    sr: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 128,
    fmin: float = 0,
    fmax: float = 12000,
) -> np.ndarray:
    """Compute mel-spectrogram.

    Args:
        y: Waveform
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel-spectrogram (n_mels, time)
    """
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S = np.abs(D)

    # Mel filterbank
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spec = np.dot(mel_fb, S)

    # Log scale
    mel_spec = np.log(np.clip(mel_spec, 1e-9, None))

    # Normalize
    mean = mel_spec.mean()
    std = mel_spec.std()
    mel_spec = (mel_spec - mean) / (std + 1e-8)

    return mel_spec.astype(np.float32)


class SpeakerDataset(Dataset):
    """Dataset for speaker encoder training."""

    def __init__(
        self,
        manifest_path: str,
        sr: int = 24000,
        n_mels: int = 128,
        min_duration: float = 1.0,
        max_duration: float = 10.0,
    ):
        """Initialize dataset.

        Args:
            manifest_path: Path to JSONL manifest
            sr: Sample rate
            n_mels: Number of mel bins
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
        """
        self.sr = sr
        self.n_mels = n_mels
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Load manifest
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

        # Create speaker to index mapping
        speakers = set(s['speaker_id'] for s in self.samples)
        self.speaker_to_idx = {spk: idx for idx, spk in enumerate(sorted(speakers))}
        self.num_speakers = len(self.speaker_to_idx)

        logger.info(f"Loaded {len(self.samples)} samples with {self.num_speakers} speakers")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'mel_spec', 'speaker_id', 'speaker_idx'
        """
        sample = self.samples[idx]

        # Load audio
        audio_path = sample['audio_path']
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        # Compute mel-spectrogram
        mel_spec = mel_spectrogram(
            y,
            sr=self.sr,
            n_mels=self.n_mels,
        )

        # Get speaker index
        speaker_id = sample['speaker_id']
        speaker_idx = self.speaker_to_idx[speaker_id]

        return {
            'mel_spec': torch.from_numpy(mel_spec).float(),
            'speaker_id': speaker_id,
            'speaker_idx': speaker_idx,
        }


class BalancedBatchSampler:
    """Sampler for balanced batch sampling (speakers & utterances)."""

    def __init__(
        self,
        samples: List[Dict],
        num_speakers: int,
        num_utterances: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initialize sampler.

        Args:
            samples: List of samples with 'speaker_id'
            num_speakers: Number of speakers per batch
            num_utterances: Number of utterances per speaker
            shuffle: Whether to shuffle
            seed: Random seed
        """
        self.num_speakers = num_speakers
        self.num_utterances = num_utterances
        self.shuffle = shuffle
        self.seed = seed

        # Group by speaker
        self.speakers_to_samples = defaultdict(list)
        for idx, sample in enumerate(samples):
            speaker_id = sample['speaker_id']
            self.speakers_to_samples[speaker_id].append(idx)

        self.speakers = list(self.speakers_to_samples.keys())
        logger.info(f"Created balanced sampler with {len(self.speakers)} speakers")

    def __iter__(self):
        """Iterate over batches."""
        np.random.seed(self.seed)

        speakers = self.speakers.copy()
        if self.shuffle:
            np.random.shuffle(speakers)

        # Group speakers into batches
        for i in range(0, len(speakers), self.num_speakers):
            batch_speakers = speakers[i:i + self.num_speakers]

            if len(batch_speakers) < self.num_speakers:
                break  # Skip incomplete batch

            # Collect utterances for each speaker
            batch_indices = []
            for speaker in batch_speakers:
                utterances = self.speakers_to_samples[speaker]
                if len(utterances) >= self.num_utterances:
                    # Sample without replacement
                    sampled = np.random.choice(
                        utterances,
                        size=self.num_utterances,
                        replace=False,
                    )
                    batch_indices.extend(sampled)

            if len(batch_indices) == self.num_speakers * self.num_utterances:
                yield batch_indices

    def __len__(self) -> int:
        """Number of batches."""
        return len(self.speakers) // self.num_speakers


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader.

    Args:
        batch: List of samples

    Returns:
        Batched dictionary
    """
    # Pad mel-spectrograms to same length
    max_len = max(item['mel_spec'].shape[1] for item in batch)

    mel_specs = []
    speaker_idxs = []

    for item in batch:
        mel_spec = item['mel_spec']

        # Pad if needed
        if mel_spec.shape[1] < max_len:
            mel_spec = torch.nn.functional.pad(
                mel_spec,
                (0, max_len - mel_spec.shape[1]),
                mode='constant',
                value=0.0,
            )

        mel_specs.append(mel_spec)
        speaker_idxs.append(item['speaker_idx'])

    return {
        'mel_spec': torch.stack(mel_specs),  # (B, n_mels, time)
        'speaker_idx': torch.tensor(speaker_idxs, dtype=torch.long),  # (B,)
    }


def create_speaker_dataloader(
    manifest_path: str,
    batch_size: int = 128,
    num_speakers_per_batch: int = 32,
    num_utterances_per_speaker: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
    sr: int = 24000,
    n_mels: int = 128,
) -> Tuple[DataLoader, int]:
    """Create dataloader for speaker encoder training.

    Args:
        manifest_path: Path to JSONL manifest
        batch_size: Batch size (not used with balanced sampler)
        num_speakers_per_batch: Speakers per batch
        num_utterances_per_speaker: Utterances per speaker per batch
        num_workers: Number of workers
        shuffle: Whether to shuffle
        seed: Random seed
        sr: Sample rate
        n_mels: Number of mel bins

    Returns:
        Tuple of (dataloader, num_speakers)
    """
    # Load dataset
    dataset = SpeakerDataset(
        manifest_path,
        sr=sr,
        n_mels=n_mels,
    )

    # Create balanced sampler
    sampler = BalancedBatchSampler(
        dataset.samples,
        num_speakers=num_speakers_per_batch,
        num_utterances=num_utterances_per_speaker,
        shuffle=shuffle,
        seed=seed,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader, dataset.num_speakers

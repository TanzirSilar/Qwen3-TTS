# coding=utf-8
"""Data loader for Talker LM training."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TalkerDataset(Dataset):
    """Dataset for Talker LM training."""

    def __init__(
        self,
        manifest_path: str,
        speaker_embeddings_dir: str,
        audio_codes_dir: str,
        tokenizer=None,
    ):
        """Initialize dataset.

        Args:
            manifest_path: Path to JSONL manifest
            speaker_embeddings_dir: Directory with precomputed speaker embeddings
            audio_codes_dir: Directory with precomputed audio codes
            tokenizer: Optional tokenizer for text encoding
        """
        self.tokenizer = tokenizer
        self.speaker_embeddings_dir = Path(speaker_embeddings_dir)
        self.audio_codes_dir = Path(audio_codes_dir)

        # Load manifest
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'input_ids', 'codec_ids', 'speaker_embedding', 'text'
        """
        sample = self.samples[idx]

        # Get text
        text = sample['text']

        # Encode text (placeholder - use a real tokenizer in practice)
        if self.tokenizer is not None:
            encoded = self.tokenizer(text, return_tensors='pt')
            input_ids = encoded['input_ids'].squeeze(0)
        else:
            # Fallback: simple encoding
            input_ids = torch.tensor([ord(c) % 256 for c in text[:200]], dtype=torch.long)

        # Load speaker embedding
        speaker_id = sample['speaker_id']
        speaker_emb_path = self.speaker_embeddings_dir / f"{speaker_id}.pt"

        if speaker_emb_path.exists():
            speaker_embedding = torch.load(speaker_emb_path)
        else:
            # Placeholder: random embedding
            speaker_embedding = torch.randn(1024)

        # Load audio codes
        audio_basename = Path(sample['audio_path']).stem
        audio_codes_path = self.audio_codes_dir / f"{audio_basename}.pt"

        if audio_codes_path.exists():
            codec_ids = torch.load(audio_codes_path)  # (seq_len, 16)
        else:
            # Placeholder: random codes
            codec_ids = torch.randint(0, 2048, (100, 16), dtype=torch.long)

        return {
            'input_ids': input_ids,
            'codec_ids': codec_ids,
            'speaker_embedding': speaker_embedding,
            'text': text,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader.

    Args:
        batch: List of samples

    Returns:
        Batched dictionary
    """
    # Pad sequences
    max_seq_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids_list = []
    codec_ids_list = []
    speaker_embeddings_list = []

    for item in batch:
        input_ids = item['input_ids']
        codec_ids = item['codec_ids']
        speaker_embedding = item['speaker_embedding']

        # Pad input_ids
        if input_ids.shape[0] < max_seq_len:
            input_ids = torch.nn.functional.pad(
                input_ids,
                (0, max_seq_len - input_ids.shape[0]),
                mode='constant',
                value=0,
            )
        else:
            input_ids = input_ids[:max_seq_len]

        # Pad codec_ids
        if codec_ids.shape[0] < max_seq_len:
            codec_ids = torch.nn.functional.pad(
                codec_ids,
                (0, 0, 0, max_seq_len - codec_ids.shape[0]),
                mode='constant',
                value=0,
            )
        else:
            codec_ids = codec_ids[:max_seq_len]

        input_ids_list.append(input_ids)
        codec_ids_list.append(codec_ids)
        speaker_embeddings_list.append(speaker_embedding)

    return {
        'input_ids': torch.stack(input_ids_list),  # (B, seq_len)
        'codec_ids': torch.stack(codec_ids_list),  # (B, seq_len, 16)
        'speaker_embedding': torch.stack(speaker_embeddings_list),  # (B, 1024)
    }


def create_talker_dataloader(
    manifest_path: str,
    speaker_embeddings_dir: str,
    audio_codes_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    tokenizer=None,
) -> DataLoader:
    """Create dataloader for Talker LM training.

    Args:
        manifest_path: Path to JSONL manifest
        speaker_embeddings_dir: Directory with speaker embeddings
        audio_codes_dir: Directory with audio codes
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        tokenizer: Optional tokenizer

    Returns:
        DataLoader
    """
    dataset = TalkerDataset(
        manifest_path,
        speaker_embeddings_dir,
        audio_codes_dir,
        tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader

# coding=utf-8
"""Audio dataset for codec training.

Reads the JSONL manifests written by training/0_data_preparation/prepare_tokenizer_data.py
(one {"audio_path": ..., "duration": ...} per line) and returns random fixed-length crops,
resampled to 24 kHz mono, with length a multiple of the 1920-sample codec frame.
"""

import json
import logging
import random
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_manifest(path: str, min_duration: float = 0.0, max_duration: float = float("inf")) -> List[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            d = e.get("duration")
            if d is not None and not (min_duration <= d <= max_duration):
                continue
            items.append(e)
    return items


def read_audio(path: str, target_sr: int, start_sec: float = 0.0, dur_sec: Optional[float] = None) -> torch.Tensor:
    """Read a (partial) mono float32 waveform at target_sr. Returns (N,)."""
    info = sf.info(path)
    sr = info.samplerate
    start = int(start_sec * sr)
    frames = -1 if dur_sec is None else int(np.ceil(dur_sec * sr)) + 1
    wav, sr = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    return wav


class CodecAudioDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        sample_rate: int = 24000,
        segment_seconds: float = 3.84,
        frame_samples: int = 1920,
        min_duration: float = 0.5,
        max_duration: float = 60.0,
        random_crop: bool = True,
        gain_db_range=(-6.0, 0.0),
        target_peak: float = 0.95,
    ):
        self.items = load_manifest(manifest, min_duration, max_duration)
        if not self.items:
            raise ValueError(f"No usable entries in {manifest}")
        self.sr = sample_rate
        self.seg = int(round(segment_seconds * sample_rate / frame_samples)) * frame_samples
        self.frame = frame_samples
        self.random_crop = random_crop
        self.gain_db_range = gain_db_range
        self.target_peak = target_peak
        logger.info(f"{manifest}: {len(self.items)} files, segment={self.seg} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        e = self.items[idx]
        seg_sec = self.seg / self.sr
        dur = e.get("duration")
        try:
            if dur is not None and dur > seg_sec:
                start = random.uniform(0, dur - seg_sec) if self.random_crop else 0.0
                wav = read_audio(e["audio_path"], self.sr, start, seg_sec)
            else:
                wav = read_audio(e["audio_path"], self.sr)
        except Exception as ex:  # corrupt file: fall back to another random item
            logger.warning(f"Failed to read {e['audio_path']}: {ex}")
            return self[random.randrange(len(self))]

        wav = wav[: self.seg]
        n_valid = wav.shape[0]
        if n_valid < self.seg:
            wav = torch.nn.functional.pad(wav, (0, self.seg - n_valid))

        # peak-normalise then random gain (keeps the codec robust to level)
        peak = wav.abs().max()
        if peak > 1e-4:
            wav = wav / peak * self.target_peak
        if self.random_crop and self.gain_db_range is not None:
            g = random.uniform(*self.gain_db_range)
            wav = wav * (10 ** (g / 20))

        n_frames = self.seg // self.frame
        valid_frames = max(1, int(np.ceil(n_valid / self.frame)))
        frame_mask = torch.zeros(n_frames, dtype=torch.bool)
        frame_mask[:valid_frames] = True
        return {"wav": wav.unsqueeze(0), "frame_mask": frame_mask, "n_valid": n_valid}


def collate(batch):
    return {
        "wav": torch.stack([b["wav"] for b in batch]),            # (B, 1, N)
        "frame_mask": torch.stack([b["frame_mask"] for b in batch]),  # (B, T)
        "n_valid": torch.tensor([b["n_valid"] for b in batch]),
    }


class SyntheticAudioDataset(Dataset):
    """Random harmonic signals - used only for smoke tests without real data."""

    def __init__(self, n: int = 256, sample_rate: int = 24000, segment_seconds: float = 3.84, frame_samples: int = 1920):
        self.n, self.sr, self.frame = n, sample_rate, frame_samples
        self.seg = int(round(segment_seconds * sample_rate / frame_samples)) * frame_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        t = torch.arange(self.seg) / self.sr
        f0 = 80 + 220 * torch.rand(1, generator=g)
        wav = sum((0.5 / k) * torch.sin(2 * np.pi * f0 * k * t * (1 + 0.05 * torch.sin(2 * np.pi * 3 * t)))
                  for k in range(1, 8))
        wav = wav + 0.01 * torch.randn(self.seg, generator=g)
        wav = wav / wav.abs().max() * 0.9
        return {"wav": wav.unsqueeze(0).float(), "frame_mask": torch.ones(self.seg // self.frame, dtype=torch.bool),
                "n_valid": self.seg}

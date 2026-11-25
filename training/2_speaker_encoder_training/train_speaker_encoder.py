# coding=utf-8
"""Train Speaker Encoder (ECAPA-TDNN) from scratch."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from data_loader import create_speaker_dataloader
from model import SpeakerEncoderConfig, SpeakerEncoderWithHead
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


class SpeakerEncoderTrainer:
    """Trainer for Speaker Encoder."""

    def __init__(
        self,
        output_dir: str,
        train_manifest: str,
        val_manifest: str,
        config_path: str,
        batch_size: int = 128,
        num_speakers_per_batch: int = 32,
        num_utterances_per_speaker: int = 4,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        warmup_steps: int = 5000,
        max_grad_norm: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 100,
        gradient_accumulation_steps: int = 1,
    ):
        """Initialize trainer.

        Args:
            output_dir: Output directory
            train_manifest: Training manifest path
            val_manifest: Validation manifest path
            config_path: Config YAML path
            batch_size: Batch size (for other parameters)
            num_speakers_per_batch: Speakers per batch
            num_utterances_per_speaker: Utterances per speaker
            learning_rate: Learning rate
            num_epochs: Number of epochs
            warmup_steps: Warmup steps
            max_grad_norm: Max gradient norm
            save_steps: Save every N steps
            eval_steps: Evaluate every N steps
            gradient_accumulation_steps: Gradient accumulation steps
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.batch_size = batch_size
        self.num_speakers_per_batch = num_speakers_per_batch
        self.num_utterances_per_speaker = num_utterances_per_speaker
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self._setup_logging()
        self._load_config(config_path)
        self._setup_accelerator()

    def _setup_logging(self):
        """Setup logging."""
        log_file = self.output_dir / "training.log"
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )
        logger.info(f"Training started at {datetime.now().isoformat()}")

    def _load_config(self, config_path: str):
        """Load config from YAML."""
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        self.config = SpeakerEncoderConfig(**config_dict.get('model', {}))
        logger.info(f"Loaded config: {self.config}")

    def _setup_accelerator(self):
        """Setup Accelerate."""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision="bf16",
            log_with="tensorboard",
            logging_dir=str(self.output_dir / "logs"),
        )
        logger.info(f"Accelerator initialized on device: {self.accelerator.device}")

    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info("Starting Speaker Encoder Training")
        logger.info("=" * 80)

        # Load dataloaders
        logger.info("Loading data...")
        train_loader, num_speakers = create_speaker_dataloader(
            self.train_manifest,
            batch_size=self.batch_size,
            num_speakers_per_batch=self.num_speakers_per_batch,
            num_utterances_per_speaker=self.num_utterances_per_speaker,
        )

        val_loader, _ = create_speaker_dataloader(
            self.val_manifest,
            batch_size=self.num_speakers_per_batch,
            num_speakers_per_batch=self.num_speakers_per_batch,
            num_utterances_per_speaker=self.num_utterances_per_speaker,
            shuffle=False,
        )

        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Total speakers: {num_speakers}")

        # Create model
        logger.info("Creating model...")
        model = SpeakerEncoderWithHead(
            self.config,
            num_speakers=num_speakers,
            margin=0.3,
            scale=30.0,
        )

        # Create optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )

        # Setup scheduler
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        # Prepare with accelerator
        model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        global_step = 0
        best_val_accuracy = 0.0
        best_val_eer = float('inf')

        model.train()

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch}/{self.num_epochs - 1}")

            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0

            for step, batch in enumerate(train_loader):
                with self.accelerator.accumulate(model):
                    # Forward pass
                    mel_specs = batch['mel_spec']
                    speaker_idxs = batch['speaker_idx']

                    embeddings, logits = model(mel_specs, speaker_idxs)
                    loss = criterion(logits, speaker_idxs)

                    # Backward pass
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Metrics
                    with torch.no_grad():
                        preds = logits.argmax(dim=1)
                        accuracy = (preds == speaker_idxs).float().mean()

                    epoch_loss += loss.item()
                    epoch_accuracy += accuracy.item()
                    num_batches += 1
                    global_step += 1

                if step % 10 == 0 and self.accelerator.is_main_process:
                    avg_loss = epoch_loss / (step + 1)
                    avg_acc = epoch_accuracy / (step + 1)
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"  Step {step} | Loss: {avg_loss:.4f} | "
                        f"Accuracy: {avg_acc:.4f} | LR: {lr:.2e}"
                    )

                # Evaluation
                if self.eval_steps > 0 and step % self.eval_steps == 0 and step > 0:
                    val_loss, val_accuracy = self._evaluate(model, val_loader, criterion)

                    if self.accelerator.is_main_process:
                        logger.info(
                            f"  Validation | Loss: {val_loss:.4f} | "
                            f"Accuracy: {val_accuracy:.4f}"
                        )

                        if val_accuracy > best_val_accuracy:
                            best_val_accuracy = val_accuracy
                            self._save_checkpoint(model, f"checkpoint-step-{global_step}-best")

                # Checkpointing
                if self.save_steps > 0 and step % self.save_steps == 0 and step > 0:
                    if self.accelerator.is_main_process:
                        self._save_checkpoint(model, f"checkpoint-step-{global_step}")

            # Save epoch checkpoint
            if self.accelerator.is_main_process:
                self._save_checkpoint(model, f"checkpoint-epoch-{epoch}")

            # Epoch statistics
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
        logger.info("=" * 80)

    def _evaluate(self, model, dataloader, criterion):
        """Evaluate model.

        Args:
            model: Model
            dataloader: Validation dataloader
            criterion: Loss function

        Returns:
            Tuple of (loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                mel_specs = batch['mel_spec']
                speaker_idxs = batch['speaker_idx']

                embeddings, logits = model(mel_specs, speaker_idxs)
                loss = criterion(logits, speaker_idxs)

                preds = logits.argmax(dim=1)
                accuracy = (preds == speaker_idxs).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1

        model.train()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0

        return avg_loss, avg_accuracy

    def _save_checkpoint(self, model, checkpoint_name: str):
        """Save checkpoint.

        Args:
            model: Model
            checkpoint_name: Checkpoint name
        """
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        unwrapped_model = self.accelerator.unwrap_model(model)
        torch.save(
            unwrapped_model.state_dict(),
            checkpoint_dir / "model.pt",
        )

        # Save config
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump({
                'mel_dim': self.config.mel_dim,
                'enc_dim': self.config.enc_dim,
                'enc_channels': self.config.enc_channels,
                'enc_kernel_sizes': self.config.enc_kernel_sizes,
                'enc_dilations': self.config.enc_dilations,
            }, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Speaker Encoder")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--train_manifest", type=str, required=True, help="Training manifest path")
    parser.add_argument("--val_manifest", type=str, required=True, help="Validation manifest path")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_speakers_per_batch", type=int, default=32, help="Speakers per batch")
    parser.add_argument("--num_utterances_per_speaker", type=int, default=4, help="Utterances per speaker")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")

    args = parser.parse_args()

    trainer = SpeakerEncoderTrainer(
        output_dir=args.output_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        config_path=args.config,
        batch_size=args.batch_size,
        num_speakers_per_batch=args.num_speakers_per_batch,
        num_utterances_per_speaker=args.num_utterances_per_speaker,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )

    trainer.train()


if __name__ == "__main__":
    main()

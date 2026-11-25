# coding=utf-8
"""Train Talker LM (Transformer) from scratch."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from accelerate import Accelerator
from data_loader import create_talker_dataloader
from model import TalkerConfig, TalkerLMWithLoss
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


class TalkerLMTrainer:
    """Trainer for Talker LM."""

    def __init__(
        self,
        output_dir: str,
        train_manifest: str,
        val_manifest: str,
        speaker_embeddings_dir: str,
        audio_codes_dir: str,
        config_path: str,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 500,
        gradient_accumulation_steps: int = 4,
    ):
        """Initialize trainer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.speaker_embeddings_dir = speaker_embeddings_dir
        self.audio_codes_dir = audio_codes_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
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
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        self.config = TalkerConfig(**config_dict.get('model', {}))
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
        logger.info("Starting Talker LM Training")
        logger.info("=" * 80)

        # Load dataloaders
        logger.info("Loading data...")
        train_loader = create_talker_dataloader(
            self.train_manifest,
            self.speaker_embeddings_dir,
            self.audio_codes_dir,
            batch_size=self.batch_size,
        )

        val_loader = create_talker_dataloader(
            self.val_manifest,
            self.speaker_embeddings_dir,
            self.audio_codes_dir,
            batch_size=self.batch_size,
            shuffle=False,
        )

        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")

        # Create model
        logger.info("Creating model...")
        model = TalkerLMWithLoss(self.config)

        # Create optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        # Setup scheduler
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Prepare with accelerator
        model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

        # Training loop
        global_step = 0
        best_val_loss = float('inf')

        model.train()

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch}/{self.num_epochs - 1}")

            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(train_loader):
                with self.accelerator.accumulate(model):
                    # Forward pass
                    input_ids = batch['input_ids']
                    codec_ids = batch['codec_ids']
                    speaker_embedding = batch['speaker_embedding']

                    output = model(
                        input_ids=input_ids,
                        codec_ids=codec_ids,
                        speaker_embeddings=speaker_embedding,
                        return_loss=True,
                    )
                    loss = output['loss']

                    # Backward pass
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()
                    num_batches += 1
                    global_step += 1

                if step % 10 == 0 and self.accelerator.is_main_process:
                    avg_loss = epoch_loss / (step + 1)
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"  Step {step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

                # Evaluation
                if self.eval_steps > 0 and step % self.eval_steps == 0 and step > 0:
                    val_loss = self._evaluate(model, val_loader)

                    if self.accelerator.is_main_process:
                        logger.info(f"  Validation Loss: {val_loss:.4f}")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self._save_checkpoint(model, f"checkpoint-step-{global_step}-best")

                # Checkpointing
                if self.save_steps > 0 and step % self.save_steps == 0 and step > 0:
                    if self.accelerator.is_main_process:
                        self._save_checkpoint(model, f"checkpoint-step-{global_step}")

            # Save epoch checkpoint
            if self.accelerator.is_main_process:
                self._save_checkpoint(model, f"checkpoint-epoch-{epoch}")

            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info("=" * 80)

    def _evaluate(self, model, dataloader):
        """Evaluate model.

        Args:
            model: Model
            dataloader: Validation dataloader

        Returns:
            Average loss
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                speaker_embedding = batch['speaker_embedding']

                output = model(
                    input_ids=input_ids,
                    codec_ids=codec_ids,
                    speaker_embeddings=speaker_embedding,
                    return_loss=True,
                )
                loss = output['loss']

                total_loss += loss.item()
                num_batches += 1

        model.train()

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _save_checkpoint(self, model, checkpoint_name: str):
        """Save checkpoint."""
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
                'hidden_size': self.config.hidden_size,
                'num_hidden_layers': self.config.num_hidden_layers,
                'num_attention_heads': self.config.num_attention_heads,
                'intermediate_size': self.config.intermediate_size,
                'vocab_size': self.config.vocab_size,
                'num_codebooks': self.config.num_codebooks,
                'codebook_size': self.config.codebook_size,
            }, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Talker LM")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--speaker_embeddings_dir", type=str, required=True)
    parser.add_argument("--audio_codes_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)

    args = parser.parse_args()

    trainer = TalkerLMTrainer(
        output_dir=args.output_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        speaker_embeddings_dir=args.speaker_embeddings_dir,
        audio_codes_dir=args.audio_codes_dir,
        config_path=args.config,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )

    trainer.train()


if __name__ == "__main__":
    main()

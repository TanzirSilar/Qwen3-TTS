# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """Tracks training metrics for logging and analysis."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.sub_talker_losses = []
        self.main_losses = []

    def update(self, loss: float, main_loss: float, sub_loss: float):
        self.losses.append(loss)
        self.main_losses.append(main_loss)
        self.sub_talker_losses.append(sub_loss)

    def get_average(self) -> Dict[str, float]:
        return {
            "avg_loss": np.mean(self.losses) if self.losses else 0,
            "avg_main_loss": np.mean(self.main_losses) if self.main_losses else 0,
            "avg_sub_talker_loss": np.mean(self.sub_talker_losses) if self.sub_talker_losses else 0,
        }


class Qwen3TTSTrainer:
    """Main training orchestrator for Qwen3-TTS model."""

    def __init__(
        self,
        init_model_path: str,
        output_dir: str,
        train_jsonl: str,
        val_jsonl: Optional[str] = None,
        batch_size: int = 2,
        lr: float = 2e-5,
        num_epochs: int = 3,
        speaker_name: str = "speaker_test",
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 500,
        gradient_accumulation_steps: int = 4,
    ):
        self.init_model_path = init_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_jsonl = train_jsonl
        self.val_jsonl = val_jsonl
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.speaker_name = speaker_name
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.target_speaker_embedding = None
        self.global_step = 0
        self.best_val_loss = float("inf")

        self._setup_logging()
        self._setup_accelerator()
        self._load_data()
        self._setup_model()

    def _setup_logging(self):
        """Configure logging for training."""
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

    def _setup_accelerator(self):
        """Initialize Accelerate for distributed training."""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision="bf16",
            log_with="tensorboard",
            logging_dir=str(self.output_dir / "logs"),
        )
        logger.info(f"Accelerator initialized: {self.accelerator.device}")

    def _load_data(self):
        """Load training and validation datasets."""
        logger.info(f"Loading training data from {self.train_jsonl}")
        train_data = [json.loads(line) for line in open(self.train_jsonl)]
        logger.info(f"Loaded {len(train_data)} training samples")

        self.config = AutoConfig.from_pretrained(self.init_model_path)
        self.train_dataset = TTSDataset(train_data, None, self.config)

        if self.val_jsonl:
            logger.info(f"Loading validation data from {self.val_jsonl}")
            val_data = [json.loads(line) for line in open(self.val_jsonl)]
            logger.info(f"Loaded {len(val_data)} validation samples")
            self.val_dataset = TTSDataset(val_data, None, self.config)
        else:
            self.val_dataset = None

    def _setup_model(self):
        """Load and prepare model for training."""
        logger.info(f"Loading model from {self.init_model_path}")
        self.model_wrapper = Qwen3TTSModel.from_pretrained(
            self.init_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = self.model_wrapper.processor
        self.train_dataset.processor = self.processor
        if self.val_dataset:
            self.val_dataset.processor = self.processor

        self.model = self.model_wrapper.model
        self.model.train()
        logger.info("Model loaded and set to training mode")

    def _setup_dataloaders(self):
        """Create training and validation dataloaders."""
        logger.info("Creating dataloaders")
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

        if self.val_dataset:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.val_dataset.collate_fn,
            )
        else:
            self.val_dataloader = None

        logger.info(f"Train batches: {len(self.train_dataloader)}")
        if self.val_dataloader:
            logger.info(f"Val batches: {len(self.val_dataloader)}")

    def _setup_optimizer_and_scheduler(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

        total_steps = len(self.train_dataloader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.0, total_iters=warmup_steps),
                CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps),
            ],
            milestones=[warmup_steps],
        )

        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        if self.val_dataloader:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        self.lr_scheduler = lr_scheduler
        logger.info(f"Optimizer and scheduler initialized (warmup_steps={warmup_steps})")

    def _training_step(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single training step."""
        input_ids = batch["input_ids"]
        codec_ids = batch["codec_ids"]
        ref_mels = batch["ref_mels"]
        text_embedding_mask = batch["text_embedding_mask"]
        codec_embedding_mask = batch["codec_embedding_mask"]
        attention_mask = batch["attention_mask"]
        codec_0_labels = batch["codec_0_labels"]
        codec_mask = batch["codec_mask"]

        speaker_embedding = self.model.speaker_encoder(
            ref_mels.to(self.model.device).to(self.model.dtype)
        ).detach()

        if self.target_speaker_embedding is None:
            self.target_speaker_embedding = speaker_embedding

        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]

        input_text_embedding = (
            self.model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        )
        input_codec_embedding = (
            self.model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        )
        input_codec_embedding[:, 6, :] = speaker_embedding

        input_embeddings = input_text_embedding + input_codec_embedding

        for i in range(1, 16):
            codec_i_embedding = self.model.talker.code_predictor.get_input_embeddings()[
                i - 1
            ](codec_ids[:, :, i])
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding

        outputs = self.model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[0][-1]
        talker_hidden_states = hidden_states[codec_mask[:, :-1]]
        talker_codec_ids = codec_ids[codec_mask]

        sub_talker_logits, sub_talker_loss = self.model.talker.forward_sub_talker_finetune(
            talker_codec_ids, talker_hidden_states
        )

        main_loss = outputs.loss
        loss = main_loss + 0.3 * sub_talker_loss

        return loss, main_loss, sub_talker_loss

    @torch.no_grad()
    def _validation_step(self) -> float:
        """Perform validation on the validation set."""
        if self.val_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            loss, _, _ = self._training_step(batch)
            total_loss += self.accelerator.gather(loss.detach()).mean().item()
            num_batches += 1

        self.model.train()
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_val_loss

    def _save_checkpoint(self, checkpoint_dir: Path, is_best: bool = False):
        """Save model checkpoint with config updates."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        shutil.copytree(self.init_model_path, checkpoint_dir, dirs_exist_ok=True)

        input_config_file = Path(self.init_model_path) / "config.json"
        output_config_file = checkpoint_dir / "config.json"
        with open(input_config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {self.speaker_name: 3000}
        talker_config["spk_is_dialect"] = {self.speaker_name: False}
        config_dict["talker_config"] = talker_config

        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

        drop_prefix = "speaker_encoder"
        keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
        for k in keys_to_drop:
            del state_dict[k]

        weight = state_dict["talker.model.codec_embedding.weight"]
        state_dict["talker.model.codec_embedding.weight"][3000] = (
            self.target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        )

        save_path = checkpoint_dir / "model.safetensors"
        save_file(state_dict, str(save_path))

        if is_best:
            best_link = self.output_dir / "best_model"
            if best_link.exists():
                best_link.unlink()
            best_link.symlink_to(checkpoint_dir)
            logger.info(f"Saved best model to {best_link}")

    def _save_training_state(self):
        """Save training state for resumption."""
        state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        with open(self.output_dir / "training_state.json", "w") as f:
            json.dump(state, f)

    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info("Starting Qwen3-TTS Training")
        logger.info("=" * 80)
        logger.info(f"Model: {self.init_model_path}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Val samples: {len(self.val_dataset)}")
        logger.info(f"Batch size: {self.batch_size} (effective: {self.batch_size * self.gradient_accumulation_steps})")
        logger.info(f"Learning rate: {self.lr}")
        logger.info(f"Num epochs: {self.num_epochs}")
        logger.info("=" * 80)

        self._setup_dataloaders()
        self._setup_optimizer_and_scheduler()

        metrics = TrainingMetrics()

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch}/{self.num_epochs - 1}")
            metrics.reset()
            epoch_loss = 0.0

            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    loss, main_loss, sub_talker_loss = self._training_step(batch)
                    metrics.update(loss.item(), main_loss.item(), sub_talker_loss.item())

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                if step % 10 == 0 and self.accelerator.is_main_process:
                    avg_metrics = metrics.get_average()
                    logger.info(
                        f"  Step {step} | Loss: {avg_metrics['avg_loss']:.4f} | "
                        f"Main: {avg_metrics['avg_main_loss']:.4f} | "
                        f"Sub: {avg_metrics['avg_sub_talker_loss']:.4f} | "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                    )

                if self.save_steps > 0 and step % self.save_steps == 0 and self.global_step > 0:
                    if self.accelerator.is_main_process:
                        checkpoint_dir = self.output_dir / f"checkpoint-step-{self.global_step}"
                        self._save_checkpoint(checkpoint_dir)

                if self.eval_steps > 0 and step % self.eval_steps == 0:
                    val_loss = self._validation_step()
                    if val_loss is not None and self.accelerator.is_main_process:
                        logger.info(f"  Validation loss: {val_loss:.4f}")
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            best_checkpoint = self.output_dir / f"checkpoint-step-{self.global_step}-best"
                            self._save_checkpoint(best_checkpoint, is_best=True)

            if self.accelerator.is_main_process:
                checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
                self._save_checkpoint(checkpoint_dir)
                self._save_training_state()

                avg_metrics = metrics.get_average()
                logger.info(f"Epoch {epoch} finished - Avg Loss: {avg_metrics['avg_loss']:.4f}")

        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Best model saved to: {self.output_dir / 'best_model'}")
        logger.info(f"All checkpoints saved to: {self.output_dir}")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Fine-tuning Script")
    parser.add_argument(
        "--init_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Path to initial model",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val_jsonl",
        type=str,
        default=None,
        help="Path to validation JSONL file (optional)",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="speaker_test",
        help="Speaker name for custom voice",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (0 to disable)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (0 to disable)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()

    trainer = Qwen3TTSTrainer(
        init_model_path=args.init_model_path,
        output_dir=args.output_model_path,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        speaker_name=args.speaker_name,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    trainer.train()


if __name__ == "__main__":
    main()

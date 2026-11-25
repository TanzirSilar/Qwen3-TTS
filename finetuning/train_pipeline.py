#!/usr/bin/env python3
# coding=utf-8
"""
Complete training pipeline for Qwen3-TTS.
Handles data preparation, training, and inference in one script.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class Qwen3TTSPipeline:
    """Orchestrates the complete Qwen3-TTS training pipeline."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def prepare_data(
        self,
        input_jsonl: str,
        output_jsonl: Optional[str] = None,
        device: str = "cuda:0",
    ) -> str:
        """
        Prepare training data by extracting audio codes.

        Args:
            input_jsonl: Path to raw JSONL file
            output_jsonl: Path to output JSONL with codes
            device: Device to use for tokenizer

        Returns:
            Path to output JSONL file
        """
        if output_jsonl is None:
            output_jsonl = input_jsonl.replace(".jsonl", "_with_codes.jsonl")

        logger.info(f"Preparing data: {input_jsonl} -> {output_jsonl}")

        cmd = [
            "python",
            "prepare_data.py",
            "--device",
            device,
            "--tokenizer_model_path",
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "--input_jsonl",
            input_jsonl,
            "--output_jsonl",
            output_jsonl,
        ]

        result = subprocess.run(cmd, check=True)
        logger.info(f"Data preparation completed: {output_jsonl}")
        return output_jsonl

    def train(
        self,
        train_jsonl: str,
        val_jsonl: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> str:
        """
        Run training with the specified configuration.

        Args:
            train_jsonl: Path to training JSONL
            val_jsonl: Path to validation JSONL (optional)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Path to output directory
        """
        logger.info("Starting training...")

        cmd = [
            "python",
            "train_qwen3_tts.py",
            "--init_model_path",
            self.config["init_model_path"],
            "--output_model_path",
            self.config["output_model_path"],
            "--train_jsonl",
            train_jsonl,
            "--batch_size",
            str(self.config["batch_size"]),
            "--lr",
            str(self.config["learning_rate"]),
            "--num_epochs",
            str(self.config["num_epochs"]),
            "--speaker_name",
            self.config["speaker_name"],
            "--warmup_ratio",
            str(self.config["warmup_ratio"]),
            "--max_grad_norm",
            str(self.config["max_grad_norm"]),
            "--save_steps",
            str(self.config["save_steps"]),
            "--eval_steps",
            str(self.config["eval_steps"]),
            "--gradient_accumulation_steps",
            str(self.config["gradient_accumulation_steps"]),
        ]

        if val_jsonl:
            cmd.extend(["--val_jsonl", val_jsonl])

        result = subprocess.run(cmd, check=True)
        logger.info(f"Training completed. Output: {self.config['output_model_path']}")
        return self.config["output_model_path"]

    def infer(
        self,
        checkpoint_dir: str,
        text: str,
        output_wav: str = "output.wav",
    ) -> str:
        """
        Run inference with trained model.

        Args:
            checkpoint_dir: Path to checkpoint directory
            text: Text to synthesize
            output_wav: Output wav file path

        Returns:
            Path to output wav file
        """
        logger.info(f"Running inference: {text}")

        inference_script = """
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "{checkpoint}",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="{text}",
    speaker="{speaker}",
)
sf.write("{output}", wavs[0], sr)
print(f"Output saved to {output}")
""".format(
            checkpoint=checkpoint_dir,
            text=text,
            speaker=self.config["speaker_name"],
            output=output_wav,
        )

        result = subprocess.run(
            ["python", "-c", inference_script],
            check=True,
        )
        logger.info(f"Inference completed: {output_wav}")
        return output_wav

    def run_full_pipeline(
        self,
        raw_data_jsonl: str,
        prepared_data_jsonl: Optional[str] = None,
        val_jsonl: Optional[str] = None,
        test_text: Optional[str] = None,
    ):
        """
        Run the complete pipeline: prepare data -> train -> test inference.

        Args:
            raw_data_jsonl: Path to raw training data
            prepared_data_jsonl: Path to prepared data output
            val_jsonl: Path to validation data
            test_text: Text to synthesize after training for testing
        """
        logger.info("=" * 80)
        logger.info("QWEN3-TTS COMPLETE TRAINING PIPELINE")
        logger.info("=" * 80)

        logger.info("Step 1: Preparing data...")
        if prepared_data_jsonl is None:
            prepared_data_jsonl = raw_data_jsonl.replace(".jsonl", "_with_codes.jsonl")
        prepared_path = self.prepare_data(raw_data_jsonl, prepared_data_jsonl)

        logger.info("Step 2: Training model...")
        output_dir = self.train(prepared_path, val_jsonl)

        if test_text:
            logger.info("Step 3: Testing inference...")
            checkpoint = Path(output_dir) / "best_model"
            if not checkpoint.exists():
                checkpoint = Path(output_dir) / f"checkpoint-epoch-{self.config['num_epochs'] - 1}"

            if checkpoint.exists():
                self.infer(str(checkpoint), test_text)
            else:
                logger.warning(f"No checkpoint found at {output_dir}")

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED!")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data only
  %(prog)s --config train_config.yaml --mode prepare --raw_data train_raw.jsonl

  # Train only (data must be prepared)
  %(prog)s --config train_config.yaml --mode train --train_data train_with_codes.jsonl

  # Run full pipeline
  %(prog)s --config train_config.yaml --mode full --raw_data train_raw.jsonl \\
           --test_text "She said she would be here by noon."
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prepare", "train", "infer", "full"],
        default="train",
        help="Pipeline mode: prepare data, train, infer, or run full pipeline",
    )
    parser.add_argument(
        "--raw_data",
        type=str,
        help="Path to raw JSONL data for preparation",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        help="Path to prepared training JSONL data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        help="Path to validation JSONL data",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for inference",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text for inference",
    )
    parser.add_argument(
        "--output_wav",
        type=str,
        default="output.wav",
        help="Output wav file path",
    )
    parser.add_argument(
        "--test_text",
        type=str,
        help="Text to test after training in full pipeline mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for data preparation",
    )

    args = parser.parse_args()

    pipeline = Qwen3TTSPipeline(args.config)

    if args.mode == "prepare":
        if not args.raw_data:
            parser.error("--raw_data is required for prepare mode")
        pipeline.prepare_data(args.raw_data, device=args.device)

    elif args.mode == "train":
        if not args.train_data:
            parser.error("--train_data is required for train mode")
        pipeline.train(args.train_data, args.val_data)

    elif args.mode == "infer":
        if not args.checkpoint or not args.text:
            parser.error("--checkpoint and --text are required for infer mode")
        pipeline.infer(args.checkpoint, args.text, args.output_wav)

    elif args.mode == "full":
        if not args.raw_data:
            parser.error("--raw_data is required for full mode")
        pipeline.run_full_pipeline(
            args.raw_data,
            args.train_data,
            args.val_data,
            args.test_text,
        )


if __name__ == "__main__":
    main()

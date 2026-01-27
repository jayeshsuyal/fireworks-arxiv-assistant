"""
Train reinforcement fine-tuned (RFT) model using Fireworks AI.
Trains on preference pairs (chosen vs rejected responses).
"""
import json
import argparse
from pathlib import Path
import logging
import time
from typing import Optional

from utils.fireworks_client import FireworksClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_preference_data(filepath: Path) -> bool:
    """Validate preference training data format."""
    logger.info(f"Validating preference data: {filepath}")

    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                if 'chosen' not in data or 'rejected' not in data:
                    raise ValueError(f"Line {i}: Missing 'chosen' or 'rejected' field")

                for key in ['chosen', 'rejected']:
                    messages = data[key]
                    if not isinstance(messages, list) or len(messages) < 2:
                        raise ValueError(f"Line {i}: '{key}' must be a list with at least 2 messages")

            except json.JSONDecodeError:
                raise ValueError(f"Line {i}: Invalid JSON")

    logger.info("✓ Preference data validated successfully")
    return True


def train_rft_model(
    training_file: Path,
    base_model: str,  # Should be the SFT model ID
    base_model_file: Optional[Path] = None,
    output_dir: Path = Path("data/models"),
    model_suffix: str = "arxiv-rft",
    n_epochs: int = 3,
    learning_rate: float = 5e-6,
    batch_size: int = 1,
    wait_for_completion: bool = True
) -> Optional[str]:
    """
    Train RFT model on top of SFT model.

    Args:
        training_file: Path to preference pairs JSONL file
        base_model: SFT model ID to train from
        base_model_file: Path to file containing SFT model ID
        output_dir: Directory to save model ID
        model_suffix: Suffix for model name
        n_epochs: Number of training epochs
        learning_rate: Learning rate (typically lower than SFT)
        batch_size: Batch size
        wait_for_completion: Whether to wait for training to complete

    Returns:
        Fine-tuned model ID (if wait_for_completion=True)
    """
    # Get base model ID
    if base_model_file:
        if not base_model_file.exists():
            raise FileNotFoundError(f"Base model file not found: {base_model_file}")
        with open(base_model_file, 'r') as f:
            base_model = f.read().strip()
        logger.info(f"Loaded SFT model ID from {base_model_file}: {base_model}")

    # Validate training data
    validate_preference_data(training_file)

    # Initialize client
    client = FireworksClient()

    # Upload dataset
    logger.info(f"Uploading preference dataset from {training_file}...")
    dataset_id = client.upload_dataset(training_file)
    logger.info(f"✓ Dataset uploaded: {dataset_id}")

    # Prepare hyperparameters
    hyperparameters = {
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }

    logger.info(f"Creating RFT fine-tuning job...")
    logger.info(f"  Base model (SFT): {base_model}")
    logger.info(f"  Hyperparameters: {hyperparameters}")
    logger.info(f"  Training type: Preference-based (RFT/DPO)")

    # Create fine-tuning job
    job_id = client.create_fine_tune_job(
        training_file_id=dataset_id,
        base_model=base_model,
        hyperparameters=hyperparameters,
        suffix=model_suffix
    )

    logger.info(f"✓ RFT fine-tuning job created: {job_id}")

    # Save job ID
    output_dir.mkdir(parents=True, exist_ok=True)
    job_file = output_dir / "rft_job_id.txt"
    with open(job_file, 'w') as f:
        f.write(job_id)
    logger.info(f"Job ID saved to {job_file}")

    if not wait_for_completion:
        logger.info("\nRFT fine-tuning job started. Check status with:")
        logger.info(f"  Job ID: {job_id}")
        logger.info(f"  Estimated time: 30-60 minutes for {n_epochs} epochs")
        return None

    # Wait for completion
    logger.info("\nWaiting for RFT fine-tuning to complete...")
    logger.info("This may take 30-60 minutes. You can safely exit and check status later.")
    logger.info("-" * 70)

    try:
        model_id = client.wait_for_fine_tune(job_id, poll_interval=60)

        # Save model ID
        model_file = output_dir / "rft_model_id.txt"
        with open(model_file, 'w') as f:
            f.write(model_id)
        logger.info(f"Model ID saved to {model_file}")

        # Get cost estimate
        cost = client.get_cost_estimate()

        logger.info("\n" + "=" * 70)
        logger.info("RFT TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Base Model (SFT): {base_model}")
        logger.info(f"Training File: {training_file}")
        logger.info(f"Epochs: {n_epochs}")
        logger.info(f"\nEstimated Cost: ${cost['total_cost_usd']:.4f}")
        logger.info("=" * 70)

        return model_id

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user. Fine-tuning continues in background.")
        logger.info(f"Check status later with job ID: {job_id}")
        return None
    except Exception as e:
        logger.error(f"\nError during fine-tuning: {e}")
        logger.info(f"Job ID for debugging: {job_id}")
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Train RFT model using Fireworks AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--training-file',
        type=Path,
        default=Path('data/rft_training.jsonl'),
        help='Path to preference pairs JSONL file'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        help='SFT model ID to train from'
    )
    parser.add_argument(
        '--base-model-file',
        type=Path,
        default=Path('data/models/sft_model_id.txt'),
        help='Path to file containing SFT model ID'
    )
    parser.add_argument(
        '--model-suffix',
        type=str,
        default='arxiv-rft',
        help='Suffix for model name'
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-6,
        help='Learning rate (typically lower than SFT)'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        help='Do not wait for training to complete'
    )

    args = parser.parse_args()

    # Check training file
    if not args.training_file.exists():
        logger.error(f"Training file not found: {args.training_file}")
        logger.info("Run generate_preference_data.py first to create preference data")
        return

    # Check base model
    if not args.base_model and not args.base_model_file.exists():
        logger.error(f"Base model file not found: {args.base_model_file}")
        logger.info("Train SFT model first with train_sft.py")
        return

    # Train model
    train_rft_model(
        training_file=args.training_file,
        base_model=args.base_model or "",
        base_model_file=args.base_model_file if not args.base_model else None,
        model_suffix=args.model_suffix,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        wait_for_completion=not args.no_wait
    )


if __name__ == "__main__":
    main()

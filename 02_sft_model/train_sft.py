"""
Train supervised fine-tuned (SFT) model using Fireworks AI.
Uploads training data and creates fine-tuning job.
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


def validate_training_data(filepath: Path) -> bool:
    """
    Validate training data format.

    Args:
        filepath: Path to training JSONL file

    Returns:
        True if valid, raises ValueError otherwise
    """
    logger.info(f"Validating training data: {filepath}")

    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                # Check for required format
                if 'messages' not in data:
                    raise ValueError(f"Line {i}: Missing 'messages' field")

                messages = data['messages']
                if not isinstance(messages, list):
                    raise ValueError(f"Line {i}: 'messages' must be a list")

                if len(messages) < 2:
                    raise ValueError(f"Line {i}: Need at least 2 messages (user + assistant)")

                # Check message format
                for msg in messages:
                    if 'role' not in msg or 'content' not in msg:
                        raise ValueError(f"Line {i}: Messages must have 'role' and 'content'")

            except json.JSONDecodeError:
                raise ValueError(f"Line {i}: Invalid JSON")

    logger.info(f"✓ Training data validated successfully")
    return True


def train_sft_model(
    training_file: Path,
    base_model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
    output_dir: Path = Path("data/models"),
    model_suffix: str = "arxiv-sft",
    n_epochs: int = 3,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    wait_for_completion: bool = True
) -> Optional[str]:
    """
    Train SFT model on Fireworks.

    Args:
        training_file: Path to training JSONL file
        base_model: Base model to fine-tune
        output_dir: Directory to save model ID
        model_suffix: Suffix for model name
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        wait_for_completion: Whether to wait for training to complete

    Returns:
        Fine-tuned model ID (if wait_for_completion=True)
    """
    # Validate training data
    validate_training_data(training_file)

    # Initialize client
    client = FireworksClient()

    # Upload dataset
    logger.info(f"Uploading training dataset from {training_file}...")
    dataset_id = client.upload_dataset(training_file)
    logger.info(f"✓ Dataset uploaded: {dataset_id}")

    # Prepare hyperparameters
    hyperparameters = {
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }

    logger.info(f"Creating fine-tuning job...")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Hyperparameters: {hyperparameters}")

    # Create fine-tuning job
    job_id = client.create_fine_tune_job(
        training_file_id=dataset_id,
        base_model=base_model,
        hyperparameters=hyperparameters,
        suffix=model_suffix
    )

    logger.info(f"✓ Fine-tuning job created: {job_id}")

    # Save job ID
    output_dir.mkdir(parents=True, exist_ok=True)
    job_file = output_dir / "sft_job_id.txt"
    with open(job_file, 'w') as f:
        f.write(job_id)
    logger.info(f"Job ID saved to {job_file}")

    if not wait_for_completion:
        logger.info("\nFine-tuning job started. Check status with:")
        logger.info(f"  Job ID: {job_id}")
        logger.info(f"  Estimated time: 30-60 minutes for {n_epochs} epochs")
        return None

    # Wait for completion
    logger.info("\nWaiting for fine-tuning to complete...")
    logger.info("This may take 30-60 minutes. You can safely exit and check status later.")
    logger.info("-" * 70)

    try:
        model_id = client.wait_for_fine_tune(job_id, poll_interval=60)

        # Save model ID
        model_file = output_dir / "sft_model_id.txt"
        with open(model_file, 'w') as f:
            f.write(model_id)
        logger.info(f"Model ID saved to {model_file}")

        # Get cost estimate
        cost = client.get_cost_estimate()

        logger.info("\n" + "=" * 70)
        logger.info("SFT TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Base Model: {base_model}")
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


def check_job_status(job_id: str):
    """Check the status of a fine-tuning job."""
    client = FireworksClient()
    status = client.get_fine_tune_status(job_id)

    logger.info("\n" + "=" * 70)
    logger.info("FINE-TUNING JOB STATUS")
    logger.info("=" * 70)
    logger.info(f"Job ID: {status['id']}")
    logger.info(f"Status: {status['status']}")
    logger.info(f"Base Model: {status['model']}")

    if status['fine_tuned_model']:
        logger.info(f"Fine-tuned Model: {status['fine_tuned_model']}")

    if status['created_at']:
        logger.info(f"Created: {time.ctime(status['created_at'])}")

    if status['finished_at']:
        logger.info(f"Finished: {time.ctime(status['finished_at'])}")

    if status['error']:
        logger.info(f"Error: {status['error']}")

    logger.info("=" * 70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Train SFT model using Fireworks AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Start fine-tuning job')
    train_parser.add_argument(
        '--training-file',
        type=Path,
        default=Path('data/sft_training.jsonl'),
        help='Path to training JSONL file'
    )
    train_parser.add_argument(
        '--base-model',
        type=str,
        default='accounts/fireworks/models/llama-v3p1-8b-instruct',
        help='Base model to fine-tune'
    )
    train_parser.add_argument(
        '--model-suffix',
        type=str,
        default='arxiv-sft',
        help='Suffix for model name'
    )
    train_parser.add_argument(
        '--n-epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size'
    )
    train_parser.add_argument(
        '--no-wait',
        action='store_true',
        help='Do not wait for training to complete'
    )

    # Status command
    status_parser = subparsers.add_parser('status', help='Check job status')
    status_parser.add_argument(
        'job_id',
        type=str,
        help='Fine-tuning job ID'
    )

    args = parser.parse_args()

    if args.command == 'train':
        # Check training file
        if not args.training_file.exists():
            logger.error(f"Training file not found: {args.training_file}")
            logger.info("Run generate_training_data.py first to create training data")
            return

        # Train model
        train_sft_model(
            training_file=args.training_file,
            base_model=args.base_model,
            model_suffix=args.model_suffix,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            wait_for_completion=not args.no_wait
        )

    elif args.command == 'status':
        # Check status
        check_job_status(args.job_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
Fireworks AI API wrapper.
Provides a unified interface for chat completions and fine-tuning operations.
"""
import os
import time
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FireworksClient:
    """
    Unified client for Fireworks AI operations.

    Supports:
    - Chat completions
    - Fine-tuning (SFT and RFT)
    - Model management
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        default_model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
        rate_limit_delay: float = 0.5
    ):
        """
        Initialize Fireworks client.

        Args:
            api_key: Fireworks API key (or use FIREWORKS_API_KEY env var)
            base_url: API base URL
            default_model: Default model for completions
            rate_limit_delay: Delay between API calls
        """
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("Fireworks API key required. Set FIREWORKS_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.default_model = default_model
        self.rate_limit_delay = rate_limit_delay

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> tuple[str, Dict[str, Any]]:
        """
        Create a chat completion with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to self.default_model)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional arguments for the API

        Returns:
            Tuple of (response_text, metadata_dict)
        """
        time.sleep(self.rate_limit_delay)

        model = model or self.default_model

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Track usage
        usage = {
            'input_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
            'model': model
        }

        self.total_input_tokens += usage['input_tokens']
        self.total_output_tokens += usage['output_tokens']
        self.total_requests += 1

        response_text = response.choices[0].message.content

        metadata = {
            'usage': usage,
            'finish_reason': response.choices[0].finish_reason,
            'model': model
        }

        return response_text, metadata

    def upload_dataset(
        self,
        file_path: Path,
        dataset_name: Optional[str] = None
    ) -> str:
        """
        Upload a training dataset to Fireworks.

        Args:
            file_path: Path to JSONL training file
            dataset_name: Optional dataset name

        Returns:
            Dataset ID
        """
        logger.info(f"Uploading dataset from {file_path}...")

        with open(file_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )

        dataset_id = response.id
        logger.info(f"Dataset uploaded successfully: {dataset_id}")

        return dataset_id

    def create_fine_tune_job(
        self,
        training_file_id: str,
        base_model: str,
        hyperparameters: Optional[Dict] = None,
        suffix: Optional[str] = None
    ) -> str:
        """
        Create a fine-tuning job.

        Args:
            training_file_id: ID of uploaded training file
            base_model: Base model to fine-tune
            hyperparameters: Training hyperparameters
            suffix: Model name suffix

        Returns:
            Fine-tune job ID
        """
        logger.info(f"Creating fine-tune job on base model: {base_model}")

        hyperparameters = hyperparameters or {}

        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=base_model,
            hyperparameters=hyperparameters,
            suffix=suffix
        )

        job_id = response.id
        logger.info(f"Fine-tune job created: {job_id}")

        return job_id

    def get_fine_tune_status(self, job_id: str) -> Dict:
        """
        Get the status of a fine-tuning job.

        Args:
            job_id: Fine-tune job ID

        Returns:
            Job status dict
        """
        response = self.client.fine_tuning.jobs.retrieve(job_id)

        return {
            'id': response.id,
            'status': response.status,
            'model': response.model,
            'fine_tuned_model': response.fine_tuned_model,
            'created_at': response.created_at,
            'finished_at': response.finished_at,
            'error': response.error if hasattr(response, 'error') else None
        }

    def wait_for_fine_tune(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: int = 7200  # 2 hours
    ) -> str:
        """
        Wait for a fine-tuning job to complete.

        Args:
            job_id: Fine-tune job ID
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds

        Returns:
            Fine-tuned model ID
        """
        logger.info(f"Waiting for fine-tune job {job_id} to complete...")
        logger.info(f"This may take up to {timeout//60} minutes...")

        start_time = time.time()

        while True:
            status = self.get_fine_tune_status(job_id)

            logger.info(f"Status: {status['status']}")

            if status['status'] == 'succeeded':
                logger.info(f"Fine-tuning complete! Model: {status['fine_tuned_model']}")
                return status['fine_tuned_model']

            elif status['status'] in ['failed', 'cancelled']:
                error_msg = status.get('error', 'Unknown error')
                raise RuntimeError(f"Fine-tuning failed: {error_msg}")

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Fine-tuning timeout after {timeout} seconds")

            # Wait before next check
            time.sleep(poll_interval)

    def list_models(self, limit: int = 20) -> List[Dict]:
        """
        List available models.

        Args:
            limit: Maximum number of models to return

        Returns:
            List of model dicts
        """
        response = self.client.models.list()

        models = []
        for model in response.data[:limit]:
            models.append({
                'id': model.id,
                'created': model.created,
                'owned_by': model.owned_by
            })

        return models

    def get_cost_estimate(
        self,
        input_cost_per_1m: float = 0.90,
        output_cost_per_1m: float = 0.90
    ) -> Dict:
        """
        Get cost estimate for API usage.

        Args:
            input_cost_per_1m: Cost per 1M input tokens
            output_cost_per_1m: Cost per 1M output tokens

        Returns:
            Cost estimate dict
        """
        input_cost = (self.total_input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (self.total_output_tokens / 1_000_000) * output_cost_per_1m
        total_cost = input_cost + output_cost

        return {
            'total_requests': self.total_requests,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost_usd': round(input_cost, 4),
            'output_cost_usd': round(output_cost, 4),
            'total_cost_usd': round(total_cost, 4)
        }

    def reset_cost_tracking(self):
        """Reset cost tracking counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0


# Convenience function
def get_client(model: Optional[str] = None) -> FireworksClient:
    """
    Get a configured Fireworks client.

    Args:
        model: Optional default model

    Returns:
        FireworksClient instance
    """
    return FireworksClient(
        default_model=model or "accounts/fireworks/models/llama-v3p3-70b-instruct"
    )

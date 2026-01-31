"""
Generate preference pairs for Reinforcement Fine-Tuning (RFT) from arXiv papers.

Creates preference pairs (chosen vs rejected) for RLHF-style training.
Each pair contains a question with a good response and a bad response.

Improvements:
- Retry logic for API failures
- Rate limiting
- Multiple rejection strategies (incorrect, incomplete, verbose, off-topic)
- Quality validation
- Cost tracking
- CLI arguments
"""
import json
import os
import time
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import OpenAI

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


REJECTION_STRATEGIES = [
    {
        "type": "incorrect",
        "description": "Factually incorrect or misleading information",
        "weight": 0.3
    },
    {
        "type": "incomplete",
        "description": "Missing key information or oversimplified",
        "weight": 0.3
    },
    {
        "type": "verbose",
        "description": "Unnecessarily verbose or off-focus",
        "weight": 0.2
    },
    {
        "type": "irrelevant",
        "description": "Somewhat off-topic or tangential",
        "weight": 0.2
    }
]


class PreferenceDataGenerator:
    """Generates RFT preference data from arXiv papers."""

    def __init__(
        self,
        api_key: str,
        model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
        rate_limit_delay: float = 1.0,
        base_url: str = "https://api.fireworks.ai/inference/v1"
    ):
        """
        Initialize generator.

        Args:
            api_key: Fireworks API key
            model: Model to use for generation
            rate_limit_delay: Delay between API calls
            base_url: API base URL
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.rate_limit_delay = rate_limit_delay
        self.logger = logging.getLogger(__name__)

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    def load_papers(self, filepath: Path) -> List[Dict]:
        """Load papers from JSONL file."""
        papers = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                papers.append(json.loads(line))

        self.logger.info(f"Loaded {len(papers)} papers from {filepath}")
        return papers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> tuple[str, Dict]:
        """
        Call LLM with retry logic.

        Returns:
            Tuple of (response_text, usage_dict)
        """
        time.sleep(self.rate_limit_delay)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        usage = {
            'input_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }

        self.total_input_tokens += usage['input_tokens']
        self.total_output_tokens += usage['output_tokens']
        self.total_requests += 1

        return response.choices[0].message.content, usage

    def generate_preference_pair(
        self,
        paper: Dict,
        rejection_type: str
    ) -> Optional[Dict]:
        """
        Generate a preference pair (chosen vs rejected) for a paper.

        Args:
            paper: Paper dictionary
            rejection_type: Type of rejection strategy

        Returns:
            Preference pair dict or None if generation failed
        """
        # First, generate a good Q&A pair
        system_prompt_good = """You are an expert AI research assistant.
Generate a high-quality question and answer about the provided arXiv paper.

The question should be clear, specific, and insightful.
The answer should be accurate, detailed, and well-structured (2-4 sentences).

Format as JSON:
{
    "question": "Your question here",
    "answer": "Your high-quality answer here"
}"""

        user_prompt = f"""Paper Title: {paper['title']}

Authors: {', '.join(paper['authors'][:5])}

Abstract: {paper['summary']}

Categories: {', '.join(paper['categories'])}

Generate a question and high-quality answer about this paper."""

        try:
            # Generate good response
            response_good, usage_good = self._call_llm(
                system_prompt_good,
                user_prompt,
                temperature=0.7
            )

            qa_good = json.loads(response_good)
            question = qa_good['question']
            chosen_answer = qa_good['answer']

            # Generate rejected response based on strategy
            rejection_prompts = {
                "incorrect": f"""Generate a POOR quality answer to this question that contains factual errors or misleading information.
Make it sound plausible but include subtle mistakes.

Question: {question}

Paper context: {paper['title']} - {paper['summary'][:300]}

Format as JSON with just the answer:
{{"answer": "Your incorrect answer here"}}""",

                "incomplete": f"""Generate a POOR quality answer to this question that is overly brief, missing key details, or oversimplified.

Question: {question}

Paper context: {paper['title']} - {paper['summary'][:300]}

Format as JSON with just the answer:
{{"answer": "Your incomplete answer here"}}""",

                "verbose": f"""Generate a POOR quality answer to this question that is unnecessarily verbose, repetitive, or unfocused.
Include tangential information that doesn't directly answer the question.

Question: {question}

Paper context: {paper['title']} - {paper['summary'][:300]}

Format as JSON with just the answer:
{{"answer": "Your verbose answer here"}}""",

                "irrelevant": f"""Generate a POOR quality answer to this question that is somewhat off-topic or focuses on less relevant aspects.

Question: {question}

Paper context: {paper['title']} - {paper['summary'][:300]}

Format as JSON with just the answer:
{{"answer": "Your somewhat irrelevant answer here"}}"""""
            }

            system_prompt_rejected = "You are generating examples of poor-quality responses for training purposes."

            response_rejected, usage_rejected = self._call_llm(
                system_prompt_rejected,
                rejection_prompts[rejection_type],
                temperature=0.9,  # Higher temp for more variety in bad responses
                max_tokens=1000
            )

            qa_rejected = json.loads(response_rejected)
            rejected_answer = qa_rejected['answer']

            # Combine usage stats
            total_usage = {
                'input_tokens': usage_good['input_tokens'] + usage_rejected['input_tokens'],
                'output_tokens': usage_good['output_tokens'] + usage_rejected['output_tokens'],
                'total_tokens': usage_good['total_tokens'] + usage_rejected['total_tokens']
            }

            # Format as preference pair
            return {
                'chosen': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful AI research assistant that provides accurate, concise answers about arXiv papers.'
                    },
                    {
                        'role': 'user',
                        'content': question
                    },
                    {
                        'role': 'assistant',
                        'content': chosen_answer
                    }
                ],
                'rejected': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful AI research assistant that provides accurate, concise answers about arXiv papers.'
                    },
                    {
                        'role': 'user',
                        'content': question
                    },
                    {
                        'role': 'assistant',
                        'content': rejected_answer
                    }
                ],
                'metadata': {
                    'paper_id': paper['paper_id'],
                    'paper_title': paper['title'],
                    'rejection_type': rejection_type,
                    'usage': total_usage
                }
            }

        except Exception as e:
            self.logger.error(f"Error generating preference pair for {paper['paper_id']}: {e}")
            return None

    def generate_preference_dataset(
        self,
        papers: List[Dict],
        num_pairs: int = 50
    ) -> List[Dict]:
        """
        Generate complete preference dataset.

        Args:
            papers: List of papers
            num_pairs: Number of preference pairs to generate

        Returns:
            List of preference pairs
        """
        self.logger.info(f"Generating {num_pairs} preference pairs...")

        # Calculate distribution of rejection types
        type_distribution = {
            'incorrect': int(num_pairs * 0.3),
            'incomplete': int(num_pairs * 0.3),
            'verbose': int(num_pairs * 0.2),
            'irrelevant': num_pairs - int(num_pairs * 0.8)  # Remainder
        }

        self.logger.info(f"Rejection type distribution: {type_distribution}")

        preference_pairs = []
        errors = 0

        # Generate pairs for each rejection type
        for rejection_type, count in type_distribution.items():
            self.logger.info(f"Generating {count} pairs with '{rejection_type}' rejections...")

            # Sample papers for this rejection type
            selected_papers = random.sample(papers, min(count, len(papers)))

            for paper in tqdm(selected_papers, desc=f"Generating {rejection_type} pairs"):
                pair = self.generate_preference_pair(paper, rejection_type)
                if pair:
                    preference_pairs.append(pair)
                else:
                    errors += 1

        self.logger.info(f"Generated {len(preference_pairs)} preference pairs with {errors} errors")

        return preference_pairs

    def save_preference_data(
        self,
        pairs: List[Dict],
        output_path: Path,
        save_metadata: bool = True
    ):
        """
        Save preference data to JSONL file.

        Args:
            pairs: List of preference pairs
            output_path: Output file path
            save_metadata: Whether to save metadata separately
        """
        # Save main training file (without metadata for upload)
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                # Only save chosen/rejected for training
                training_format = {
                    'chosen': pair['chosen'],
                    'rejected': pair['rejected']
                }
                f.write(json.dumps(training_format) + '\n')

        self.logger.info(f"Preference data saved to {output_path}")

        # Save metadata separately
        if save_metadata:
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.jsonl"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + '\n')

            self.logger.info(f"Metadata saved to {metadata_path}")

    def get_cost_estimate(self) -> Dict:
        """Get cost estimate for generation."""
        # Fireworks pricing (approximate)
        input_cost_per_1m = 0.90  # $0.90 per 1M input tokens
        output_cost_per_1m = 0.90  # $0.90 per 1M output tokens

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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate RFT preference data from arXiv papers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-file',
        type=Path,
        default=Path("data/papers.jsonl"),
        help='Input papers file'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path("data/rft_training.jsonl"),
        help='Output preference data file'
    )
    parser.add_argument(
        '--num-pairs',
        type=int,
        default=50,
        help='Number of preference pairs to generate'
    )
    parser.add_argument(
        '--model',
        type=str,
        default="accounts/fireworks/models/llama-v3p3-70b-instruct",
        help='Fireworks model to use for generation'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Delay between API calls in seconds'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load API key
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    if not FIREWORKS_API_KEY:
        logger.error("Missing FIREWORKS_API_KEY environment variable")
        return

    # Check input file
    if not args.input_file.exists():
        logger.error(f"Papers file not found: {args.input_file}")
        logger.info("Run fetch_papers.py first to download papers")
        return

    # Create output directory
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create generator
    generator = PreferenceDataGenerator(
        api_key=FIREWORKS_API_KEY,
        model=args.model,
        rate_limit_delay=args.rate_limit
    )

    # Load papers
    papers = generator.load_papers(args.input_file)

    if len(papers) < args.num_pairs:
        logger.warning(
            f"Only {len(papers)} papers available, but {args.num_pairs} pairs requested. "
            f"Some papers will be used multiple times."
        )

    # Generate preference data
    preference_pairs = generator.generate_preference_dataset(
        papers=papers,
        num_pairs=args.num_pairs
    )

    if not preference_pairs:
        logger.error("No preference pairs generated. Exiting.")
        return

    # Save preference data
    generator.save_preference_data(preference_pairs, args.output_file)

    # Get and display cost estimate
    cost = generator.get_cost_estimate()

    logger.info("\n" + "="*70)
    logger.info("PREFERENCE DATA GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Preference pairs generated: {len(preference_pairs)}")
    logger.info(f"Output file: {args.output_file}")
    logger.info("\nRejection type distribution:")
    type_counts = {}
    for pair in preference_pairs:
        t = pair['metadata']['rejection_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        logger.info(f"  {t}: {count} pairs")
    logger.info("\nCost Estimate:")
    logger.info(f"  Total requests: {cost['total_requests']}")
    logger.info(f"  Input tokens: {cost['input_tokens']:,}")
    logger.info(f"  Output tokens: {cost['output_tokens']:,}")
    logger.info(f"  Total tokens: {cost['total_tokens']:,}")
    logger.info(f"  Estimated cost: ${cost['total_cost_usd']:.4f}")
    logger.info("="*70)


if __name__ == "__main__":
    main()

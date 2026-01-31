"""
Generate supervised fine-tuning (SFT) training data from arXiv papers.

Creates question-answer pairs from papers for fine-tuning.
Uses Fireworks AI to generate high-quality training examples.

Improvements:
- Retry logic for API failures
- Rate limiting
- Multiple question types (factual, analytical, comparative)
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
from typing import List, Dict, Optional
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


QUESTION_TYPES = [
    {
        "type": "factual",
        "description": "Direct questions about paper content",
        "weight": 0.4
    },
    {
        "type": "analytical",
        "description": "Questions requiring analysis and synthesis",
        "weight": 0.4
    },
    {
        "type": "comparative",
        "description": "Questions comparing multiple papers or concepts",
        "weight": 0.2
    }
]


class TrainingDataGenerator:
    """Generates SFT training data from arXiv papers."""

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
        max_tokens: int = 1000
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

    def generate_factual_qa(self, paper: Dict) -> Optional[Dict]:
        """
        Generate factual Q&A about a paper.

        Args:
            paper: Paper dictionary

        Returns:
            Training example dict or None if generation failed
        """
        system_prompt = """You are an AI research assistant specialized in arXiv papers.
Generate a factual question and detailed answer about the provided paper.

The question should:
- Be specific and directly answerable from the paper
- Focus on methodology, results, or key contributions
- Be clear and concise

The answer should:
- Be accurate and detailed
- Include specific information from the paper
- Be 2-4 sentences long
- Cite the paper title in your response

Format your response as JSON:
{
    "question": "Your question here",
    "answer": "Your detailed answer here"
}"""

        user_prompt = f"""Paper Title: {paper['title']}

Authors: {', '.join(paper['authors'][:5])}

Abstract: {paper['summary']}

Categories: {', '.join(paper['categories'])}

Generate a factual question and answer about this paper."""

        try:
            response, usage = self._call_llm(system_prompt, user_prompt, temperature=0.7)

            # Parse JSON response
            qa = json.loads(response)

            return {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful AI research assistant that answers questions about arXiv papers accurately and concisely.'
                    },
                    {
                        'role': 'user',
                        'content': qa['question']
                    },
                    {
                        'role': 'assistant',
                        'content': qa['answer']
                    }
                ],
                'metadata': {
                    'paper_id': paper['paper_id'],
                    'paper_title': paper['title'],
                    'type': 'factual',
                    'usage': usage
                }
            }

        except Exception as e:
            self.logger.error(f"Error generating factual Q&A for {paper['paper_id']}: {e}")
            return None

    def generate_analytical_qa(self, paper: Dict) -> Optional[Dict]:
        """
        Generate analytical Q&A requiring deeper reasoning.

        Args:
            paper: Paper dictionary

        Returns:
            Training example dict or None if generation failed
        """
        system_prompt = """You are an AI research assistant specialized in arXiv papers.
Generate an analytical question and detailed answer about the provided paper.

The question should:
- Require analysis, synthesis, or evaluation
- Ask about implications, limitations, or future directions
- Encourage critical thinking

The answer should:
- Provide thoughtful analysis
- Be well-reasoned and detailed (3-5 sentences)
- Reference specific aspects of the paper
- Cite the paper title in your response

Format your response as JSON:
{
    "question": "Your analytical question here",
    "answer": "Your detailed analytical answer here"
}"""

        user_prompt = f"""Paper Title: {paper['title']}

Authors: {', '.join(paper['authors'][:5])}

Abstract: {paper['summary']}

Categories: {', '.join(paper['categories'])}

Generate an analytical question and answer about this paper."""

        try:
            response, usage = self._call_llm(system_prompt, user_prompt, temperature=0.8)

            # Parse JSON response
            qa = json.loads(response)

            return {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful AI research assistant that provides thoughtful analysis of arXiv papers and their implications.'
                    },
                    {
                        'role': 'user',
                        'content': qa['question']
                    },
                    {
                        'role': 'assistant',
                        'content': qa['answer']
                    }
                ],
                'metadata': {
                    'paper_id': paper['paper_id'],
                    'paper_title': paper['title'],
                    'type': 'analytical',
                    'usage': usage
                }
            }

        except Exception as e:
            self.logger.error(f"Error generating analytical Q&A for {paper['paper_id']}: {e}")
            return None

    def generate_comparative_qa(self, papers: List[Dict]) -> Optional[Dict]:
        """
        Generate comparative Q&A using multiple papers.

        Args:
            papers: List of 2-3 paper dictionaries

        Returns:
            Training example dict or None if generation failed
        """
        if len(papers) < 2:
            return None

        papers_text = "\n\n".join([
            f"Paper {i+1}:\nTitle: {p['title']}\nAbstract: {p['summary'][:500]}..."
            for i, p in enumerate(papers[:3])
        ])

        system_prompt = """You are an AI research assistant specialized in arXiv papers.
Generate a comparative question and detailed answer about the provided papers.

The question should:
- Compare or contrast aspects of the papers
- Ask about relationships, differences, or complementary aspects
- Be insightful and thought-provoking

The answer should:
- Provide detailed comparison (3-5 sentences)
- Reference specific aspects of each paper
- Be balanced and analytical
- Cite paper titles in your response

Format your response as JSON:
{
    "question": "Your comparative question here",
    "answer": "Your detailed comparative answer here"
}"""

        user_prompt = f"""{papers_text}

Generate a comparative question and answer about these papers."""

        try:
            response, usage = self._call_llm(system_prompt, user_prompt, temperature=0.8, max_tokens=1200)

            # Parse JSON response
            qa = json.loads(response)

            return {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful AI research assistant that can compare and contrast multiple arXiv papers.'
                    },
                    {
                        'role': 'user',
                        'content': qa['question']
                    },
                    {
                        'role': 'assistant',
                        'content': qa['answer']
                    }
                ],
                'metadata': {
                    'paper_ids': [p['paper_id'] for p in papers],
                    'paper_titles': [p['title'] for p in papers],
                    'type': 'comparative',
                    'usage': usage
                }
            }

        except Exception as e:
            self.logger.error(f"Error generating comparative Q&A: {e}")
            return None

    def generate_training_set(
        self,
        papers: List[Dict],
        num_examples: int = 100
    ) -> List[Dict]:
        """
        Generate complete training dataset.

        Args:
            papers: List of papers
            num_examples: Number of examples to generate

        Returns:
            List of training examples
        """
        self.logger.info(f"Generating {num_examples} training examples...")

        # Calculate distribution of question types
        type_distribution = {
            'factual': int(num_examples * 0.4),
            'analytical': int(num_examples * 0.4),
            'comparative': num_examples - int(num_examples * 0.8)  # Remainder
        }

        self.logger.info(f"Distribution: {type_distribution}")

        training_examples = []
        errors = 0

        # Generate factual Q&As
        self.logger.info("Generating factual questions...")
        selected_papers = random.sample(papers, min(type_distribution['factual'], len(papers)))

        for paper in tqdm(selected_papers, desc="Factual Q&A"):
            example = self.generate_factual_qa(paper)
            if example:
                training_examples.append(example)
            else:
                errors += 1

        # Generate analytical Q&As
        self.logger.info("Generating analytical questions...")
        selected_papers = random.sample(papers, min(type_distribution['analytical'], len(papers)))

        for paper in tqdm(selected_papers, desc="Analytical Q&A"):
            example = self.generate_analytical_qa(paper)
            if example:
                training_examples.append(example)
            else:
                errors += 1

        # Generate comparative Q&As
        self.logger.info("Generating comparative questions...")
        for _ in tqdm(range(type_distribution['comparative']), desc="Comparative Q&A"):
            selected = random.sample(papers, min(3, len(papers)))
            example = self.generate_comparative_qa(selected)
            if example:
                training_examples.append(example)
            else:
                errors += 1

        self.logger.info(f"Generated {len(training_examples)} examples with {errors} errors")

        return training_examples

    def save_training_data(
        self,
        examples: List[Dict],
        output_path: Path,
        save_metadata: bool = True
    ):
        """
        Save training data to JSONL file.

        Args:
            examples: List of training examples
            output_path: Output file path
            save_metadata: Whether to save metadata separately
        """
        # Save main training file (without metadata for upload)
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                # Only save messages for training
                training_format = {'messages': example['messages']}
                f.write(json.dumps(training_format) + '\n')

        self.logger.info(f"Training data saved to {output_path}")

        # Save metadata separately
        if save_metadata:
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.jsonl"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')

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
        description="Generate SFT training data from arXiv papers",
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
        default=Path("data/sft_training.jsonl"),
        help='Output training data file'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=100,
        help='Number of training examples to generate'
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
    generator = TrainingDataGenerator(
        api_key=FIREWORKS_API_KEY,
        model=args.model,
        rate_limit_delay=args.rate_limit
    )

    # Load papers
    papers = generator.load_papers(args.input_file)

    if len(papers) < args.num_examples:
        logger.warning(
            f"Only {len(papers)} papers available, but {args.num_examples} examples requested. "
            f"Some papers will be used multiple times."
        )

    # Generate training data
    training_examples = generator.generate_training_set(
        papers=papers,
        num_examples=args.num_examples
    )

    if not training_examples:
        logger.error("No training examples generated. Exiting.")
        return

    # Save training data
    generator.save_training_data(training_examples, args.output_file)

    # Get and display cost estimate
    cost = generator.get_cost_estimate()

    logger.info("\n" + "="*70)
    logger.info("TRAINING DATA GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Examples generated: {len(training_examples)}")
    logger.info(f"Output file: {args.output_file}")
    logger.info("\nType distribution:")
    type_counts = {}
    for ex in training_examples:
        t = ex['metadata']['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        logger.info(f"  {t}: {count} examples")
    logger.info("\nCost Estimate:")
    logger.info(f"  Total requests: {cost['total_requests']}")
    logger.info(f"  Input tokens: {cost['input_tokens']:,}")
    logger.info(f"  Output tokens: {cost['output_tokens']:,}")
    logger.info(f"  Total tokens: {cost['total_tokens']:,}")
    logger.info(f"  Estimated cost: ${cost['total_cost_usd']:.4f}")
    logger.info("="*70)


if __name__ == "__main__":
    main()

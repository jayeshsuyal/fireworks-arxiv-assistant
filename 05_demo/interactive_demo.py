"""
Interactive CLI demo for arXiv RAG Assistant.
Customer-facing demo showing all three models.
"""
import argparse
from pathlib import Path
import logging
from typing import Optional

from utils.fireworks_client import FireworksClient
from utils.pinecone_client import PineconeClient

# Import RAG systems
import sys
sys.path.append(str(Path(__file__).parent.parent))

from baseline.base_rag import BaselineRAG
from sft_model.sft_rag import SFTRAG
from rft_model.rft_rag import RFTRAG

logging.basicConfig(level=logging.WARNING)  # Quiet mode for demo
logger = logging.getLogger(__name__)


class InteractiveDemo:
    """Interactive demo for comparing all three RAG systems."""

    def __init__(
        self,
        sft_model_id: Optional[str] = None,
        rft_model_id: Optional[str] = None,
        model_dir: Path = Path("data/models")
    ):
        """
        Initialize demo with all three models.

        Args:
            sft_model_id: SFT model ID (or loaded from file)
            rft_model_id: RFT model ID (or loaded from file)
            model_dir: Directory containing model ID files
        """
        print("Initializing arXiv RAG Assistant Demo...")
        print("Loading models...")

        # Initialize baseline
        self.baseline = BaselineRAG()
        print("✓ Baseline model loaded")

        # Initialize SFT
        if not sft_model_id and (model_dir / "sft_model_id.txt").exists():
            with open(model_dir / "sft_model_id.txt", 'r') as f:
                sft_model_id = f.read().strip()

        if sft_model_id:
            self.sft = SFTRAG(sft_model_id=sft_model_id)
            print("✓ SFT model loaded")
        else:
            self.sft = None
            print("⚠ SFT model not available")

        # Initialize RFT
        if not rft_model_id and (model_dir / "rft_model_id.txt").exists():
            with open(model_dir / "rft_model_id.txt", 'r') as f:
                rft_model_id = f.read().strip()

        if rft_model_id:
            self.rft = RFTRAG(rft_model_id=rft_model_id)
            print("✓ RFT model loaded")
        else:
            self.rft = None
            print("⚠ RFT model not available")

        print("\n" + "=" * 70)

    def query_all(self, question: str):
        """Query all available models and display results."""
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        models = [
            ("Baseline (Vanilla Llama-3.1-8B)", self.baseline),
            ("SFT (Fine-tuned on arXiv Q&A)", self.sft),
            ("RFT (Preference-optimized)", self.rft)
        ]

        for model_name, model in models:
            if model is None:
                continue

            print(f"\n{model_name}")
            print("-" * 70)

            try:
                response, metadata = model.query(question)
                print(response)
                print(f"\n[Latency: {metadata['latency_ms']:.0f}ms | "
                      f"Tokens: {metadata['usage']['total_tokens']}]")
            except Exception as e:
                print(f"Error: {e}")

        print("\n" + "=" * 70)

    def run_interactive(self):
        """Run interactive query loop."""
        print("\n" + "=" * 70)
        print("INTERACTIVE arXiv RAG ASSISTANT")
        print("=" * 70)
        print("\nAvailable models:")
        print("  • Baseline: Vanilla Llama-3.1-8B")
        if self.sft:
            print("  • SFT: Fine-tuned on arXiv Q&A data")
        if self.rft:
            print("  • RFT: Preference-optimized")
        print("\nCommands:")
        print("  • Type your question to query all models")
        print("  • 'examples' to see example questions")
        print("  • 'quit' or 'exit' to end")
        print("=" * 70)

        while True:
            try:
                print()
                user_input = input("Your question: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using arXiv RAG Assistant!")
                    break

                if user_input.lower() in ['examples', 'help']:
                    self.show_examples()
                    continue

                # Query all models
                self.query_all(user_input)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def show_examples(self):
        """Show example questions."""
        print("\n" + "=" * 70)
        print("EXAMPLE QUESTIONS")
        print("=" * 70)
        examples = [
            "What are the latest advances in transformer architectures?",
            "How do retrieval-augmented generation systems work?",
            "What techniques are used for fine-tuning large language models?",
            "Explain the concept of attention mechanisms in neural networks.",
            "What are the current challenges in reinforcement learning?",
        ]

        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")

        print("=" * 70)

    def run_demo_script(self):
        """Run pre-defined demo script with example questions."""
        print("\n" + "=" * 70)
        print("DEMO SCRIPT: Showing Improvements Across Models")
        print("=" * 70)

        demo_questions = [
            "What are retrieval-augmented generation systems?",
            "How do transformer attention mechanisms work?",
            "What are the latest fine-tuning techniques for LLMs?"
        ]

        for i, question in enumerate(demo_questions, 1):
            print(f"\n\n{'#' * 70}")
            print(f"DEMO QUESTION {i}/{len(demo_questions)}")
            print(f"{'#' * 70}")

            self.query_all(question)

            if i < len(demo_questions):
                input("\n[Press Enter to continue to next question...]")

        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nKey Observations:")
        print("  • Baseline provides general AI knowledge")
        print("  • SFT shows better understanding of research papers")
        print("  • RFT provides more focused and relevant responses")
        print("=" * 70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Interactive demo for arXiv RAG Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'demo'],
        default='interactive',
        help='Demo mode: interactive (Q&A) or demo (scripted)'
    )
    parser.add_argument(
        '--sft-model-id',
        type=str,
        help='SFT model ID (overrides file)'
    )
    parser.add_argument(
        '--rft-model-id',
        type=str,
        help='RFT model ID (overrides file)'
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=Path('data/models'),
        help='Directory containing model ID files'
    )

    args = parser.parse_args()

    # Initialize demo
    demo = InteractiveDemo(
        sft_model_id=args.sft_model_id,
        rft_model_id=args.rft_model_id,
        model_dir=args.model_dir
    )

    # Run in selected mode
    if args.mode == 'interactive':
        demo.run_interactive()
    else:
        demo.run_demo_script()


if __name__ == "__main__":
    main()

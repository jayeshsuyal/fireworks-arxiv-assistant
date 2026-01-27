"""
Evaluate SFT RAG system.
Measures performance after supervised fine-tuning.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

from sft_rag import SFTRAG
from utils.metrics import (
    EvaluationResult,
    calculate_response_metrics,
    calculate_latency_metrics,
    calculate_cost_metrics,
    simple_relevance_score,
    calculate_citation_rate,
    format_evaluation_summary
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_queries(filepath: Path) -> List[Dict]:
    """Load test queries from JSON file."""
    with open(filepath, 'r') as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} test queries")
    return queries


def evaluate_sft(
    rag_system: SFTRAG,
    test_queries: List[Dict],
    output_dir: Path
) -> EvaluationResult:
    """Evaluate SFT RAG system."""
    logger.info("Starting SFT evaluation...")

    results = []
    responses = []
    latencies = []
    relevance_scores = []
    total_input_tokens = 0
    total_output_tokens = 0

    for query_data in tqdm(test_queries, desc="Evaluating queries"):
        query = query_data['query']
        expected_keywords = query_data.get('expected_keywords', [])

        try:
            response, metadata = rag_system.query(query, return_context=True)

            responses.append(response)
            latencies.append(metadata['latency_ms'])

            relevance = simple_relevance_score(response, expected_keywords)
            relevance_scores.append(relevance)

            usage = metadata['usage']
            total_input_tokens += usage['input_tokens']
            total_output_tokens += usage['output_tokens']

            result = {
                'query_id': query_data['id'],
                'query': query,
                'response': response,
                'category': query_data.get('category', 'unknown'),
                'latency_ms': metadata['latency_ms'],
                'relevance_score': relevance,
                'usage': usage,
                'context_preview': metadata.get('context', '')[:200] + '...'
            }
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing query {query_data['id']}: {e}")
            results.append({
                'query_id': query_data['id'],
                'query': query,
                'error': str(e)
            })

    # Calculate metrics
    response_metrics = calculate_response_metrics(responses)
    latency_metrics = calculate_latency_metrics(latencies)
    cost_metrics = calculate_cost_metrics(total_input_tokens, total_output_tokens)
    citation_metrics = calculate_citation_rate(responses)
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

    eval_result = EvaluationResult(
        model_name="sft-llama-3.1-8b",
        total_queries=len(test_queries),
        avg_response_length=response_metrics['avg_length'],
        avg_latency_ms=latency_metrics['avg_ms'],
        total_cost_usd=cost_metrics['total_cost_usd'],
        relevance_score=avg_relevance,
        metadata={
            'response_metrics': response_metrics,
            'latency_metrics': latency_metrics,
            'cost_metrics': cost_metrics,
            'citation_metrics': citation_metrics,
            'system_info': rag_system.get_system_info()
        }
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'sft_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    summary_file = output_dir / 'sft_evaluation.json'
    with open(summary_file, 'w') as f:
        json.dump(eval_result.to_dict(), f, indent=2)
    logger.info(f"Summary saved to {summary_file}")

    return eval_result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate SFT RAG system")
    parser.add_argument('--test-queries', type=Path, default=Path('01_baseline/test_queries.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/test_results'))
    parser.add_argument('--model-id-file', type=Path, default=Path('data/models/sft_model_id.txt'))
    parser.add_argument('--model-id', type=str, help='SFT model ID (overrides file)')
    parser.add_argument('--top-k', type=int, default=3)

    args = parser.parse_args()

    if not args.test_queries.exists():
        logger.error(f"Test queries not found: {args.test_queries}")
        return

    # Initialize SFT RAG
    logger.info("Initializing SFT RAG system...")
    rag_system = SFTRAG(
        sft_model_id=args.model_id,
        model_id_file=args.model_id_file if not args.model_id else None,
        top_k=args.top_k
    )

    # Load queries and evaluate
    test_queries = load_test_queries(args.test_queries)
    eval_result = evaluate_sft(rag_system, test_queries, args.output_dir)

    # Print summary
    print("\n" + format_evaluation_summary(eval_result))
    print("\nKey Insights:")
    print(f"  Citation Rate: {eval_result.metadata['citation_metrics']['citation_rate']:.1%}")
    print(f"  P95 Latency: {eval_result.metadata['latency_metrics']['p95_ms']:.0f} ms")
    print(f"  Cost per Query: ${eval_result.total_cost_usd / eval_result.total_queries:.4f}")


if __name__ == "__main__":
    main()

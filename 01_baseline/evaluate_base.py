"""
Evaluate baseline RAG system.
Measures performance before fine-tuning.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

from base_rag import BaselineRAG
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
    logger.info(f"Loaded {len(queries)} test queries from {filepath}")
    return queries


def evaluate_baseline(
    rag_system: BaselineRAG,
    test_queries: List[Dict],
    output_dir: Path
) -> EvaluationResult:
    """
    Evaluate baseline RAG system.

    Args:
        rag_system: BaselineRAG instance
        test_queries: List of test query dicts
        output_dir: Directory to save results

    Returns:
        EvaluationResult object
    """
    logger.info("Starting baseline evaluation...")

    results = []
    responses = []
    latencies = []
    relevance_scores = []
    total_input_tokens = 0
    total_output_tokens = 0

    # Process each query
    for query_data in tqdm(test_queries, desc="Evaluating queries"):
        query = query_data['query']
        expected_keywords = query_data.get('expected_keywords', [])

        try:
            # Query the system
            response, metadata = rag_system.query(query, return_context=True)

            # Collect metrics
            responses.append(response)
            latencies.append(metadata['latency_ms'])

            # Calculate relevance
            relevance = simple_relevance_score(response, expected_keywords)
            relevance_scores.append(relevance)

            # Track token usage
            usage = metadata['usage']
            total_input_tokens += usage['input_tokens']
            total_output_tokens += usage['output_tokens']

            # Store detailed result
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

    # Calculate aggregate metrics
    response_metrics = calculate_response_metrics(responses)
    latency_metrics = calculate_latency_metrics(latencies)
    cost_metrics = calculate_cost_metrics(total_input_tokens, total_output_tokens)
    citation_metrics = calculate_citation_rate(responses)

    # Calculate average relevance
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

    # Create evaluation result
    eval_result = EvaluationResult(
        model_name="baseline-llama-3.1-8b",
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

    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual query results
    results_file = output_dir / 'base_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to {results_file}")

    # Save evaluation summary
    summary_file = output_dir / 'base_evaluation.json'
    with open(summary_file, 'w') as f:
        json.dump(eval_result.to_dict(), f, indent=2)
    logger.info(f"Evaluation summary saved to {summary_file}")

    return eval_result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline RAG system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--test-queries',
        type=Path,
        default=Path('01_baseline/test_queries.json'),
        help='Path to test queries JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/test_results'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='accounts/fireworks/models/llama-v3p3-70b-instruct',
        help='Base model to use'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of papers to retrieve'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Max tokens to generate'
    )

    args = parser.parse_args()

    # Check test queries file
    if not args.test_queries.exists():
        logger.error(f"Test queries file not found: {args.test_queries}")
        return

    # Load test queries
    test_queries = load_test_queries(args.test_queries)

    # Initialize baseline RAG system
    logger.info(f"Initializing baseline RAG with model: {args.base_model}")
    rag_system = BaselineRAG(
        base_model=args.base_model,
        top_k=args.top_k
    )

    # Run evaluation
    eval_result = evaluate_baseline(
        rag_system=rag_system,
        test_queries=test_queries,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + format_evaluation_summary(eval_result))

    # Print additional insights
    print("\nKey Insights:")
    print(f"  Citation Rate: {eval_result.metadata['citation_metrics']['citation_rate']:.1%}")
    print(f"  Avg Citations per Response: {eval_result.metadata['citation_metrics']['avg_citations_per_response']:.2f}")
    print(f"  P95 Latency: {eval_result.metadata['latency_metrics']['p95_ms']:.0f} ms")
    print(f"  Cost per Query: ${eval_result.total_cost_usd / eval_result.total_queries:.4f}")
    print()


if __name__ == "__main__":
    main()

"""
Evaluate RFT RAG system.
Measures final performance after reinforcement fine-tuning.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

from rft_rag import RFTRAG
from utils.metrics import (
    EvaluationResult,
    calculate_response_metrics,
    calculate_latency_metrics,
    calculate_cost_metrics,
    simple_relevance_score,
    calculate_citation_rate,
    format_evaluation_summary
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_queries(filepath: Path) -> List[Dict]:
    """Load test queries from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def evaluate_rft(rag_system: RFTRAG, test_queries: List[Dict], output_dir: Path) -> EvaluationResult:
    """Evaluate RFT RAG system."""
    logger.info("Starting RFT evaluation...")

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

            results.append({
                'query_id': query_data['id'],
                'query': query,
                'response': response,
                'category': query_data.get('category', 'unknown'),
                'latency_ms': metadata['latency_ms'],
                'relevance_score': relevance,
                'usage': usage
            })

        except Exception as e:
            logger.error(f"Error processing query {query_data['id']}: {e}")

    response_metrics = calculate_response_metrics(responses)
    latency_metrics = calculate_latency_metrics(latencies)
    cost_metrics = calculate_cost_metrics(total_input_tokens, total_output_tokens)
    citation_metrics = calculate_citation_rate(responses)
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

    eval_result = EvaluationResult(
        model_name="rft-llama-3.1-8b",
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

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'rft_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(output_dir / 'rft_evaluation.json', 'w') as f:
        json.dump(eval_result.to_dict(), f, indent=2)

    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate RFT RAG system")
    parser.add_argument('--test-queries', type=Path, default=Path('01_baseline/test_queries.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/test_results'))
    parser.add_argument('--model-id-file', type=Path, default=Path('data/models/rft_model_id.txt'))
    parser.add_argument('--model-id', type=str, help='RFT model ID')
    parser.add_argument('--top-k', type=int, default=3)

    args = parser.parse_args()

    rag_system = RFTRAG(
        rft_model_id=args.model_id,
        model_id_file=args.model_id_file if not args.model_id else None,
        top_k=args.top_k
    )

    test_queries = load_test_queries(args.test_queries)
    eval_result = evaluate_rft(rag_system, test_queries, args.output_dir)

    print("\n" + format_evaluation_summary(eval_result))
    print(f"\nCitation Rate: {eval_result.metadata['citation_metrics']['citation_rate']:.1%}")
    print(f"P95 Latency: {eval_result.metadata['latency_metrics']['p95_ms']:.0f} ms")


if __name__ == "__main__":
    main()

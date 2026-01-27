"""
Evaluation metrics and helpers.
Provides utilities for measuring model performance.
"""
import re
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    model_name: str
    total_queries: int
    avg_response_length: float
    avg_latency_ms: float
    total_cost_usd: float
    relevance_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'total_queries': self.total_queries,
            'avg_response_length': self.avg_response_length,
            'avg_latency_ms': self.avg_latency_ms,
            'total_cost_usd': self.total_cost_usd,
            'relevance_score': self.relevance_score,
            'accuracy_score': self.accuracy_score,
            'metadata': self.metadata or {}
        }


def calculate_response_metrics(responses: List[str]) -> Dict:
    """
    Calculate metrics for a list of responses.

    Args:
        responses: List of response strings

    Returns:
        Dict with response metrics
    """
    if not responses:
        return {
            'count': 0,
            'avg_length': 0,
            'min_length': 0,
            'max_length': 0,
            'avg_word_count': 0
        }

    lengths = [len(r) for r in responses]
    word_counts = [len(r.split()) for r in responses]

    return {
        'count': len(responses),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_word_count': sum(word_counts) / len(word_counts)
    }


def calculate_latency_metrics(latencies_ms: List[float]) -> Dict:
    """
    Calculate latency metrics.

    Args:
        latencies_ms: List of latencies in milliseconds

    Returns:
        Dict with latency metrics
    """
    if not latencies_ms:
        return {
            'count': 0,
            'avg_ms': 0,
            'min_ms': 0,
            'max_ms': 0,
            'p50_ms': 0,
            'p95_ms': 0,
            'p99_ms': 0
        }

    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)

    def percentile(p):
        idx = int(n * p / 100)
        return sorted_latencies[min(idx, n - 1)]

    return {
        'count': n,
        'avg_ms': sum(latencies_ms) / n,
        'min_ms': min(latencies_ms),
        'max_ms': max(latencies_ms),
        'p50_ms': percentile(50),
        'p95_ms': percentile(95),
        'p99_ms': percentile(99)
    }


def calculate_cost_metrics(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1m: float = 0.20,
    output_cost_per_1m: float = 0.20
) -> Dict:
    """
    Calculate cost metrics.

    Args:
        input_tokens: Total input tokens
        output_tokens: Total output tokens
        input_cost_per_1m: Cost per 1M input tokens
        output_cost_per_1m: Cost per 1M output tokens

    Returns:
        Dict with cost metrics
    """
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost

    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
        'input_cost_usd': round(input_cost, 4),
        'output_cost_usd': round(output_cost, 4),
        'total_cost_usd': round(total_cost, 4),
        'cost_per_1k_tokens': round((total_cost / (input_tokens + output_tokens)) * 1000, 4) if (input_tokens + output_tokens) > 0 else 0
    }


def simple_relevance_score(
    response: str,
    expected_keywords: List[str],
    case_sensitive: bool = False
) -> float:
    """
    Calculate a simple relevance score based on keyword presence.

    Args:
        response: Response text
        expected_keywords: List of keywords that should appear
        case_sensitive: Whether to match case-sensitively

    Returns:
        Relevance score (0-1)
    """
    if not expected_keywords:
        return 1.0

    if not case_sensitive:
        response = response.lower()
        expected_keywords = [kw.lower() for kw in expected_keywords]

    matches = sum(1 for kw in expected_keywords if kw in response)
    return matches / len(expected_keywords)


def calculate_token_overlap(text1: str, text2: str) -> float:
    """
    Calculate token overlap between two texts (simple Jaccard similarity).

    Args:
        text1: First text
        text2: Second text

    Returns:
        Overlap score (0-1)
    """
    # Tokenize (simple word split)
    tokens1 = set(re.findall(r'\w+', text1.lower()))
    tokens2 = set(re.findall(r'\w+', text2.lower()))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


def extract_paper_citations(response: str) -> List[str]:
    """
    Extract paper citations/mentions from response.

    Args:
        response: Response text

    Returns:
        List of cited paper titles or IDs
    """
    # Look for common citation patterns
    patterns = [
        r'(?:in|from|by|paper titled?|titled?) ["\']([^"\']+)["\']',
        r'(?:arxiv|arXiv)[:\s]+(\d+\.\d+)',
        r'\[([^\]]+)\]\s*\(',  # Markdown links
    ]

    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        citations.extend(matches)

    return citations


def calculate_citation_rate(responses: List[str]) -> Dict:
    """
    Calculate citation statistics across responses.

    Args:
        responses: List of response texts

    Returns:
        Dict with citation metrics
    """
    total_citations = 0
    responses_with_citations = 0

    for response in responses:
        citations = extract_paper_citations(response)
        if citations:
            responses_with_citations += 1
            total_citations += len(citations)

    return {
        'total_responses': len(responses),
        'responses_with_citations': responses_with_citations,
        'citation_rate': responses_with_citations / len(responses) if responses else 0,
        'total_citations': total_citations,
        'avg_citations_per_response': total_citations / len(responses) if responses else 0
    }


def compare_models(results: List[EvaluationResult]) -> Dict:
    """
    Compare multiple model evaluation results.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Dict with comparison metrics
    """
    if not results:
        return {}

    comparison = {
        'models': [r.model_name for r in results],
        'total_queries': results[0].total_queries,
        'metrics': {}
    }

    # Compare each metric
    metrics_to_compare = [
        'avg_response_length',
        'avg_latency_ms',
        'total_cost_usd',
        'relevance_score',
        'accuracy_score'
    ]

    for metric in metrics_to_compare:
        values = [getattr(r, metric) for r in results if getattr(r, metric) is not None]

        if values:
            comparison['metrics'][metric] = {
                'values': dict(zip([r.model_name for r in results], values)),
                'best_model': results[values.index(max(values) if 'score' in metric else min(values))].model_name,
                'best_value': max(values) if 'score' in metric else min(values)
            }

    # Calculate improvement percentages
    if len(results) >= 2:
        baseline = results[0]
        improvements = {}

        for i, result in enumerate(results[1:], 1):
            model_improvements = {}

            if baseline.relevance_score and result.relevance_score:
                improvement = ((result.relevance_score - baseline.relevance_score) / baseline.relevance_score) * 100
                model_improvements['relevance_improvement_pct'] = round(improvement, 2)

            if baseline.accuracy_score and result.accuracy_score:
                improvement = ((result.accuracy_score - baseline.accuracy_score) / baseline.accuracy_score) * 100
                model_improvements['accuracy_improvement_pct'] = round(improvement, 2)

            if baseline.avg_latency_ms and result.avg_latency_ms:
                improvement = ((baseline.avg_latency_ms - result.avg_latency_ms) / baseline.avg_latency_ms) * 100
                model_improvements['latency_improvement_pct'] = round(improvement, 2)

            improvements[result.model_name] = model_improvements

        comparison['improvements_vs_baseline'] = improvements

    return comparison


def format_evaluation_summary(result: EvaluationResult) -> str:
    """
    Format evaluation result as a readable summary.

    Args:
        result: EvaluationResult object

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 70,
        f"Model: {result.model_name}",
        "=" * 70,
        f"Total Queries: {result.total_queries}",
        f"Avg Response Length: {result.avg_response_length:.0f} characters",
        f"Avg Latency: {result.avg_latency_ms:.0f} ms",
        f"Total Cost: ${result.total_cost_usd:.4f}",
    ]

    if result.relevance_score is not None:
        lines.append(f"Relevance Score: {result.relevance_score:.2%}")

    if result.accuracy_score is not None:
        lines.append(f"Accuracy Score: {result.accuracy_score:.2%}")

    if result.metadata:
        lines.append("\nAdditional Metrics:")
        for key, value in result.metadata.items():
            lines.append(f"  {key}: {value}")

    lines.append("=" * 70)

    return "\n".join(lines)


def format_comparison_summary(comparison: Dict) -> str:
    """
    Format model comparison as a readable summary.

    Args:
        comparison: Comparison dict from compare_models()

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 70,
        "MODEL COMPARISON",
        "=" * 70,
        f"Models: {', '.join(comparison['models'])}",
        f"Total Queries: {comparison['total_queries']}",
        "",
        "Metrics Comparison:",
    ]

    for metric, data in comparison['metrics'].items():
        lines.append(f"\n{metric.replace('_', ' ').title()}:")
        for model, value in data['values'].items():
            marker = " ⭐" if model == data['best_model'] else ""
            lines.append(f"  {model}: {value:.2f}{marker}")

    if 'improvements_vs_baseline' in comparison:
        lines.append("\nImprovements vs Baseline:")
        for model, improvements in comparison['improvements_vs_baseline'].items():
            lines.append(f"\n{model}:")
            for metric, value in improvements.items():
                sign = "+" if value > 0 else ""
                lines.append(f"  {metric.replace('_', ' ').title()}: {sign}{value}%")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)

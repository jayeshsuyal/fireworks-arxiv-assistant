"""
Shared utilities for Fireworks arXiv Assistant.
Provides common clients and helpers used across modules.
"""

from .fireworks_client import FireworksClient, get_client as get_fireworks_client
from .pinecone_client import PineconeClient, get_client as get_pinecone_client
from .metrics import (
    EvaluationResult,
    calculate_response_metrics,
    calculate_latency_metrics,
    calculate_cost_metrics,
    simple_relevance_score,
    calculate_token_overlap,
    extract_paper_citations,
    calculate_citation_rate,
    compare_models,
    format_evaluation_summary,
    format_comparison_summary
)

__version__ = "0.1.0"

__all__ = [
    'FireworksClient',
    'get_fireworks_client',
    'PineconeClient',
    'get_pinecone_client',
    'EvaluationResult',
    'calculate_response_metrics',
    'calculate_latency_metrics',
    'calculate_cost_metrics',
    'simple_relevance_score',
    'calculate_token_overlap',
    'extract_paper_citations',
    'calculate_citation_rate',
    'compare_models',
    'format_evaluation_summary',
    'format_comparison_summary',
]

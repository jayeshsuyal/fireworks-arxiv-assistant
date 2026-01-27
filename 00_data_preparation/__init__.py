"""
arXiv data preparation pipeline.
Fetches, embeds, and generates training data from arXiv papers.
"""

from .fetch_papers import ArxivPaper, ArxivFetcher
from .embed_papers import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    FireworksEmbeddings,
    PaperEmbedder
)
from .generate_training_data import TrainingDataGenerator
from .generate_preference_data import PreferenceDataGenerator

__version__ = "0.1.0"

__all__ = [
    'ArxivPaper',
    'ArxivFetcher',
    'EmbeddingProvider',
    'OpenAIEmbeddings',
    'FireworksEmbeddings',
    'PaperEmbedder',
    'TrainingDataGenerator',
    'PreferenceDataGenerator',
]

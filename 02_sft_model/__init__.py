"""
Supervised Fine-Tuning (SFT) model module.
Fine-tunes base model on arXiv Q&A data.
"""

from .sft_rag import SFTRAG

__version__ = "0.1.0"

__all__ = ['SFTRAG']

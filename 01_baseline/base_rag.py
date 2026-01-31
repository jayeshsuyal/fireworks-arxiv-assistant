"""
Baseline RAG system using vanilla Llama-3.1-8B.
Establishes performance baseline before fine-tuning.
"""
import time
from typing import List, Dict, Optional, Tuple
import logging
from openai import OpenAI

from utils.fireworks_client import FireworksClient
from utils.pinecone_client import PineconeClient

logger = logging.getLogger(__name__)


class BaselineRAG:
    """
    Baseline RAG system with vanilla (non-fine-tuned) model.

    Architecture:
    1. User query → Embedding
    2. Retrieve relevant papers from Pinecone
    3. Construct prompt with context
    4. Generate answer with base model
    """

    def __init__(
        self,
        fireworks_client: Optional[FireworksClient] = None,
        pinecone_client: Optional[PineconeClient] = None,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        base_model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize baseline RAG system.

        Args:
            fireworks_client: Fireworks client (created if None)
            pinecone_client: Pinecone client (created if None)
            embedding_model: Model for query embeddings
            base_model: Base model for generation
            top_k: Number of papers to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.fireworks_client = fireworks_client or FireworksClient(default_model=base_model)
        self.pinecone_client = pinecone_client or PineconeClient()
        self.embedding_model = embedding_model
        self.base_model = base_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # For embeddings (using Fireworks)
        self.embedding_client = OpenAI(
            api_key=self.fireworks_client.api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )

        logger.info(f"Initialized BaselineRAG with model: {base_model}")

    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query."""
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding

    def _retrieve_context(self, query_embedding: List[float]) -> str:
        """Retrieve relevant papers and format as context."""
        return self.pinecone_client.get_context_for_query(
            query_vector=query_embedding,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )

    def _construct_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        Construct messages for chat completion.

        Args:
            query: User query
            context: Retrieved context from papers

        Returns:
            List of message dicts
        """
        system_message = """You are a helpful AI research assistant specializing in arXiv papers.

Your role is to:
1. Answer questions about AI/ML research based on the provided papers
2. Cite specific papers when referencing information
3. Provide accurate, concise, and well-structured responses
4. Acknowledge when information is not available in the context

Always ground your responses in the provided papers."""

        user_message = f"""{context}

User Question: {query}

Please provide a comprehensive answer based on the papers above. Cite specific papers when referencing their content."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def query(
        self,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        return_context: bool = False
    ) -> Tuple[str, Dict]:
        """
        Query the RAG system.

        Args:
            question: User question
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            return_context: Whether to return retrieved context

        Returns:
            Tuple of (answer, metadata)
        """
        start_time = time.time()

        # 1. Embed query
        query_embedding = self._embed_query(question)

        # 2. Retrieve context
        context = self._retrieve_context(query_embedding)

        # 3. Construct prompt
        messages = self._construct_prompt(question, context)

        # 4. Generate response
        response, completion_metadata = self.fireworks_client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Prepare metadata
        metadata = {
            'model': self.base_model,
            'latency_ms': latency_ms,
            'usage': completion_metadata['usage'],
            'top_k': self.top_k
        }

        if return_context:
            metadata['context'] = context

        return response, metadata

    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[Tuple[str, Dict]]:
        """
        Process multiple queries.

        Args:
            questions: List of questions
            **kwargs: Arguments passed to query()

        Returns:
            List of (answer, metadata) tuples
        """
        results = []

        for question in questions:
            try:
                answer, metadata = self.query(question, **kwargs)
                results.append((answer, metadata))
            except Exception as e:
                logger.error(f"Error processing query '{question[:50]}...': {e}")
                results.append((f"Error: {str(e)}", {'error': str(e)}))

        return results

    def get_system_info(self) -> Dict:
        """Get system configuration info."""
        return {
            'type': 'baseline',
            'base_model': self.base_model,
            'embedding_model': self.embedding_model,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'pinecone_index': self.pinecone_client.index_name,
            'pinecone_namespace': self.pinecone_client.namespace
        }

"""
RAG system using reinforcement fine-tuned (RFT) model.
Final model trained with preference optimization on top of SFT.
"""
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from openai import OpenAI

from utils.fireworks_client import FireworksClient
from utils.pinecone_client import PineconeClient

logger = logging.getLogger(__name__)


class RFTRAG:
    """
    RAG system with reinforcement fine-tuned model.

    Built on top of SFT model with additional preference training.
    Expected improvements over SFT:
    - Better response quality (avoiding verbose/incorrect responses)
    - More focused and relevant answers
    - Improved citation accuracy
    """

    def __init__(
        self,
        rft_model_id: Optional[str] = None,
        model_id_file: Optional[Path] = None,
        fireworks_client: Optional[FireworksClient] = None,
        pinecone_client: Optional[PineconeClient] = None,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ):
        """Initialize RFT RAG system."""
        # Get RFT model ID
        if rft_model_id:
            self.rft_model_id = rft_model_id
        elif model_id_file:
            if not model_id_file.exists():
                raise FileNotFoundError(f"Model ID file not found: {model_id_file}")
            with open(model_id_file, 'r') as f:
                self.rft_model_id = f.read().strip()
        else:
            raise ValueError("Either rft_model_id or model_id_file must be provided")

        self.fireworks_client = fireworks_client or FireworksClient(default_model=self.rft_model_id)
        self.pinecone_client = pinecone_client or PineconeClient()
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self.embedding_client = OpenAI(
            api_key=self.fireworks_client.api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )

        logger.info(f"Initialized RFTRAG with model: {self.rft_model_id}")

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
        """Construct messages for chat completion."""
        system_message = """You are a helpful AI research assistant that provides accurate, concise answers about arXiv papers."""

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
        """Query the RAG system."""
        start_time = time.time()

        query_embedding = self._embed_query(question)
        context = self._retrieve_context(query_embedding)
        messages = self._construct_prompt(question, context)

        response, completion_metadata = self.fireworks_client.chat_completion(
            messages=messages,
            model=self.rft_model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )

        latency_ms = (time.time() - start_time) * 1000

        metadata = {
            'model': self.rft_model_id,
            'model_type': 'rft',
            'latency_ms': latency_ms,
            'usage': completion_metadata['usage'],
            'top_k': self.top_k
        }

        if return_context:
            metadata['context'] = context

        return response, metadata

    def batch_query(self, questions: List[str], **kwargs) -> List[Tuple[str, Dict]]:
        """Process multiple queries."""
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
            'type': 'rft',
            'rft_model_id': self.rft_model_id,
            'embedding_model': self.embedding_model,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'pinecone_index': self.pinecone_client.index_name,
            'pinecone_namespace': self.pinecone_client.namespace
        }

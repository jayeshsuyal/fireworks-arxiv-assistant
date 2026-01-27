"""
Pinecone operations wrapper.
Provides utilities for vector search and retrieval.
"""
import os
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class PineconeClient:
    """
    Wrapper for Pinecone vector database operations.

    Provides:
    - Vector search/retrieval for RAG
    - Context augmentation
    - Query utilities
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: str = "arxiv-papers",
        namespace: str = "fireworks-demo"
    ):
        """
        Initialize Pinecone client.

        Args:
            api_key: Pinecone API key (or use PINECONE_API_KEY env var)
            index_name: Pinecone index name
            namespace: Pinecone namespace
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY environment variable.")

        self.index_name = index_name
        self.namespace = namespace

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(index_name)

        logger.info(f"Connected to Pinecone index: {index_name}, namespace: {namespace}")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            include_metadata: Whether to include metadata
            filter: Optional metadata filter

        Returns:
            List of match dicts with id, score, and metadata
        """
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=include_metadata,
            filter=filter
        )

        matches = []
        for match in results.matches:
            match_dict = {
                'id': match.id,
                'score': match.score
            }

            if include_metadata and hasattr(match, 'metadata'):
                match_dict['metadata'] = match.metadata

            matches.append(match_dict)

        return matches

    def get_context_for_query(
        self,
        query_vector: List[float],
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ) -> str:
        """
        Get formatted context for RAG from similar papers.

        Args:
            query_vector: Query embedding vector
            top_k: Number of papers to retrieve
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            Formatted context string
        """
        matches = self.search(query_vector, top_k=top_k)

        # Filter by similarity threshold
        relevant_matches = [m for m in matches if m['score'] >= similarity_threshold]

        if not relevant_matches:
            return "No relevant papers found in the database."

        context_parts = []

        for i, match in enumerate(relevant_matches, 1):
            metadata = match.get('metadata', {})

            paper_context = f"""Paper {i}:
Title: {metadata.get('title', 'Unknown')}
Authors: {metadata.get('authors', 'Unknown')}
Summary: {metadata.get('summary', 'No summary available')}
Categories: {metadata.get('categories', 'Unknown')}
Similarity Score: {match['score']:.3f}
"""
            context_parts.append(paper_context)

        context = "\n".join(context_parts)

        return f"""Relevant Papers from arXiv:

{context}

Please answer the user's question based on these papers."""

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """
        Fetch a specific paper by ID.

        Args:
            paper_id: Paper ID to fetch

        Returns:
            Paper dict or None if not found
        """
        try:
            result = self.index.fetch(ids=[paper_id], namespace=self.namespace)

            if paper_id in result.vectors:
                vector = result.vectors[paper_id]
                return {
                    'id': vector.id,
                    'metadata': vector.metadata if hasattr(vector, 'metadata') else {}
                }

            return None

        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = self.index.describe_index_stats()

        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension,
            'namespaces': {
                ns: info.vector_count
                for ns, info in stats.namespaces.items()
            }
        }

    def batch_search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 5
    ) -> List[List[Dict]]:
        """
        Perform multiple searches in batch.

        Args:
            query_vectors: List of query embedding vectors
            top_k: Number of results per query

        Returns:
            List of match lists
        """
        all_results = []

        for query_vector in query_vectors:
            results = self.search(query_vector, top_k=top_k)
            all_results.append(results)

        return all_results


# Convenience function
def get_client(
    index_name: str = "arxiv-papers",
    namespace: str = "fireworks-demo"
) -> PineconeClient:
    """
    Get a configured Pinecone client.

    Args:
        index_name: Pinecone index name
        namespace: Pinecone namespace

    Returns:
        PineconeClient instance
    """
    return PineconeClient(index_name=index_name, namespace=namespace)

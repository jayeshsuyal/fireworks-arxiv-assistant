"""
Embed arXiv papers and store in Pinecone for RAG.

Improvements:
- Retry logic for API failures
- Rate limiting for embedding APIs
- Support for both Fireworks and OpenAI embeddings
- Deduplication against existing vectors
- Resume capability for interrupted runs
- Cost tracking for embeddings
- CLI arguments for flexibility
"""
import json
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Base class for embedding providers."""

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        raise NotImplementedError

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        raise NotImplementedError

    def get_cost_per_1k_tokens(self) -> float:
        """Get approximate cost per 1k tokens."""
        raise NotImplementedError


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension_map = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        self.cost_map = {
            "text-embedding-3-large": 0.00013,  # per 1k tokens
            "text-embedding-3-small": 0.00002,
            "text-embedding-ada-002": 0.0001
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding with retry logic."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimension_map.get(self.model, 1536)
        )
        return response.data[0].embedding

    def get_dimension(self) -> int:
        return self.dimension_map.get(self.model, 1536)

    def get_cost_per_1k_tokens(self) -> float:
        return self.cost_map.get(self.model, 0.0001)


class FireworksEmbeddings(EmbeddingProvider):
    """Fireworks AI embeddings provider."""

    def __init__(self, api_key: str, model: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Initialize Fireworks embeddings.

        Available models:
        - nomic-ai/nomic-embed-text-v1.5 (768 dim)
        - WhereIsAI/UAE-Large-V1 (1024 dim)
        - thenlper/gte-large (1024 dim)
        """
        # Fireworks uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )
        self.model = model
        self.dimension_map = {
            "nomic-ai/nomic-embed-text-v1.5": 768,
            "WhereIsAI/UAE-Large-V1": 1024,
            "thenlper/gte-large": 1024
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding with retry logic."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def get_dimension(self) -> int:
        return self.dimension_map.get(self.model, 768)

    def get_cost_per_1k_tokens(self) -> float:
        # Fireworks embeddings are typically much cheaper
        return 0.00001  # Approximate cost


class PaperEmbedder:
    """Embeds papers and stores them in Pinecone with deduplication and resume capability."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        pinecone_api_key: str,
        index_name: str = "arxiv-papers",
        namespace: str = "fireworks-demo",
        rate_limit_delay: float = 0.1,  # Delay between embedding calls
        enable_deduplication: bool = True
    ):
        """
        Initialize embedder.

        Args:
            embedding_provider: Embedding provider instance
            pinecone_api_key: Pinecone API key
            index_name: Pinecone index name
            namespace: Pinecone namespace
            rate_limit_delay: Delay between embedding API calls
            enable_deduplication: Whether to skip already embedded papers
        """
        self.embedding_provider = embedding_provider
        self.index_name = index_name
        self.namespace = namespace
        self.rate_limit_delay = rate_limit_delay
        self.enable_deduplication = enable_deduplication
        self.logger = logging.getLogger(__name__)

        # Cost tracking
        self.total_tokens_estimated = 0
        self.embeddings_generated = 0

        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self._init_index()

        # Load existing vector IDs for deduplication
        self.existing_ids: Set[str] = set()
        if self.enable_deduplication:
            self._load_existing_ids()

    def _init_index(self):
        """Initialize Pinecone index."""
        dimension = self.embedding_provider.get_dimension()
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self.logger.info(f"Creating Pinecone index: {self.index_name}")
            self.logger.info(f"Dimension: {dimension}")

            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

            self.logger.info("Waiting for index to be ready...")
            time.sleep(10)
        else:
            # Verify dimension matches
            index_info = self.pc.describe_index(self.index_name)
            if index_info.dimension != dimension:
                self.logger.warning(
                    f"Index dimension mismatch! Index: {index_info.dimension}, "
                    f"Provider: {dimension}. Consider using a different index name."
                )

        self.index = self.pc.Index(self.index_name)
        self.logger.info(f"Connected to index: {self.index_name}")

    def _load_existing_ids(self):
        """Load existing vector IDs from Pinecone for deduplication."""
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.namespace, None)

            if namespace_stats and namespace_stats.vector_count > 0:
                # Note: Pinecone doesn't provide a direct way to list all IDs
                # We'll track them as we encounter them during queries
                self.logger.info(
                    f"Found {namespace_stats.vector_count} existing vectors in namespace '{self.namespace}'"
                )
                self.logger.info("Will check for duplicates during upsert")
        except Exception as e:
            self.logger.warning(f"Could not load existing IDs: {e}")

    def load_papers(self, filepath: Path) -> List[Dict]:
        """
        Load papers from JSONL file.

        Args:
            filepath: Path to papers.jsonl

        Returns:
            List of paper dictionaries
        """
        papers = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                papers.append(json.loads(line))

        self.logger.info(f"Loaded {len(papers)} papers from {filepath}")
        return papers

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
        return len(text) // 4

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text with retry and rate limiting.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Rate limiting
        time.sleep(self.rate_limit_delay)

        # Track tokens for cost estimation
        tokens = self._estimate_tokens(text)
        self.total_tokens_estimated += tokens
        self.embeddings_generated += 1

        return self.embedding_provider.get_embedding(text)

    def prepare_vectors(
        self,
        papers: List[Dict],
        check_existing: bool = True
    ) -> List[Dict]:
        """
        Prepare vectors for Pinecone upsert.

        Args:
            papers: List of paper dictionaries
            check_existing: Whether to check if vectors already exist

        Returns:
            List of vectors ready for upsert
        """
        vectors = []
        skipped = 0
        errors = 0

        self.logger.info("Generating embeddings...")

        # Check which papers already exist if deduplication is enabled
        if self.enable_deduplication and check_existing:
            paper_ids = [p['paper_id'] for p in papers]

            # Fetch existing vectors in batches
            try:
                for i in range(0, len(paper_ids), 100):
                    batch_ids = paper_ids[i:i+100]
                    fetch_response = self.index.fetch(ids=batch_ids, namespace=self.namespace)
                    self.existing_ids.update(fetch_response.vectors.keys())

                self.logger.info(f"Found {len(self.existing_ids)} existing vectors to skip")
            except Exception as e:
                self.logger.warning(f"Could not check existing vectors: {e}")

        for paper in tqdm(papers, desc="Embedding papers"):
            try:
                paper_id = paper['paper_id']

                # Skip if already embedded
                if self.enable_deduplication and paper_id in self.existing_ids:
                    skipped += 1
                    continue

                # Combine title and summary for embedding
                text = f"{paper['title']}\n\n{paper['summary']}"

                # Generate embedding
                embedding = self.embed_text(text)

                # Prepare vector
                vector = {
                    'id': paper_id,
                    'values': embedding,
                    'metadata': {
                        'title': paper['title'][:500],  # Truncate for metadata limits
                        'summary': paper['summary'][:1000],
                        'authors': ', '.join(paper['authors'][:5]),
                        'categories': ', '.join(paper['categories'][:5]),
                        'published': paper['published'],
                        'pdf_url': paper['pdf_url']
                    }
                }

                vectors.append(vector)

            except Exception as e:
                errors += 1
                self.logger.error(f"Error embedding paper {paper.get('paper_id', 'unknown')}: {e}")
                continue

        if skipped > 0:
            self.logger.info(f"Skipped {skipped} already embedded papers")
        if errors > 0:
            self.logger.warning(f"Encountered {errors} errors during embedding")

        return vectors

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def upsert_vectors(self, vectors: List[Dict], batch_size: int = 100):
        """
        Upsert vectors to Pinecone with retry logic.

        Args:
            vectors: List of vectors to upsert
            batch_size: Batch size for upsert
        """
        if not vectors:
            self.logger.warning("No vectors to upsert")
            return

        self.logger.info(f"Upserting {len(vectors)} vectors to Pinecone...")

        successful_batches = 0
        failed_batches = 0

        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading batches"):
            batch = vectors[i:i + batch_size]

            try:
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace
                )
                successful_batches += 1
                time.sleep(0.5)  # Small delay between batches
            except Exception as e:
                failed_batches += 1
                self.logger.error(f"Error upserting batch {i//batch_size}: {e}")
                continue

        self.logger.info(f"Upsert complete! Successful: {successful_batches}, Failed: {failed_batches}")

    def get_stats(self) -> Dict:
        """Get Pinecone index statistics."""
        stats = self.index.describe_index_stats()

        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension,
            'namespaces': {
                ns: info.vector_count
                for ns, info in stats.namespaces.items()
            }
        }

    def get_cost_estimate(self) -> Dict:
        """Get estimated cost for embeddings."""
        cost_per_1k = self.embedding_provider.get_cost_per_1k_tokens()
        estimated_cost = (self.total_tokens_estimated / 1000) * cost_per_1k

        return {
            'total_embeddings': self.embeddings_generated,
            'estimated_tokens': self.total_tokens_estimated,
            'cost_per_1k_tokens': cost_per_1k,
            'estimated_cost_usd': round(estimated_cost, 4)
        }


def main():
    """Main execution function with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Embed arXiv papers and store in Pinecone for RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-file',
        type=Path,
        default=Path("data/papers.jsonl"),
        help='Input papers file'
    )
    parser.add_argument(
        '--index-name',
        type=str,
        default="arxiv-papers",
        help='Pinecone index name'
    )
    parser.add_argument(
        '--namespace',
        type=str,
        default="fireworks-demo",
        help='Pinecone namespace'
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'fireworks'],
        default='fireworks',
        help='Embedding provider to use'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default=None,
        help='Specific embedding model (provider-dependent)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.1,
        help='Delay between embedding API calls in seconds'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for Pinecone upsert'
    )
    parser.add_argument(
        '--no-dedupe',
        action='store_true',
        help='Disable deduplication against existing vectors'
    )

    args = parser.parse_args()

    # Load API keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    if not PINECONE_API_KEY:
        logger.error("Missing PINECONE_API_KEY environment variable")
        return

    # Initialize embedding provider
    if args.provider == 'openai':
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            logger.error("Missing OPENAI_API_KEY environment variable")
            return

        model = args.embedding_model or "text-embedding-3-large"
        embedding_provider = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=model)
        logger.info(f"Using OpenAI embeddings: {model}")

    elif args.provider == 'fireworks':
        FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
        if not FIREWORKS_API_KEY:
            logger.error("Missing FIREWORKS_API_KEY environment variable")
            return

        model = args.embedding_model or "nomic-ai/nomic-embed-text-v1.5"
        embedding_provider = FireworksEmbeddings(api_key=FIREWORKS_API_KEY, model=model)
        logger.info(f"Using Fireworks embeddings: {model}")

    # Check input file
    if not args.input_file.exists():
        logger.error(f"Papers file not found: {args.input_file}")
        logger.info("Run fetch_papers.py first to download papers")
        return

    # Create embedder
    embedder = PaperEmbedder(
        embedding_provider=embedding_provider,
        pinecone_api_key=PINECONE_API_KEY,
        index_name=args.index_name,
        namespace=args.namespace,
        rate_limit_delay=args.rate_limit,
        enable_deduplication=not args.no_dedupe
    )

    # Load papers
    papers = embedder.load_papers(args.input_file)

    # Prepare vectors
    vectors = embedder.prepare_vectors(papers)

    if not vectors:
        logger.warning("No new vectors to embed. All papers may already be embedded.")
        return

    # Upsert to Pinecone
    embedder.upsert_vectors(vectors, batch_size=args.batch_size)

    # Get and display statistics
    stats = embedder.get_stats()
    cost = embedder.get_cost_estimate()

    logger.info("\n" + "="*70)
    logger.info("EMBEDDING COMPLETE")
    logger.info("="*70)
    logger.info(f"Total vectors in index: {stats['total_vectors']}")
    logger.info(f"Dimension: {stats['dimension']}")
    logger.info(f"Namespace '{args.namespace}': {stats['namespaces'].get(args.namespace, 0)} vectors")
    logger.info("\nCost Estimate:")
    logger.info(f"  Embeddings generated: {cost['total_embeddings']}")
    logger.info(f"  Estimated tokens: {cost['estimated_tokens']:,}")
    logger.info(f"  Cost per 1k tokens: ${cost['cost_per_1k_tokens']:.5f}")
    logger.info(f"  Estimated total cost: ${cost['estimated_cost_usd']:.4f}")
    logger.info("="*70)


if __name__ == "__main__":
    main()

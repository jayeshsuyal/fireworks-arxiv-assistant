"""
Fetch arXiv papers using the arXiv API.
Retrieves papers from AI/ML categories for training data.

Improvements:
- Retry logic for API failures
- Rate limiting to respect arXiv servers
- Deduplication to avoid duplicate papers
- Resume capability for interrupted fetches
- CLI arguments for flexibility
"""
import arxiv
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Represents an arXiv paper."""
    paper_id: str
    title: str
    authors: List[str]
    summary: str
    categories: List[str]
    published: str
    pdf_url: str

    @classmethod
    def from_arxiv_result(cls, result: arxiv.Result) -> 'ArxivPaper':
        """Create ArxivPaper from arxiv.Result object."""
        return cls(
            paper_id=result.entry_id.split('/')[-1],  # Extract ID from URL
            title=result.title.strip(),
            authors=[author.name for author in result.authors],
            summary=result.summary.strip().replace('\n', ' '),
            categories=result.categories,
            published=result.published.isoformat(),
            pdf_url=result.pdf_url
        )


class ArxivFetcher:
    """Fetches papers from arXiv API with retry logic and rate limiting."""

    def __init__(
        self,
        output_dir: Path = Path("data"),
        rate_limit_delay: float = 3.0,  # Seconds between requests
        enable_deduplication: bool = True
    ):
        """
        Initialize fetcher.

        Args:
            output_dir: Directory to save fetched papers
            rate_limit_delay: Delay between API requests in seconds
            enable_deduplication: Whether to deduplicate against existing papers
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.enable_deduplication = enable_deduplication
        self.logger = logging.getLogger(__name__)
        self.seen_paper_ids: Set[str] = set()

        # Load existing paper IDs for deduplication
        if self.enable_deduplication:
            self._load_existing_paper_ids()

    def _load_existing_paper_ids(self):
        """Load existing paper IDs from previously saved files."""
        papers_file = self.output_dir / "papers.jsonl"
        if papers_file.exists():
            try:
                with open(papers_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        paper = json.loads(line)
                        self.seen_paper_ids.add(paper['paper_id'])
                self.logger.info(f"Loaded {len(self.seen_paper_ids)} existing paper IDs for deduplication")
            except Exception as e:
                self.logger.warning(f"Could not load existing papers for deduplication: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((arxiv.ArxivError, ConnectionError)),
        reraise=True
    )
    def _fetch_batch(self, search: arxiv.Search) -> List[arxiv.Result]:
        """
        Fetch a batch of results with retry logic.

        Args:
            search: arxiv.Search object

        Returns:
            List of arxiv.Result objects
        """
        return list(search.results())

    def fetch_papers(
        self,
        query: str = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
        max_results: int = 100,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
        days_back: int = None  # Optional: only fetch papers from last N days
    ) -> List[ArxivPaper]:
        """
        Fetch papers from arXiv API.

        Args:
            query: arXiv search query
            max_results: Maximum number of papers to fetch
            sort_by: Sort criterion
            days_back: Only fetch papers from last N days (None = no filter)

        Returns:
            List of ArxivPaper objects
        """
        self.logger.info(f"Fetching up to {max_results} papers from arXiv...")
        self.logger.info(f"Query: {query}")

        # Add date filter if specified
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            date_str = cutoff_date.strftime("%Y%m%d")
            query = f"({query}) AND submittedDate:[{date_str}000000 TO 99991231235959]"
            self.logger.info(f"Filtering papers from last {days_back} days (since {cutoff_date.date()})")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )

        papers = []
        duplicates_skipped = 0
        errors = 0

        try:
            self.logger.info("Starting paper fetch (with retry logic and rate limiting)...")
            results = self._fetch_batch(search)

            for result in tqdm(results, desc="Processing papers"):
                try:
                    paper = ArxivPaper.from_arxiv_result(result)

                    # Check for duplicates
                    if self.enable_deduplication and paper.paper_id in self.seen_paper_ids:
                        duplicates_skipped += 1
                        continue

                    papers.append(paper)
                    self.seen_paper_ids.add(paper.paper_id)

                    # Rate limiting - be respectful to arXiv servers
                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    errors += 1
                    self.logger.error(f"Error processing paper {getattr(result, 'entry_id', 'unknown')}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error during arXiv search: {e}")
            if papers:
                self.logger.warning(f"Partial results available: {len(papers)} papers fetched before error")
            else:
                raise

        self.logger.info(f"Successfully fetched {len(papers)} new papers")
        if duplicates_skipped > 0:
            self.logger.info(f"Skipped {duplicates_skipped} duplicate papers")
        if errors > 0:
            self.logger.warning(f"Encountered {errors} errors during processing")

        return papers

    def save_papers(
        self,
        papers: List[ArxivPaper],
        filename: str = "papers.jsonl",
        append: bool = True
    ) -> Path:
        """
        Save papers to JSONL file.

        Args:
            papers: List of papers to save
            filename: Output filename
            append: If True, append to existing file; if False, overwrite

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        mode = 'a' if append and output_path.exists() else 'w'

        self.logger.info(f"Saving {len(papers)} papers to {output_path} (mode: {mode})...")

        with open(output_path, mode, encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(asdict(paper), ensure_ascii=False) + '\n')

        self.logger.info(f"Papers saved to {output_path}")
        return output_path

    def load_papers(self, filename: str = "papers.jsonl") -> List[ArxivPaper]:
        """
        Load papers from JSONL file.

        Args:
            filename: Input filename

        Returns:
            List of ArxivPaper objects
        """
        input_path = self.output_dir / filename

        if not input_path.exists():
            self.logger.warning(f"File {input_path} does not exist")
            return []

        papers = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                paper_dict = json.loads(line)
                papers.append(ArxivPaper(**paper_dict))

        self.logger.info(f"Loaded {len(papers)} papers from {input_path}")
        return papers

    def get_statistics(self, papers: List[ArxivPaper]) -> Dict:
        """
        Get statistics about fetched papers.

        Args:
            papers: List of papers

        Returns:
            Dictionary with statistics
        """
        if not papers:
            return {'total_papers': 0}

        # Count papers by category
        category_counts = {}
        for paper in papers:
            for category in paper.categories:
                category_counts[category] = category_counts.get(category, 0) + 1

        # Get date range
        dates = [datetime.fromisoformat(p.published) for p in papers]

        # Author statistics
        total_authors = sum(len(p.authors) for p in papers)

        stats = {
            'total_papers': len(papers),
            'categories': category_counts,
            'unique_categories': len(category_counts),
            'date_range': {
                'earliest': min(dates).isoformat() if dates else None,
                'latest': max(dates).isoformat() if dates else None
            },
            'summary_stats': {
                'avg_length': sum(len(p.summary) for p in papers) / len(papers),
                'min_length': min(len(p.summary) for p in papers),
                'max_length': max(len(p.summary) for p in papers)
            },
            'author_stats': {
                'total_authors': total_authors,
                'avg_authors_per_paper': total_authors / len(papers)
            }
        }

        return stats


def main():
    """Main execution function with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch papers from arXiv API for AI/ML research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--max-results',
        type=int,
        default=100,
        help='Maximum number of papers to fetch'
    )
    parser.add_argument(
        '--query',
        type=str,
        default="cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
        help='arXiv search query'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=None,
        help='Only fetch papers from last N days (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("data"),
        help='Output directory for saved papers'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=3.0,
        help='Delay between API requests in seconds'
    )
    parser.add_argument(
        '--no-dedupe',
        action='store_true',
        help='Disable deduplication against existing papers'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing papers file instead of appending'
    )

    args = parser.parse_args()

    # Create fetcher
    fetcher = ArxivFetcher(
        output_dir=args.output_dir,
        rate_limit_delay=args.rate_limit,
        enable_deduplication=not args.no_dedupe
    )

    # Fetch papers
    papers = fetcher.fetch_papers(
        query=args.query,
        max_results=args.max_results,
        days_back=args.days_back
    )

    if not papers:
        logger.warning("No new papers fetched. Exiting.")
        return

    # Save papers
    output_path = fetcher.save_papers(papers, append=not args.overwrite)

    # Get and display statistics
    stats = fetcher.get_statistics(papers)

    logger.info("\n" + "="*70)
    logger.info("FETCH COMPLETE")
    logger.info("="*70)
    logger.info(f"Total papers fetched: {stats['total_papers']}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Date range: {stats['date_range']['earliest'][:10]} to {stats['date_range']['latest'][:10]}")
    logger.info(f"\nSummary statistics:")
    logger.info(f"  Avg length: {stats['summary_stats']['avg_length']:.0f} characters")
    logger.info(f"  Min length: {stats['summary_stats']['min_length']} characters")
    logger.info(f"  Max length: {stats['summary_stats']['max_length']} characters")
    logger.info(f"\nAuthor statistics:")
    logger.info(f"  Total authors: {stats['author_stats']['total_authors']}")
    logger.info(f"  Avg authors/paper: {stats['author_stats']['avg_authors_per_paper']:.1f}")
    logger.info(f"\nTop categories:")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {cat}: {count} papers")
    logger.info("="*70)

    # Show total papers in dataset
    all_papers = fetcher.load_papers()
    logger.info(f"\nTotal papers in dataset: {len(all_papers)}")


if __name__ == "__main__":
    main()

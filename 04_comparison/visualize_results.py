"""
Visualize comparison results.
Generates charts for presentations and demos.
"""
import json
import argparse
from pathlib import Path
import logging
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available. Install with: pip install matplotlib")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_comparison_results(filepath: Path) -> Dict:
    """Load comparison results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_comparison_chart(comparison: Dict, output_file: Path):
    """
    Create bar chart comparing all three models.

    Args:
        comparison: Comparison dict from benchmark_all
        output_file: Path to save chart image
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib is required for visualization")
        logger.info("Install with: pip install matplotlib")
        return

    models = comparison.get('models', [])
    if len(models) < 2:
        logger.error("Need at least 2 models to compare")
        return

    # Extract metrics
    metrics_data = comparison.get('metrics', {})

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('arXiv RAG System: Model Comparison', fontsize=16, fontweight='bold')

    # Metric 1: Relevance Score (higher is better)
    if 'relevance_score' in metrics_data:
        ax = axes[0, 0]
        values = list(metrics_data['relevance_score']['values'].values())
        colors = ['#e74c3c', '#f39c12', '#27ae60'][:len(values)]

        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Relevance Score', fontsize=11)
        ax.set_title('Relevance Score (Higher = Better)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=10)

    # Metric 2: Average Latency (lower is better)
    if 'avg_latency_ms' in metrics_data:
        ax = axes[0, 1]
        values = list(metrics_data['avg_latency_ms']['values'].values())
        colors = ['#e74c3c', '#f39c12', '#27ae60'][:len(values)]

        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title('Average Latency (Lower = Better)', fontsize=12, fontweight='bold')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}ms', ha='center', va='bottom', fontsize=10)

    # Metric 3: Total Cost
    if 'total_cost_usd' in metrics_data:
        ax = axes[1, 0]
        values = list(metrics_data['total_cost_usd']['values'].values())
        colors = ['#e74c3c', '#f39c12', '#27ae60'][:len(values)]

        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Cost (USD)', fontsize=11)
        ax.set_title('Total Cost for Test Set', fontsize=12, fontweight='bold')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.4f}', ha='center', va='bottom', fontsize=10)

    # Metric 4: Response Length
    if 'avg_response_length' in metrics_data:
        ax = axes[1, 1]
        values = list(metrics_data['avg_response_length']['values'].values())
        colors = ['#e74c3c', '#f39c12', '#27ae60'][:len(values)]

        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Characters', fontsize=11)
        ax.set_title('Average Response Length', fontsize=12, fontweight='bold')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    # Adjust layout and save
    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Chart saved to {output_file}")

    plt.close()


def create_improvement_chart(comparison: Dict, output_file: Path):
    """Create chart showing improvements over baseline."""
    if not MATPLOTLIB_AVAILABLE or 'improvements_vs_baseline' not in comparison:
        return

    improvements = comparison['improvements_vs_baseline']

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Improvements Over Baseline', fontsize=16, fontweight='bold')

    models = list(improvements.keys())
    metrics = ['relevance_improvement_pct', 'latency_improvement_pct']
    metric_labels = ['Relevance (%)', 'Latency (%)']

    x = range(len(models))
    width = 0.35

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [improvements[model].get(metric, 0) for model in models]
        offset = width * (i - 0.5)
        bars = ax.bar([xi + offset for xi in x], values,
                      width, label=label, alpha=0.8, edgecolor='black')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}%', ha='center',
                   va='bottom' if height > 0 else 'top', fontsize=9)

    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Improvement chart saved to {output_file}")
    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Visualize comparison results")
    parser.add_argument(
        '--comparison-file',
        type=Path,
        default=Path('04_comparison/results.json'),
        help='Path to comparison results JSON'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('04_comparison'),
        help='Output directory for charts'
    )

    args = parser.parse_args()

    if not args.comparison_file.exists():
        logger.error(f"Comparison file not found: {args.comparison_file}")
        logger.info("Run benchmark_all.py first to generate comparison results")
        return

    # Load comparison results
    comparison = load_comparison_results(args.comparison_file)

    # Create charts
    create_comparison_chart(comparison, args.output_dir / 'comparison_chart.png')

    if 'improvements_vs_baseline' in comparison:
        create_improvement_chart(comparison, args.output_dir / 'improvement_chart.png')

    logger.info("\nVisualization complete!")
    logger.info(f"Charts saved to {args.output_dir}")


if __name__ == "__main__":
    main()

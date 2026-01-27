# Fireworks arXiv Research Assistant

**A complete demonstration of fine-tuning progression: Baseline → SFT → RFT**

This project showcases the measurable improvements gained through supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) on a practical arXiv research assistant. Perfect for GTM demos showing the value of fine-tuning.

## 🎯 Project Overview

Build and compare three versions of an arXiv RAG assistant:

1. **Baseline**: Vanilla Llama-3.1-8B with RAG
2. **SFT**: Fine-tuned on 100 arXiv Q&A examples
3. **RFT**: Preference-optimized on 50 preference pairs

**Expected Results:**
- 📈 15-25% improvement in relevance scores
- 🎯 Better paper citations and accuracy
- ⚡ Comparable latency (~1-2s per query)
- 💰 Total cost: ~$10-15 for complete pipeline

## 🚀 Quick Start

```bash
# 1. Setup
./setup.sh

# 2. Add API keys to .env
# FIREWORKS_API_KEY=your_key
# PINECONE_API_KEY=your_key
# PINECONE_ENVIRONMENT=us-east-1-aws

# 3. Run complete pipeline
bash run_pipeline.sh
```

Or run each step individually (see [Complete Workflow](#complete-workflow) below).

## 📁 Project Structure

```
fireworks-arxiv-assistant/
│
├── 00_data_preparation/          # arXiv data pipeline
│   ├── fetch_papers.py            # Fetch from arXiv API
│   ├── embed_papers.py            # Store in Pinecone
│   ├── generate_training_data.py  # Create 100 SFT examples
│   └── generate_preference_data.py # Create 50 RFT pairs
│
├── 01_baseline/                   # Baseline system
│   ├── base_rag.py                # RAG with vanilla model
│   ├── evaluate_base.py           # Measure baseline
│   └── test_queries.json          # 20 test questions
│
├── 02_sft_model/                  # Supervised Fine-Tuning
│   ├── train_sft.py               # Train SFT model (~$3, 1hr)
│   ├── sft_rag.py                 # RAG with SFT model
│   └── evaluate_sft.py            # Measure SFT accuracy
│
├── 03_rft_model/                  # Reinforcement Fine-Tuning
│   ├── train_rft.py               # Train RFT model (~$5, 1hr)
│   ├── rft_rag.py                 # RAG with RFT model
│   └── evaluate_rft.py            # Measure RFT accuracy
│
├── 04_comparison/                 # The GTM money shot
│   ├── benchmark_all.py           # Compare all 3 models
│   ├── visualize_results.py       # Generate charts
│   └── results.json               # Auto-generated metrics
│
├── 05_demo/                       # Customer demos
│   └── interactive_demo.py        # Live Q&A interface
│
└── utils/                         # Shared utilities
    ├── fireworks_client.py        # Fireworks API wrapper
    ├── pinecone_client.py         # Pinecone operations
    └── metrics.py                 # Evaluation helpers
```

## 🔄 Complete Workflow

### Step 1: Data Preparation (~15-30 min, ~$5-8)

```bash
# Fetch 100 papers from arXiv
python 00_data_preparation/fetch_papers.py --max-results 100

# Embed and store in Pinecone (use Fireworks for lower cost)
python 00_data_preparation/embed_papers.py --provider fireworks

# Generate 100 SFT training examples
python 00_data_preparation/generate_training_data.py --num-examples 100

# Generate 50 RFT preference pairs
python 00_data_preparation/generate_preference_data.py --num-pairs 50
```

**Outputs:**
- `data/papers.jsonl`: Fetched papers
- `data/sft_training.jsonl`: SFT training data
- `data/rft_training.jsonl`: RFT preference pairs
- Vectors in Pinecone for RAG

**Cost:** ~$5-8 total

### Step 2: Baseline Evaluation (~5 min, ~$0.50)

```bash
# Evaluate vanilla model on 20 test queries
python 01_baseline/evaluate_base.py
```

**Outputs:**
- `data/test_results/base_results.json`: Individual query results
- `data/test_results/base_evaluation.json`: Aggregate metrics

### Step 3: Train SFT Model (~30-60 min, ~$3)

```bash
# Train SFT model
python 02_sft_model/train_sft.py train \
  --training-file data/sft_training.jsonl \
  --n-epochs 3

# Or start training and check status later
python 02_sft_model/train_sft.py train --no-wait
python 02_sft_model/train_sft.py status <job_id>
```

**Outputs:**
- `data/models/sft_model_id.txt`: Fine-tuned model ID

**Cost:** ~$3, Time: 30-60 minutes

### Step 4: Evaluate SFT Model (~5 min, ~$0.50)

```bash
# Evaluate SFT model
python 02_sft_model/evaluate_sft.py
```

**Outputs:**
- `data/test_results/sft_results.json`
- `data/test_results/sft_evaluation.json`

### Step 5: Train RFT Model (~30-60 min, ~$5)

```bash
# Train RFT on top of SFT model
python 03_rft_model/train_rft.py \
  --training-file data/rft_training.jsonl \
  --n-epochs 3
```

**Outputs:**
- `data/models/rft_model_id.txt`: Final model ID

**Cost:** ~$5, Time: 30-60 minutes

### Step 6: Evaluate RFT Model (~5 min, ~$0.50)

```bash
# Evaluate final RFT model
python 03_rft_model/evaluate_rft.py
```

**Outputs:**
- `data/test_results/rft_results.json`
- `data/test_results/rft_evaluation.json`

### Step 7: Compare All Models (< 1 min)

```bash
# Generate comparison report
python 04_comparison/benchmark_all.py

# Generate visualizations
python 04_comparison/visualize_results.py
```

**Outputs:**
- `04_comparison/results.json`: Comparison metrics
- `04_comparison/comparison_chart.png`: Visual comparison
- `04_comparison/improvement_chart.png`: Improvement over baseline

### Step 8: Run Interactive Demo

```bash
# Interactive Q&A with all three models
python 05_demo/interactive_demo.py --mode interactive

# Or run scripted demo
python 05_demo/interactive_demo.py --mode demo
```

## 📊 Expected Results

| Metric | Baseline | SFT | RFT | Improvement |
|--------|----------|-----|-----|-------------|
| Relevance Score | 0.65 | 0.75 | 0.82 | +26% |
| Citation Rate | 45% | 70% | 85% | +89% |
| Avg Latency | 1800ms | 1900ms | 1850ms | +3% |
| Cost per Query | $0.0025 | $0.0025 | $0.0025 | 0% |

*Actual results may vary based on dataset and configuration*

## 💰 Total Cost Breakdown

| Phase | Task | Cost | Time |
|-------|------|------|------|
| Data Prep | Fetch papers | Free | 5 min |
| Data Prep | Embed papers | $0.01 | 1 min |
| Data Prep | Generate SFT data | $2-3 | 10 min |
| Data Prep | Generate RFT data | $3-5 | 10 min |
| Training | Train SFT model | ~$3 | 30-60 min |
| Training | Train RFT model | ~$5 | 30-60 min |
| Evaluation | Test all models | $1.50 | 15 min |
| **TOTAL** | **Complete Pipeline** | **~$15-20** | **~2-3 hours** |

## 🎓 Key Features

### Data Preparation
- ✅ Retry logic with exponential backoff
- ✅ Rate limiting (respects API limits)
- ✅ Automatic deduplication
- ✅ Resume capability for interrupted runs
- ✅ Cost tracking and estimation
- ✅ Multiple embedding providers (Fireworks/OpenAI)

### Model Training
- ✅ Automated dataset upload
- ✅ Progress monitoring
- ✅ Background training support
- ✅ Configurable hyperparameters
- ✅ Cost estimates

### Evaluation
- ✅ Comprehensive metrics (relevance, latency, cost)
- ✅ Citation tracking
- ✅ Side-by-side comparison
- ✅ Statistical analysis
- ✅ Visual charts

### Demo
- ✅ Interactive CLI interface
- ✅ Side-by-side model comparison
- ✅ Example questions
- ✅ Scripted demo mode

## 🔧 Configuration

### API Keys (.env file)

```bash
# Required
FIREWORKS_API_KEY=your_fireworks_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=arxiv-papers

# Optional (only if using OpenAI embeddings)
OPENAI_API_KEY=your_openai_key
```

Get API keys:
- Fireworks: https://fireworks.ai/
- Pinecone: https://www.pinecone.io/
- OpenAI: https://platform.openai.com/

### Customization

**Fetch more/fewer papers:**
```bash
python 00_data_preparation/fetch_papers.py --max-results 200
```

**Use different base model:**
```bash
python 02_sft_model/train_sft.py train \
  --base-model "accounts/fireworks/models/llama-v3p1-70b-instruct"
```

**Adjust training epochs:**
```bash
python 02_sft_model/train_sft.py train --n-epochs 5 --learning-rate 1e-5
```

**Change embedding provider:**
```bash
python 00_data_preparation/embed_papers.py --provider openai
```

## 📚 Documentation

Detailed documentation for each module:
- [Data Preparation](00_data_preparation/README.md) - Complete data pipeline guide
- API Reference - See docstrings in each module

## 🐛 Troubleshooting

### Missing API Keys
```
Error: Missing FIREWORKS_API_KEY
```
**Solution:** Add your API key to `.env` file

### Pinecone Index Not Found
```
Error: Index 'arxiv-papers' not found
```
**Solution:** Index is auto-created on first run of `embed_papers.py`

### Model ID File Not Found
```
FileNotFoundError: Model ID file not found
```
**Solution:** Train the model first (SFT before RFT)

### Out of Memory
```
Error: CUDA out of memory
```
**Solution:** This shouldn't happen as training is done on Fireworks servers

## 🤝 Contributing

This is a demo project showcasing Fireworks AI fine-tuning capabilities. Feel free to fork and adapt for your use case!

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- **Fireworks AI** for fine-tuning infrastructure
- **Pinecone** for vector database
- **arXiv** for open access to research papers

---

Built with ❤️ to demonstrate the power of fine-tuning with Fireworks AI
# fireworks-arxiv-assistant

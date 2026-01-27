#!/bin/bash
# One-command setup script for Fireworks arXiv Assistant

set -e  # Exit on error

echo "=========================================="
echo "Fireworks arXiv Assistant Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo ""
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Create data directory
echo ""
echo "Creating data directory..."
mkdir -p data
echo "✓ Data directory created"

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  IMPORTANT: Please edit .env and add your API keys:"
    echo "   - FIREWORKS_API_KEY (required)"
    echo "   - PINECONE_API_KEY (required)"
    echo "   - PINECONE_ENVIRONMENT (required)"
    echo "   - OPENAI_API_KEY (optional)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "✓ .env file exists"
fi

echo ""
echo "=========================================="
echo "Setup Complete! 🚀"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Add your API keys to .env file"
echo ""
echo "3. Run the data preparation pipeline:"
echo "   python 00_data_preparation/fetch_papers.py"
echo "   python 00_data_preparation/embed_papers.py"
echo "   python 00_data_preparation/generate_training_data.py"
echo "   python 00_data_preparation/generate_preference_data.py"
echo ""
echo "Or use CLI arguments for more control:"
echo "   python 00_data_preparation/fetch_papers.py --max-results 50 --days-back 30"
echo ""
echo "For help with any script:"
echo "   python 00_data_preparation/fetch_papers.py --help"
echo ""

#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== DeepChess Setup ==="

# Create virtual environment
python3 -m venv .venv
echo "Virtual environment created."

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create checkpoint directory
mkdir -p checkpoints

echo ""
echo "Setup complete!"
echo "  Play:  ./play.sh [easy|normal|hard]"
echo "  Train: ./train.sh [--iterations N ...]"

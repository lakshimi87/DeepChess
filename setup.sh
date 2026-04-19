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

# Build the native C++ acceleration extension (non-fatal if it fails —
# the Python fallback will be used instead).
if ./build_ext.sh; then
	echo "Native extension built."
else
	echo "Warning: native extension build failed — falling back to pure Python."
fi

echo ""
echo "Setup complete!"
echo "  Play:  ./play.sh [easy|normal|hard]"
echo "  Train: ./train.sh [--iterations N ...]"

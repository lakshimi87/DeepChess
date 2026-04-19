#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== DeepChess Setup ==="
VENV_DIR="$HOME/venvs/torch"

if [ -d "$VENV_DIR" ]; then
    echo "venv already exists at $VENV_DIR"
    read -p "Recreate? [y/N] " ans
    if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
        rm -rf "$VENV_DIR"
    else
        echo "Skipping venv creation. Installing packages into existing venv..."
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip
        pip install torch numpy pygame-ce
        echo "Done!"
        exit 0
    fi
fi

echo "Creating virtual environment at $VENV_DIR ..."
mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"

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

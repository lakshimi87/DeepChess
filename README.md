# DeepChess

AI chess game with a neural network trained via self-play (AlphaZero-style)
and a classical minimax fallback engine.

## Features

- **Neural engine** — 12.5M-parameter ResNet (10 residual blocks, 128 filters)
  with policy + value heads, guided by Monte Carlo Tree Search (MCTS)
- **Classical engine** — minimax with alpha-beta pruning, quiescence search,
  and piece-square tables (available immediately, no training required)
- **Native C++ acceleration** — hot-path routines (board encoding, move
  indexing, PUCT selection) compiled via pybind11; pure-Python fallback
  when the extension isn't built
- **Three difficulty levels** — easy / normal / hard
- **Self-play training** — run `train.sh` repeatedly to strengthen the neural
  engine; each run resumes from the latest checkpoint
- **pygame-ce GUI** — board rendering, click-to-move, legal move hints,
  promotion dialog, captured pieces, move history

## Quick Start

```bash
# 1. Install dependencies and build the C++ extension
./setup.sh

# 2. Play (uses classical engine until you train a model)
./play.sh              # normal difficulty
./play.sh easy
./play.sh hard

# 3. Train the neural engine (repeat to keep improving)
./train.sh
./train.sh --iterations 50 --simulations 400
```

If you ever need to rebuild the native extension by itself:

```bash
./build_ext.sh
```

## Training

Each invocation of `train.sh` loads the latest checkpoint from `checkpoints/`
and continues training.  Interrupt with Ctrl-C at any time — the current
progress is saved automatically.

```bash
# Defaults: 100 iterations, 10 games/iter, 200 MCTS sims/move,
# numbered snapshot every 10 iterations
./train.sh

# Customise anything
./train.sh --iterations 200 \
           --games-per-iter 20 \
           --simulations 400 \
           --batch-size 128 \
           --epochs 10 \
           --lr 0.001 \
           --checkpoint-every 5

# See all options
python -m src.train --help
```

`latest.pt` is refreshed every iteration so `play.sh` always picks up the
newest weights.  `model_iter_XXXX.pt` snapshots are only written every
`--checkpoint-every` iterations (default: 10) to keep disk usage bounded —
plus one final snapshot on the last iteration and on interrupt.

## Validation

Run ground truth tests to measure how well the model has learned:

```bash
# Test latest checkpoint against 20 curated positions
./validate_gt.sh

# More MCTS simulations for a fairer test
./validate_gt.sh --simulations 400

# Show accuracy across all saved checkpoints (training progress)
./validate_gt.sh --history
```

Tests include mate-in-1 puzzles, hanging piece captures, opening quality,
and value-head accuracy.  The classical engine is always run as a baseline
for comparison.

## In-Game Controls

| Key   | Action                      |
| ----- | --------------------------- |
| N     | New game                    |
| U     | Undo last move              |
| 1/2/3 | Set difficulty easy/normal/hard |
| Q     | Quit                        |

## Project Structure

```
src/
  board_utils.py   Board encoding (18x8x8) and move indexing (4672 moves)
  model.py         ChessNet — dual-head ResNet (policy + value)
  mcts.py          Monte Carlo Tree Search with PUCT selection
  engine.py        Unified engine (neural MCTS or classical minimax)
  train.py         Self-play training pipeline with checkpointing
  validate_gt.py   Ground truth validation (20 curated test positions)
  main.py          pygame-ce GUI
  paths.py         Project-root-relative path constants
  _ext/            Native C++ extension (pybind11) + loader

setup.py           setuptools build script for the C++ extension
build_ext.sh       One-shot wrapper to (re)build the extension
resources/         Chess piece images
checkpoints/       Saved model weights (created by setup.sh)
```

# DeepChess

AI chess game with a neural network trained via self-play (AlphaZero-style)
and a classical minimax fallback engine.

## Features

- **Neural engine** — 12.5M-parameter ResNet (10 residual blocks, 128 filters)
  with policy + value heads, guided by Monte Carlo Tree Search (MCTS)
- **Classical engine** — minimax with alpha-beta pruning, quiescence search,
  and piece-square tables (available immediately, no training required)
- **Three difficulty levels** — easy / normal / hard
- **Self-play training** — run `train.sh` repeatedly to strengthen the neural
  engine; each run resumes from the latest checkpoint
- **pygame-ce GUI** — board rendering, click-to-move, legal move hints,
  promotion dialog, captured pieces, move history

## Quick Start

```bash
# 1. Install dependencies
./setup.sh

# 2. Play (uses classical engine until you train a model)
./play.sh              # normal difficulty
./play.sh easy
./play.sh hard

# 3. Train the neural engine (repeat to keep improving)
./train.sh
./train.sh --iterations 50 --simulations 400
```

## Training

Each invocation of `train.sh` loads the latest checkpoint from `checkpoints/`
and continues training.  Interrupt with Ctrl-C at any time — the current
progress is saved automatically.

```bash
# Defaults: 100 iterations, 10 games/iter, 200 MCTS sims/move
./train.sh

# Customise anything
./train.sh --iterations 200 \
           --games-per-iter 20 \
           --simulations 400 \
           --batch-size 128 \
           --epochs 10 \
           --lr 0.001

# See all options
python train.py --help
```

After training, `play.sh` automatically picks up the neural engine.

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
board_utils.py   Board encoding (18x8x8) and move indexing (4672 moves)
model.py         ChessNet — dual-head ResNet (policy + value)
mcts.py          Monte Carlo Tree Search with PUCT selection
engine.py        Unified engine (neural MCTS or classical minimax)
train.py         Self-play training pipeline with checkpointing
validate_gt.py   Ground truth validation (20 curated test positions)
main.py          pygame-ce GUI
resources/       Chess piece images
checkpoints/     Saved model weights (created by train.sh)
```

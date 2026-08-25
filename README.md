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
# Defaults: 100 iterations, 200 games/iter, 400 MCTS sims/move, 200k-position
# replay buffer, ~3x sample reuse, arena match and ground-truth suite every
# 25 iterations
./train.sh

# Customise anything
./train.sh --iterations 200 \
           --games-per-iter 20 \
           --simulations 1600 \
           --batch-size 128 \
           --checkpoint-every 5

# See all options
python -m src.train --help
```

### Reading the output

The training loss is the least informative number printed.  A run that has
stopped learning posts a flat, healthy-looking loss curve forever, and a run
that is memorising its replay buffer posts a loss curve that keeps *improving*
while the net gets no stronger.  Three other numbers are what to read.

**The held-out line** evaluates the net on the positions this iteration's
self-play just produced, before any gradient step has touched them.  Nothing
is withheld from training to get it — the fresh batch is held out by
construction.  The gap against the training loss on the line below is the only
number in the loop that separates learning from memorising.  A held-out value
MSE above the printed predict-a-draw baseline is flagged: a value head that
scores worse than a constant is not merely uninformative on new positions, it
is feeding MCTS confident noise, which is worse for search than having no
value head at all.  The usual cause is too much training per unit of new data
— lower `--sample-reuse` or raise `--games-per-iter`.

**The ground-truth line** scores the net every `--gt-every` iterations on the
same fixed suite `validate_gt.sh` uses.  It is an *absolute* yardstick, which
is what makes it trustworthy: the arena below only ever compares the net to
another copy of itself.

**Target entropy** is the entropy of the MCTS visit distributions being used
as policy targets.  Policy cross-entropy can never fall below it, so the
`headroom` figure printed under the policy loss is the only part still
learnable.  Headroom near zero means the net already reproduces its targets
and training longer cannot help — only better targets can, which means more
`--simulations` or a stronger value head.  Target entropy close to the printed
uniform-over-legal figure means the search is spreading visits almost evenly
and the targets carry little information.

**The arena** plays the current net against a frozen reference every
`--eval-every` iterations and promotes the reference on a score of at least
`--eval-promote`.  Read the generation counter with the match's confidence
interval in hand: these nets draw ~85% of arena games, so a 30-game match
carries about +/-36 Elo of noise at 2 sigma and a 55% promotion threshold
fires on chance alone roughly one match in eight.  Because the reference is
only ever replaced on a win, that noise ratchets one way and the generation
counter climbs steadily on a net that is not improving at all.  The default of
200 games is the minimum that resolves the effect sizes this loop produces;
prefer the ground-truth line as the primary progress signal.

Learning-rate milestones (`--lr-milestones`) are **absolute** iteration
numbers, not offsets from a resume.  Set them for the whole intended run: once
the last one is behind you, every subsequent resume trains at the fully
decayed rate.  Training warns on startup when this has happened.

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

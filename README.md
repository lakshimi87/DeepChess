# DeepChess

AI chess game with a neural network trained via self-play (AlphaZero-style)
and a classical minimax fallback engine.

## Features

- **Neural engine** — dual-head ResNet (policy + value) guided by Monte Carlo
  Tree Search (MCTS).  The tower is configurable: 8x128 is 3.07M parameters,
  16x192 (the `train.sh` default) is 11.54M, and AlphaZero's 20x256 would be
  24.80M
- **Supervised bootstrap** — Stockfish-labelled Lichess positions to
  pre-train the value head before self-play starts (see *Bootstrapping* below)
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
# Defaults: 100 iterations, 200 games/iter, 400 MCTS sims/move, 1M-position
# replay buffer, ~3x sample reuse, 0.35 search-value blend, arena match and
# ground-truth suite every 25 iterations
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

## Bootstrapping (supervised pre-training)

Self-play from a random initialisation does not converge on one GPU.  The
`archive/run4_8x128_iter258` run is the evidence: 20 hours, 50,400 games,
6.1M positions, and a ground-truth score that random-walked between 41% and
63% for its last 175 iterations while target entropy sat flat from iteration
75 onward.

The reason is not model capacity — held-out policy CE ran *below* training CE
the whole way, and the policy head was within 0.15 nats of its own targets'
entropy, so there was nothing left for more parameters to fit.  The reason is
that the loop cannot bootstrap its own value head at this scale:

| | this project (run4) | AlphaGo Zero | AlphaZero chess |
| --- | --- | --- | --- |
| self-play games | 50,400 | 4.9M | 44M |
| gradient steps x batch | 24k x 256 | 700k x 2048 | 700k x 4096 |
| replay window | 100k positions (~820 games) | 500,000 games | — |
| hardware | 1 GPU | ~2000 TPU | 5000 TPUv1 + 64 TPUv2 |

Two of those gaps are ~1000x.  The one that actually bites is the value head:
57% of self-play games are drawn and every position in a drawn game is
labelled 0.0, so the constant predictor is genuinely loss-optimal and the head
converges to it — run4's held-out value MSE sat at or above the
predict-a-draw baseline for all 258 iterations.  AlphaGo Zero cannot hit this
failure mode at all, because Go has no draws and every one of its value
targets is +/-1.

The fix is to label positions with an engine that is already strong, so the
value target varies *per position* instead of per game.

```bash
# 1. Fetch Stockfish (no sudo, ~50 MB)
mkdir -p third_party && cd third_party && \
  curl -sL -o sf.tar https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-bmi2.tar && \
  tar xf sf.tar && rm sf.tar && cd ..

# 2. Download Lichess monthly dumps (one month ~= 14M usable positions)
./tools/download_lichess.sh 2017-04 2017-07 2017-10

# 3. Extract positions (waits for each download, then streams it)
./tools/extract_all.sh 2017-04 2017-07 2017-10

# 4. Label with Stockfish.  --watch keeps it running as new shards appear,
#    so extraction and labelling pipeline instead of running in sequence.
python tools/label_sf.py --in-dir data/bootstrap/positions \
                         --out-dir data/bootstrap/labels \
                         --depth 14 --workers 26 --watch 300

# 5. Pre-train, then hand the weights to the self-play loop
python -m src.pretrain --data-dir data/bootstrap/labels --epochs 3
cp checkpoints/pretrained.pt checkpoints/latest.pt
./train.sh --search-value-weight 0.35 --buffer-size 1000000
```

### Why the labelling is configured the way it is

**`--depth 14 --multipv 1`.**  Measured on a 20-core i7-14700F, multipv is by
far the most expensive knob: depth 14 costs 52 ms/position at multipv=1 and
427 ms at multipv=8, because multipv defeats alpha-beta pruning.  That 8x buys
a soft policy target — which the PGN already supplies for free, in the form of
the move a 1800+ player actually chose.  Spending it on volume instead is
worth more, since the replay window is the loop's largest deficit.  Expect
~160 positions/second with 26 workers.

**WDL, not centipawns.**  Stockfish's `UCI_ShowWDL` reports win/draw/loss
permille, so `2*expectation - 1` is already calibrated against the value
head's target semantics.  `tanh(cp/400)` needs a hand-tuned scale that
silently decides how much of the eval range saturates to +/-1.

**A policy floor.**  `--policy-floor` spreads a little target mass uniformly
over the legal moves.  MCTS explores in proportion to the prior, so a move
trained to exactly zero is one the search will never look at again, and a
one-hot target teaches exactly that.

### Draw handling in the self-play loop

Two flags carry the same fix into self-play, and both matter only once the
value head has been pre-trained — from a random init they amplify the net's
own noise:

- `--search-value-weight` (default 0.35) takes that fraction of each value
  target from the position's own MCTS root Q rather than the game outcome.
  Root Q varies ply to ply, so two drawn games stop sharing one label.
- Games cut off at `--max-moves` now take the search value outright.  They are
  unfinished, not drawn, and labelling several hundred of their positions 0.0
  was the largest remaining source of value-label noise.

`--buffer-size` also defaults to 1M rather than 200k.  Of every axis in the
table above, the replay window is the only one that costs RAM instead of GPU
time.


### Running unattended

Labelling the full corpus takes days, so the pipeline is built to outlive any
one shell.  `tools/next_round.sh` waits for the labelled corpus to reach each
size in `TARGETS`, pre-trains from scratch on everything available, plays the
match against the classical engine, and appends a row to `results.md`.

```bash
# Launch detached — survives logout, SSH drops, and closing the terminal.
setsid nohup ./tools/next_round.sh > logs_next_round.log 2>&1 < /dev/null &

./tools/status.sh     # where everything stands, any time
cat results.md        # one row per completed round
```

Each round re-trains from scratch rather than resuming.  The question a round
answers is what a given corpus size is worth, and a warm start would confound
that with however long the previous round trained.

`.run/*.pid` files let `status.sh` tell a running job from a finished one
without pattern-matching process lists, which otherwise reports the status
script's own command line as a match.

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

  pretrain.py      Supervised pre-training on Stockfish-labelled positions

tools/
  download_lichess.sh  Fetch Lichess monthly PGN dumps
  extract_all.sh       Drive extraction across downloaded months
  fetch_lichess.py     PGN stream -> sampled (FEN, played move, result)
  label_sf.py          Stockfish WDL + best-move labelling, resumable
  build_gt_suite.py    Generate the 837-position ground-truth suite
  compare_sizes.sh     Pre-train several tower sizes, compare held-out
  next_round.sh        Unattended retrain+measure as the corpus grows
  status.sh            One-shot state of the whole pipeline

setup.py           setuptools build script for the C++ extension
build_ext.sh       One-shot wrapper to (re)build the extension
resources/         Chess piece images
checkpoints/       Saved model weights (created by setup.sh)
data/bootstrap/    Downloaded PGN, extracted positions, labels (gitignored)
```

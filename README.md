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
./train.sh --optimizer adamw --warmup-iters 10 --value-weight 1.0 \
           --anchor-data-dir data/bootstrap/labels --anchor-frac 0.5 \
           --eval-every 5 --gt-every 5 --gt-abort-drop 0.10 \
           --buffer-size 400000
```

Every flag on that `train.sh` line is there because run5 collapsed without it;
see [Handing a pre-trained net to self-play](#handing-a-pre-trained-net-to-self-play).

`latest.pt` is the only checkpoint `play.sh` and `src/engine.py` look for, and
nothing writes it except `src/train.py` and that `cp`.  `next_round.sh` writes
`checkpoints/pretrained_<tag>.pt` instead, so after a round ladder finishes,
re-point it by hand at the round you actually want to play or resume from —
otherwise the engine silently keeps serving whichever net was copied last:

```bash
cp checkpoints/pretrained_70M.pt checkpoints/latest.pt   # strongest round so far
python -m src.validate_gt --games 0                      # confirm it is the one you meant
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
  Root Q varies ply to ply, so two *decided* games stop sharing one label per
  side.  It is applied to decided games only — see below for why applying it
  to draws was actively harmful.
- Games cut off at `--max-moves` take the search value outright.  They are
  unfinished, not drawn, and labelling several hundred of their positions 0.0
  was the largest remaining source of value-label noise.  A game is now far
  less likely to get there: `play_game` and `play_match` end on the fifty-move
  rule, the same test MCTS already used in `_is_terminal_fast`, rather than on
  `is_game_over()`'s default of the seventy-five-move rule.  Those games are
  a draw (0.0) and are counted on their own line, not as truncations — a run
  drowning in fifty-move draws is in a different state from one running out
  of moves.
- `--adjudicate-material` (default 5 pawns held for `--adjudicate-plies` 20)
  awards a decided game without consulting the network.  The margin does two
  separable jobs, and `--no-adjudicate-label` keeps only the first: it stays
  the arbiter that scores the resignation calibration's audit games, while no
  longer writing a result of its own.  Setting the margin to 0 switches off
  both, and the calibration then has nothing to score against — audit games
  run with every early stop off end ~95% drawn, so every resignation reads as
  a false positive and the threshold stays pinned at its default.  The
  fingerprint in the log is the two numbers on the `Resign calib` line being
  equal.
- `--syzygy-path` / `--syzygy-pieces` end a game on a tablebase verdict once
  the board is down to that many men.  Unlike the two early stops it also
  fires in audit games: it is not a stop, it is the finish reached sooner.
  Cursed wins and blessed losses read as draws.  At this strength it fires in
  0 of 180 self-play games — see `tools/endgame_reach.py`.
- **MCTS sees repetitions.**  The descent copies the board with
  `stack=False`, and `_is_terminal_fast` used to skip the repetition scan on
  the grounds that a missed repetition only affects rare leaves.  Measured, it
  was 21 of 24 games: a side that believed it was winning repeated the
  position, the search agreed every time, and the game was drawn at a median
  of thirteen men.  A move returning to a position already in the game is now
  struck off at the root (perpetual check exempt; the ban lifts if every legal
  move repeats), and a repetition met during descent scores 0.  With the same
  weights, checkmates went from 3 of 24 to 10 of 24.  Cost is a frozenset
  lookup, not the stack scan that was dropped for being ~30% of per-move CPU.
  Boards built from a FEN carry no move stack, so positional suites are
  unaffected and stay comparable with earlier runs.

`--buffer-size` defaults to 1M rather than 200k.  Of every axis in the table
above, the replay window is the only one that costs RAM instead of GPU time —
but note that a position is stored with a *dense* 4672-slot policy target, so
it costs ~24 KB, and a full 1M-position buffer is ~24 GB of resident memory.
Size it against the RAM you actually have.

### Handing a pre-trained net to self-play

Pre-training works — the 70M-position round scores 91% on the ground-truth
suite and 22-0-8 against the classical baseline.  Feeding that net to the
self-play loop at defaults destroys it: run5 went from 91% to 38% on the suite
in five iterations while every loss it printed fell monotonically, because the
degenerate fixed point it was converging on *is* the loss minimum.  The
mechanism, in order:

1. **The pre-training LR is not the self-play LR.**  `src/pretrain.py` ends a
   OneCycle schedule at 1e-5 under AdamW.  `--lr` then defaulted to 0.02 under
   fresh-momentum SGD — 2000x the step size the weights had converged at, and
   the first iteration spends it on a replay buffer holding one iteration of
   self-play.  Use `--optimizer adamw` and `--warmup-iters`.
2. **Resignation is measured against an absolute threshold.**
   `--resign-threshold -0.9` needs the value head to still output large
   magnitudes.  One iteration in, run5's max|V| was 0.36, so resignation went
   from 74% of games to 1% and never fired again — and resignation was the
   only thing keeping decided games from grinding to the 50-move rule and
   labelling all of their positions a draw.  `--adjudicate-material` supplies
   decisive labels from material alone, so it keeps working through exactly
   the collapse that silences resignation.

   **This is still open, and it is the binding constraint.**  The threshold is
   an absolute bound on root Q while root Q's scale rides on the value head's,
   so recalibrating *which* threshold to use (`8922732`'s fix) cannot help once
   no threshold is reachable at all: with mean|V| at 0.24, -0.98 is not a
   bound the search can cross.  run8's resignation died at iteration 15 by the
   same mechanism, leaving adjudication — "who has more material" — to supply
   46-56% of decisive labels for the next 85 iterations, which is where
   Endgame lost 15 points.  Removing the material label instead does not work
   either: measured over four iterations, the loop collapses to all-draw
   labels.  Both horns are measured; what self-play cannot supply at this
   strength is an outcome that is informative *and* unbiased.

   The repair was not where this section expected.  Root Q does *not* follow
   mean|V| down — it is a visit-weighted mean over a search that finds nothing
   in a lost position, so it stays extreme while the head's scale halves, and
   the head keeps ranking a lost side below a holding one (AUC 0.87 -> 0.70
   over six iterations).  What fails is that the calibration window spans five
   iterations, the scale moves inside it, and rows recorded at different
   scales get scored against one absolute candidate — with rows that do not
   fire counting in neither numerator nor denominator, so the rate freezes on
   the stale rows that still do.  Each audit row now carries the scale it was
   measured against and the window is restated in current units before any
   rate is computed.  The threshold then tracks 0.98 -> 0.49 instead of
   pinning at -0.84.  What remains is conversion: even with resignation
   healthy, a third of games end on the fifty-move rule and a fifth run out of
   moves.
3. **Blending root Q into a *draw* label contracts the value head.**  For a
   target `(1-w)*z + w*V` the MSE fixed point is `V = z` for any `w`, so on a
   draw (`z = 0`) blending does not escape the constant-0 fixed point — it
   just pulls V toward 0 by a factor of `w` per iteration, out of the net's
   own output and nothing external.  With 90% of games drawn that is a
   geometric contraction: run5's mean|V| fell 0.669 -> 0.043, ~0.58x per
   iteration.  Finished draws now take a plain 0.0.
4. **Nothing in the loop is anchored outside it.**  The policy target is the
   net's own search, the value target is the game its own moves produced.
   `--anchor-data-dir` mixes the Stockfish-labelled shards back into every
   training step; they are the only fixed quantity in the project.  It also
   stops the policy head being flattened: the MCTS visit targets carry ~2.07
   nats against pre-training's ~1.24, so training on them alone is a
   downgrade, and run5's own prior went from 1.52 to 2.48 nats.  The
   *fraction* is load-bearing and 0.25 is not enough — see below.
5. **Every diagnostic printed and none of them acted.**  The suite read 38%
   at iteration 5 and the run kept training.  `--gt-abort-drop` now stops it
   and restores the best checkpoint.  `--eval-every` establishes the arena reference before the
   first iteration instead of on the arena's first firing (which cost a whole
   extra `--eval-every` before any comparison), and the reference file is
   checked against the current architecture — run5 carried run4's 8x128
   reference into a 16x192 run, so the arena would have killed every worker
   the first time it fired.

Each iteration now reports `Value scale : mean|V| … max|V| …`.  That is the
number to watch: it is the one quantity that exposes a collapsing value head
while the loss curve still looks healthy, and it warns outright once max|V|
drops below `|--resign-threshold|`.

**Do not read `<-- WORSE THAN GUESSING` as a collapse signal.**  It compares
held-out value MSE against the variance of the outcome labels, and a
*correctly* pre-trained head fails that comparison: it is calibrated to
Stockfish's WDL and stays confident, while a single game's outcome is noisy,
so its squared error exceeds the labels' own variance.  It fired on every
iteration of run5 — and it fires on a healthy run from the same checkpoint too
(value MSE 0.47 against a 0.35 baseline at iteration 2, with mean|V| holding
at 0.57 and resignation at 70%).  A flag that is on in both cases distinguishes
nothing, and treating it as an alarm is what trains you to ignore the log.
Trend mean|V|, the resignation rate, and the suite instead.


### What `--anchor-frac` is worth

Points 1-5 keep the net from collapsing.  They do not, on their own, keep it
from *leaking*, and the difference took three runs to separate.  All three
started from the same 70M pre-trained net (759/837) and differed in one flag:

| run | change | @0 | @5 | @10 | @15 | @20 | slope (pts/iter) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run5 | none of the above | 758 | 319 | — | — | — | collapse |
| run6 | all of the above, `--anchor-frac 0.25` | 759 | 750 | 735 | — | — | **-2.40 +/- 0.35** |
| run7 | run6 + `--adjudicate-material 10 --adjudicate-plies 40` | 759 | 737 | — | — | — | — |
| run8 | run6 + `--anchor-frac 0.5` | 759 | 764 | 758 | 744 | 755 | **-0.56 +/- 0.44** |

**Read the slope, never two readings.**  Re-scoring one fixed checkpoint three
times gave 758/759/759 -- SD 0.6, which is the suite's measuring noise.  But
consecutive readings *within* run8 scatter with SD 7.4, because the net really
does wander between iterations.  Judging an iteration-to-iteration difference
against the 0.6 floor makes every wiggle look like a 5-sigma event: during run8
this produced three confident and contradictory calls in a row -- "settled" at
iteration 3, "leak stopped" at 10, "leak confirmed" at 15 -- from data that
supported none of them.  Four or more readings and a slope with its standard
error, or say nothing.

run6 does not collapse and still loses 2.40 +/- 0.35 points an iteration,
which `--gt-abort-drop` never sees: a 10% relative drop from 759 is ~30
iterations off at that rate.  A guard tuned to run5's cliff does not see a
leak, and tightening it enough to catch one would fire on the SD-7.4 scatter
above instead.  That gap is documented, not closed.

The leak is the anchor fraction.  At 0.5 the slope is -0.56 +/- 0.44 points an
iteration over 20 iterations -- not separable from flat (t = 1.3) -- against
run6's -2.40 +/- 0.35 at 0.25 (t = 6.9).  Both runs sit in a 744-764 band the
whole way; what differs is whether the band drifts.  0.5 also holds the value head's output scale up — mean|V| at
iteration 5 was 0.44 against run6's 0.41, and max|V| stayed above
`|--resign-threshold|` so resignation kept firing at all instead of dying at
iteration 4.  Anchor rows cost ~1.4s per iteration at 0.5, against ~180s of
self-play, so the fraction is free in wall-clock terms and should be chosen on
the score alone.

run7 is here because it tests a wrong diagnosis worth recording.  The two
categories that lose points are Opening and Endgame, and adjudication awarding
a game at +5 pawns plainly truncates it before any endgame appears — so
loosening adjudication to +10 pawns for 40 plies should have recovered them.
It recovered nothing: Endgame read 69/91 and Opening 85/95 in *both* runs, bit
for bit.  Resignation truncates games just as adjudication does, and it was
ending 74%/64%/50% of run7's first three iterations, so swapping which
mechanism fires never changed how early games end.  What the looser threshold
did change was label quality: adjudication fell from 58% of games to 8%,
draw labels hit 85%, and Middlegame, Tactics and Eval gave up 13 points
between them.  The termination rule governs the draw rate and nothing else.

What is left after the leak is closed is a structural deficit in the same two
categories every time, Opening and Endgame, which doubling the anchor does not
touch (Opening read 85, 83, 84, 85 across the runs and Endgame 69, 69, 72, 66).  Self-play plays 200 games an
iteration from the same start position with root Dirichlet noise as the only
divergence, so a handful of openings carry thousands of positions each while
the anchor's opening rows are scattered thin across 8M samples of every phase.
That is an opening-diversity problem in the self-play generator, not an anchor
dose, and it is the next thing to fix.

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

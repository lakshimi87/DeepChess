"""How far down the board actually gets before a self-play game ends.

The tablebase probe fired in 0 of 180 games.  Either the games end before the
board simplifies, or they simplify and the probe is broken -- this tells which,
and at what piece count a table would have to start to be reached at all.
"""
import collections, sys
import chess, torch
sys.path.insert(0, '.')
from src.model import ChessNet
from src.mcts import MCTS
from src import perf

GAMES, SIMS, MAX_MOVES = 24, 200, 512
dev = torch.device("cuda")
model = ChessNet(num_res_blocks=16, num_filters=192)
model.load_state_dict(torch.load("checkpoints/pretrained_70M.pt",
                                 map_location="cpu")["model_state_dict"])
model = perf.to_inference(model, dev, half=True)
mcts = MCTS(model, dev, num_simulations=SIMS, batch_size=8, dirichlet_eps=0.25,
            ban_repetition="--allow-repetition" not in sys.argv)

reasons, pieces, minpieces, moves_all = collections.Counter(), [], [], []
for g in range(GAMES):
    b = chess.Board()
    n = 0
    lowest = 32
    stopped_early = False
    while not b.is_game_over(claim_draw=True) and n < MAX_MOVES:
        mv, _ = mcts.search(b, temperature=1.0 if n < 30 else 0.1, add_noise=True)
        if mv is None:
            stopped_early = True
            break
        b.push(mv)
        n += 1
        lowest = min(lowest, chess.popcount(b.occupied))
    # outcome() names the termination itself; hand-rolling the elif chain put
    # ten of twenty-four games in an "other" bucket that meant nothing.
    oc = b.outcome(claim_draw=True)
    if oc is not None:
        r = oc.termination.name.lower().replace("_", " ")
    elif n >= MAX_MOVES:
        r = "move limit"
    elif stopped_early:
        r = "search returned no move"
    else:
        r = "loop exited with game unfinished"
    reasons[r] += 1
    pieces.append(chess.popcount(b.occupied))
    minpieces.append(lowest)
    moves_all.append(n)
    print(f"  game {g+1:2d}: {r:22s} {n:3d} moves, "
          f"{chess.popcount(b.occupied):2d} men at end, {lowest:2d} men at fewest")

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(len(xs) * p))]

print(f"\nend reasons: {dict(reasons)}")
print(f"men at end     median {pct(pieces,.5)}  min {min(pieces)}  "
      f"10th pct {pct(pieces,.1)}")
print(f"fewest men seen  median {pct(minpieces,.5)}  min {min(minpieces)}  "
      f"10th pct {pct(minpieces,.1)}")
print(f"games ever reaching <=5 men: "
      f"{sum(1 for m in minpieces if m <= 5)}/{GAMES}")
print(f"games ever reaching <=7 men: "
      f"{sum(1 for m in minpieces if m <= 7)}/{GAMES}")
print(f"moves: median {pct(moves_all,.5)}  max {max(moves_all)}")

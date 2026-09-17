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
mcts = MCTS(model, dev, num_simulations=SIMS, batch_size=8, dirichlet_eps=0.25)

reasons, pieces, minpieces, moves_all = collections.Counter(), [], [], []
for g in range(GAMES):
    b = chess.Board()
    n = 0
    lowest = 32
    while not b.is_game_over(claim_draw=True) and n < MAX_MOVES:
        mv, _ = mcts.search(b, temperature=1.0 if n < 30 else 0.1, add_noise=True)
        if mv is None:
            break
        b.push(mv)
        n += 1
        lowest = min(lowest, chess.popcount(b.occupied))
    if b.is_checkmate():                     r = "checkmate"
    elif b.is_stalemate():                   r = "stalemate"
    elif b.is_insufficient_material():       r = "insufficient material"
    elif b.is_fifty_moves():                 r = "50-move rule"
    elif b.can_claim_threefold_repetition(): r = "threefold repetition"
    elif n >= MAX_MOVES:                     r = "move limit"
    else:                                    r = "other"
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

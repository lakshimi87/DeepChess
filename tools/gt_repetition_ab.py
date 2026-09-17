"""Does the in-line repetition scoring move the 837 suite?

daf231a claims: "Boards built from a FEN carry no move stack, so nothing is
banned and nothing is scored differently.  The 837 suite ... read exactly as
before, and stay comparable with run8-run10."

The first half holds -- repetition_history_keys() on a FEN board is empty, so
the root ban is inert.  But the descent also keeps path_keys, which catches a
repetition *within the line being searched*, and that does not need a move
stack.  This scores the same checkpoint with ban_repetition on and off and
reports which move tests disagree.
"""
import os, sys, torch, chess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import validate_gt as V
from src.mcts import MCTS
from src.model import ChessNet
from src.board_utils import encode_board

dev = torch.device("cuda")
ckpt = torch.load(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/pretrained_70M.pt", map_location=dev,
                  weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model = ChessNet(num_res_blocks=16, num_filters=192).to(dev)
model.load_state_dict(sd)
model.eval()

SIMS = 200

def choose(fen, ban):
	b = chess.Board(fen)
	m = MCTS(model, dev, num_simulations=SIMS, ban_repetition=ban)
	mv, _ = m.search(b, temperature=0.01)
	return mv.uci() if hasattr(mv, "uci") else str(mv)

diffs, on_pass, off_pass = [], 0, 0
for i, (cat, fen, acceptable, desc) in enumerate(V.MOVE_TESTS):
	a = choose(fen, True)
	b = choose(fen, False)
	oa, ob = a in acceptable, b in acceptable
	on_pass += oa
	off_pass += ob
	if a != b:
		diffs.append((cat, desc, a, b, oa, ob))
	if (i + 1) % 100 == 0:
		print(f"  ... {i+1}/{len(V.MOVE_TESTS)}  "
		      f"on={on_pass} off={off_pass} diffs={len(diffs)}", flush=True)

print(f"\nmove tests: {len(V.MOVE_TESTS)}")
print(f"  ban_repetition=True  passed {on_pass}")
print(f"  ban_repetition=False passed {off_pass}")
print(f"  different move chosen: {len(diffs)}")
for cat, desc, a, b, oa, ob in diffs:
	print(f"    [{cat}] {desc[:54]}")
	print(f"        on={a} ({'pass' if oa else 'FAIL'})  "
	      f"off={b} ({'pass' if ob else 'FAIL'})")

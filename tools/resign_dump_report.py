"""Read a --resign-dump file and answer the questions the log lines cannot.

train.py's `Resign calib` and `Resign reach` lines collapse the calibration
sample to a pair of rates.  A rate cannot separate the two ways calibration
fails, and they call for opposite fixes:

  unreachable    trigger values bunch near 0, so the threshold admits nobody.
                 Loosen it.
  verdict-poor   audit games end drawn, so almost nothing reads as lost and
                 the false-positive rate is high at every threshold.
                 Loosening only manufactures false positives.

It also shows the failure 1204e8a fixed: the window spans several iterations
and the scale moves inside it, so rows recorded at different scales get scored
together.  The fingerprint is an unchanged false-positive rate on consecutive
iterations while the fresh rows stop firing entirely.

    python tools/resign_dump_report.py logs_run13_resign.jsonl
"""
import json
import sys


def auc(rows):
	"""P(a lost side's trigger < a holding side's), ties at 0.5.

	Invariant to any monotone rescaling of root Q, so it says whether the
	value head still *ranks* a lost side below a holding one regardless of
	how far its output scale has shrunk.  That is the question a scale fix
	cannot answer and this one can.
	"""
	lost = [r[0] for r in rows if r[1]]
	held = [r[0] for r in rows if not r[1]]
	if not lost or not held:
		return None
	wins = sum((a < b) + 0.5 * (a == b) for a in lost for b in held)
	return wins / (len(lost) * len(held))


def med(xs):
	xs = sorted(xs)
	return xs[len(xs) // 2] if xs else float("nan")


def main(path):
	rows = [json.loads(line) for line in open(path)]
	prev = 0
	print(f"{'it':>3} {'scale':>6} {'mean|V|':>8} {'thr':>6} | "
	      f"{'win':>4} {'fresh':>5} {'med fresh':>10} {'fire%fresh':>11} "
	      f"{'AUC':>6} | {'FP% window':>10}")
	for d in rows:
		s = d["samples"]
		fresh = s[prev:] if len(s) > prev else s
		prev = len(s)
		sc, thr = d.get("scale"), d["threshold"]
		# The window restated on the current scale, as the code now scores it.
		keys = [(r[0] * sc / r[3] if (sc and len(r) > 3 and r[3]) else r[0],
		         r[1]) for r in s]
		fired = [lost for k, lost in keys if k <= thr]
		fp = (100.0 * sum(1 for lost in fired if not lost) / len(fired)
		      if fired else float("nan"))
		ff = (100.0 * sum(1 for r in fresh if r[0] <= thr) / len(fresh)
		      if fresh else float("nan"))
		a = auc(s)
		mv = d.get("mean_abs_v")
		print(f"{d['iteration'] + 1:>3} "
		      f"{sc if sc else float('nan'):>6.3f} "
		      f"{mv if mv else float('nan'):>8.3f} {thr:>+6.2f} | "
		      f"{len(s):>4} {len(fresh):>5} {med([r[0] for r in fresh]):>10.3f} "
		      f"{ff:>10.0f}% {a if a is not None else float('nan'):>6.3f} | "
		      f"{fp:>9.0f}%")

	print("\nRead it this way:")
	print("  fire%fresh at 0 while FP% window looks healthy  -> the window is")
	print("    stale; the threshold is fitted to a scale the loop has left.")
	print("  FP% high at every threshold, AUC near 0.5       -> verdict-poor;")
	print("    the value head is not ranking, so no threshold rule helps.")
	print("  AUC well above 0.5 and fire%fresh 0             -> reachability;")
	print("    the information to loosen is there and in reach.")


if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "logs_run13_resign.jsonl")

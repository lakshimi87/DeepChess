"""Per-iteration conversion reading from a training log.

TODO 4o's reading is the share of games that never reach a result the rules
would call: a fifty-move draw with men still on the board, or a game cut off
at --max-moves.  Those are the games whose value labels are either a 0.0 that
contradicts the position or the net's own root Q, which is the self-confirming
loop the project is trying to leave.

Resignation rate is deliberately *not* the reading here.  Resignation labels a
side that is already lost; it does not make a game decisive, and after
1204e8a it tracks the value head's scale on its own.  Adjudication is also
kept separate: it asserts an outcome rather than reaching one, so counting it
as "converted" would hide exactly what 4o is asking about.

Reads the `Game endings` lines, so it works on any run's log, live or
finished.

    python tools/conversion_trend.py logs_run13.log
"""
import re
import sys

LINE = re.compile(
	r"Game endings\s*:\s*resigned (\d+)/(\d+)"
	r"(?:.*?adjudicated (\d+)/\d+)?"
	r"(?:.*?tablebase (\d+)/\d+)?"
	r"(?:.*?fifty-move draw (\d+)/\d+)?"
	r"(?:.*?hit move limit (\d+)/\d+)?")


def rows(path):
	for line in open(path, errors="replace"):
		m = LINE.search(line)
		if not m:
			continue
		res, total, adj, tb, fifty, limit = (int(x) if x else 0
		                                     for x in m.groups())
		yield res, total, adj, tb, fifty, limit


def main(path):
	print(f"{'it':>3} {'games':>6} {'resign':>7} {'adjud':>6} {'syzygy':>7} "
	      f"{'50-move':>8} {'limit':>6} | {'UNRESOLVED':>10}")
	unresolved = []
	for i, (res, total, adj, tb, fifty, limit) in enumerate(rows(path), 1):
		# The reading: neither the rules nor the tables ever answered.
		u = 100.0 * (fifty + limit) / total
		unresolved.append(u)
		print(f"{i:>3} {total:>6} {100.0*res/total:>6.0f}% "
		      f"{100.0*adj/total:>5.0f}% {100.0*tb/total:>6.0f}% "
		      f"{100.0*fifty/total:>7.0f}% {100.0*limit/total:>5.0f}% | "
		      f"{u:>9.0f}%")
	if len(unresolved) >= 4:
		# Slope with a standard error, because section 5 says four readings
		# and a standard error or it is not a trend.
		n = len(unresolved)
		xs = list(range(n))
		mx = sum(xs) / n
		my = sum(unresolved) / n
		sxx = sum((x - mx) ** 2 for x in xs)
		b = sum((x - mx) * (y - my) for x, y in zip(xs, unresolved)) / sxx
		resid = [y - (my + b * (x - mx)) for x, y in zip(xs, unresolved)]
		if n > 2:
			s2 = sum(r * r for r in resid) / (n - 2)
			se = (s2 / sxx) ** 0.5
			print(f"\nunresolved slope {b:+.2f} +/- {se:.2f} %/iteration "
			      f"(t={b/se:+.2f}) over {n} readings")
	else:
		print(f"\n{len(unresolved)} readings -- section 5 wants 4 before "
		      f"any trend is spoken of.")


if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "logs_run13.log")

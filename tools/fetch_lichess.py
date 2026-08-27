"""Extract sampled positions from a Lichess PGN stream.

Reads PGN text on stdin (so the caller controls the source and can stream a
.zst straight off the network without ever landing it on disk):

    curl -sL https://database.lichess.org/standard/lichess_db_standard_rated_2017-04.pgn.zst \
        | zstd -dc | python tools/fetch_lichess.py --out-dir data/bootstrap/positions

Why sample rather than take every ply: consecutive plies of one game are
near-duplicates, and the first dozen plies of *every* game are the same
handful of book positions.  Training on the raw stream would spend most of
its gradient on openings it has already fitted.  ``--skip-plies`` drops the
book, ``--sample-every`` decorrelates what is left, and the FEN dedupe drops
transpositions that survive both.

Output is plain text (``FEN<TAB>played_move_uci<TAB>result``), sharded.  Positions are *not*
encoded here: a 20x8x8 float32 tensor is 5 KB, so 20M encoded positions would
be 100 GB.  FENs are ~60 bytes and re-encode in microseconds via the C++
extension, so the pipeline stores FENs and encodes on the fly at train time.
"""
import argparse
import os
import re
import sys
from multiprocessing import Pool

import chess

_ELO_RE = re.compile(r'^\[(White|Black)Elo "(\d+)"\]', re.M)
_RESULT_RE = re.compile(r'^\[Result "([^"]+)"\]', re.M)
_TC_RE = re.compile(r'^\[TimeControl "([^"]+)"\]', re.M)
# Strip lichess movetext decorations: comments {...}, NAGs ($1), move numbers,
# and the trailing result token.
_CLEAN_RE = re.compile(r'\{[^}]*\}|\$\d+|\d+\.(\.\.)?|[?!]+')
_RESULT_TOKEN = {"1-0", "0-1", "1/2-1/2", "*"}

_RESULT_VALUE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def _parse_game(job):
	"""Replay one game's SAN movetext and return sampled (fen, result) rows.

	Runs in a worker process: SAN parsing is the pipeline's bottleneck and is
	pure CPU, while the header filtering upstream is cheap regex work.
	"""
	movetext, result, skip_plies, sample_every, max_per_game = job
	tokens = [t for t in _CLEAN_RE.sub(" ", movetext).split()
	          if t and t not in _RESULT_TOKEN]
	if len(tokens) < skip_plies + 2:
		return []

	board = chess.Board()
	rows = []
	for ply, san in enumerate(tokens):
		sample = (ply >= skip_plies
		          and (ply - skip_plies) % sample_every == 0
		          and not board.is_game_over())
		fen = board.fen() if sample else None
		try:
			move = board.push_san(san)
		except (ValueError, AssertionError):
			break  # malformed or truncated movetext — keep what we have
		if sample:
			# The position *before* the move, paired with the move actually
			# played there.  That pairing is a free policy target: labelling
			# it with Stockfish instead would mean multipv, which costs 8x
			# the search time (measured) for a signal the PGN already has.
			rows.append((fen, move.uci(), result))
			if len(rows) >= max_per_game:
				break
	return rows


def _fen_key(fen):
	"""Hashable position identity: piece placement, turn, castling, ep.

	Deliberately excludes the halfmove/fullmove counters.  Two occurrences of
	the same position at different move numbers are the same training example
	for our purposes, and keeping both would reintroduce the correlation the
	sampling is meant to remove.
	"""
	return hash(fen.rsplit(" ", 2)[0])


def _iter_games(stream):
	"""Yield (headers, movetext) per game from a PGN text stream.

	A PGN game is a header block and a movetext block separated by a blank
	line, with a blank line after the movetext.  Splitting on that structure
	is far cheaper than python-chess's full reader, which matters because most
	games are rejected on rating before anything parses them.
	"""
	headers, movetext, in_moves = [], [], False
	for line in stream:
		if line.startswith("["):
			if in_moves:
				yield "".join(headers), " ".join(movetext)
				headers, movetext, in_moves = [], [], False
			headers.append(line)
		elif line.strip():
			in_moves = True
			movetext.append(line.strip())
		# blank lines are separators; state is flushed on the next header
	if headers and movetext:
		yield "".join(headers), " ".join(movetext)


def main():
	ap = argparse.ArgumentParser(description=__doc__,
	                             formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--out-dir", required=True)
	ap.add_argument("--min-elo", type=int, default=2000,
	                help="Both players must be at least this rating.  Below "
	                     "~1800 the played moves stop being a useful policy "
	                     "signal; the positions themselves stay fine, which is "
	                     "why this defaults low enough to keep volume up.")
	ap.add_argument("--max-positions", type=int, default=0,
	                help="Stop after this many deduped positions (0 = stream "
	                     "until stdin ends).")
	ap.add_argument("--skip-plies", type=int, default=12,
	                help="Drop the opening book, which is identical across "
	                     "millions of games.")
	ap.add_argument("--sample-every", type=int, default=5,
	                help="Keep every Nth ply after the book.")
	ap.add_argument("--max-per-game", type=int, default=12)
	ap.add_argument("--shard-size", type=int, default=1_000_000)
	ap.add_argument("--shard-start", type=int, default=0,
	                help="First shard number to write.  Extracting a second "
	                     "month into the same directory would otherwise "
	                     "restart at pos_00000.tsv and overwrite the first.")
	ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
	ap.add_argument("--no-draws-frac", type=float, default=1.0,
	                help="Fraction of drawn games to keep (1.0 = all).  Drawn "
	                     "games are ~50%% of the corpus but their game-result "
	                     "label carries the least information.")
	args = ap.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	seen = set()
	shard_idx, shard_rows, total, games_used, games_seen = (
		args.shard_start, [], 0, 0, 0)
	rng_counter = 0

	def flush():
		nonlocal shard_idx, shard_rows
		if not shard_rows:
			return
		path = os.path.join(args.out_dir, f"pos_{shard_idx:05d}.tsv")
		tmp = path + ".tmp"
		with open(tmp, "w") as fh:
			fh.write("".join(shard_rows))
		os.replace(tmp, path)          # atomic: a killed run leaves no partial shard
		print(f"  wrote {path}  ({len(shard_rows)} positions)", flush=True)
		shard_idx += 1
		shard_rows = []

	def jobs():
		nonlocal games_used, games_seen, rng_counter
		for headers, movetext in _iter_games(sys.stdin):
			games_seen += 1
			elos = _ELO_RE.findall(headers)
			if len(elos) != 2 or min(int(e[1]) for e in elos) < args.min_elo:
				continue
			m = _RESULT_RE.search(headers)
			if not m or m.group(1) not in _RESULT_VALUE:
				continue
			result = m.group(1)
			if result == "1/2-1/2" and args.no_draws_frac < 1.0:
				rng_counter += 1
				if (rng_counter % 100) >= args.no_draws_frac * 100:
					continue
			tc = _TC_RE.search(headers)
			if tc and tc.group(1) != "-":
				# Bullet moves are close to random under time pressure.
				base = tc.group(1).split("+")[0]
				if base.isdigit() and int(base) < 180:
					continue
			games_used += 1
			yield (movetext, _RESULT_VALUE[result],
			       args.skip_plies, args.sample_every, args.max_per_game)

	print(f"Extracting positions -> {args.out_dir}", flush=True)
	print(f"  min Elo {args.min_elo}, skip {args.skip_plies} plies, "
	      f"every {args.sample_every}th ply, <= {args.max_per_game}/game, "
	      f"{args.workers} workers", flush=True)

	with Pool(args.workers) as pool:
		for rows in pool.imap_unordered(_parse_game, jobs(), chunksize=64):
			for fen, uci, result in rows:
				key = _fen_key(fen)
				if key in seen:
					continue
				seen.add(key)
				shard_rows.append(f"{fen}\t{uci}\t{result}\n")
				total += 1
			if len(shard_rows) >= args.shard_size:
				flush()
				print(f"  {total:,} positions from {games_used:,} games "
				      f"({games_seen:,} scanned)", flush=True)
			if args.max_positions and total >= args.max_positions:
				break

	flush()
	print(f"Done: {total:,} unique positions from {games_used:,} games "
	      f"({games_seen:,} scanned) in {shard_idx} shards", flush=True)


if __name__ == "__main__":
	main()

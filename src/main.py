#!/usr/bin/env python3
"""DeepChess — pygame-ce GUI.

Usage:
    python -m src.main --difficulty normal
    ./play.sh easy
"""

import argparse
import os
import sys
import threading

import chess
import pygame

from .engine import Engine
from .paths import RESOURCES_DIR

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
SQUARE_SIZE = 80
BOARD_PX = SQUARE_SIZE * 8  # 640
PANEL_WIDTH = 260
WIDTH = BOARD_PX + PANEL_WIDTH  # 900
HEIGHT = BOARD_PX  # 640
FPS = 60

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
PANEL_BG = (48, 46, 43)
PANEL_TEXT = (200, 200, 200)
WHITE_CLR = (255, 255, 255)
DIVIDER = (80, 80, 80)

HIGHLIGHT_SEL = (255, 255, 0, 100)
HIGHLIGHT_LAST = (186, 202, 43, 100)
HIGHLIGHT_LEGAL_DOT = (0, 0, 0, 60)
HIGHLIGHT_LEGAL_RING = (0, 0, 0, 60)
HIGHLIGHT_CHECK = (235, 97, 80, 180)

FILES = "abcdefgh"
RANKS = "87654321"


# ===================================================================
# Game
# ===================================================================

class ChessGame:
	def __init__(self, difficulty="normal"):
		pygame.init()
		self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
		pygame.display.set_caption("DeepChess")
		self.clock = pygame.time.Clock()

		# Fonts
		self.font_lg = pygame.font.SysFont("Arial", 26, bold=True)
		self.font_md = pygame.font.SysFont("Arial", 20)
		self.font_sm = pygame.font.SysFont("Arial", 15)

		# Piece images
		self.pieces = {}
		self._load_pieces()

		# State
		self.difficulty = difficulty
		self.board = chess.Board()
		self.engine = Engine(difficulty=difficulty)
		self.player_color = chess.WHITE

		self.selected_sq = None
		self.legal_for_selected = []
		self.last_move = None
		self.ai_thinking = False
		self.game_over = False
		self.promotion_pending = None  # (from_sq, to_sq)
		self.promotion_rects = {}

	# ------------------------------------------------------------------
	# Asset loading
	# ------------------------------------------------------------------

	def _load_pieces(self):
		mapping = {
			(chess.PAWN, chess.WHITE): "w_Pawn.png",
			(chess.KNIGHT, chess.WHITE): "w_Knight.png",
			(chess.BISHOP, chess.WHITE): "w_Bishop.png",
			(chess.ROOK, chess.WHITE): "w_Rook.png",
			(chess.QUEEN, chess.WHITE): "w_Queen.png",
			(chess.KING, chess.WHITE): "w_King.png",
			(chess.PAWN, chess.BLACK): "b_Pawn.png",
			(chess.KNIGHT, chess.BLACK): "b_Knight.png",
			(chess.BISHOP, chess.BLACK): "b_Bishop.png",
			(chess.ROOK, chess.BLACK): "b_Rook.png",
			(chess.QUEEN, chess.BLACK): "b_Queen.png",
			(chess.KING, chess.BLACK): "b_King.png",
		}
		for key, fname in mapping.items():
			img = pygame.image.load(os.path.join(RESOURCES_DIR, fname))
			self.pieces[key] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))

	# ------------------------------------------------------------------
	# Coordinate helpers
	# ------------------------------------------------------------------

	@staticmethod
	def sq_to_screen(sq):
		col = chess.square_file(sq)
		row = 7 - chess.square_rank(sq)
		return col * SQUARE_SIZE, row * SQUARE_SIZE

	@staticmethod
	def screen_to_sq(pos):
		x, y = pos
		if x >= BOARD_PX or y >= BOARD_PX:
			return None
		return chess.square(x // SQUARE_SIZE, 7 - y // SQUARE_SIZE)

	# ------------------------------------------------------------------
	# Drawing
	# ------------------------------------------------------------------

	def _draw_board(self):
		for row in range(8):
			for col in range(8):
				colour = LIGHT_SQ if (row + col) % 2 == 0 else DARK_SQ
				pygame.draw.rect(
					self.screen, colour,
					(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
				)
		# Coordinate labels
		for col in range(8):
			c = DARK_SQ if col % 2 == 0 else LIGHT_SQ
			lbl = self.font_sm.render(FILES[col], True, c)
			self.screen.blit(lbl, (col * SQUARE_SIZE + SQUARE_SIZE - 12, BOARD_PX - 16))
		for row in range(8):
			c = LIGHT_SQ if row % 2 == 0 else DARK_SQ
			lbl = self.font_sm.render(RANKS[row], True, c)
			self.screen.blit(lbl, (2, row * SQUARE_SIZE + 1))

	def _draw_highlights(self):
		# Last move
		if self.last_move:
			for sq in (self.last_move.from_square, self.last_move.to_square):
				self._overlay(sq, HIGHLIGHT_LAST)

		# Selected square
		if self.selected_sq is not None:
			self._overlay(self.selected_sq, HIGHLIGHT_SEL)

		# Legal move indicators
		for move in self.legal_for_selected:
			x, y = self.sq_to_screen(move.to_square)
			surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
			if self.board.piece_at(move.to_square):
				# Capture ring
				pygame.draw.circle(
					surf, HIGHLIGHT_LEGAL_RING,
					(SQUARE_SIZE // 2, SQUARE_SIZE // 2),
					SQUARE_SIZE // 2, 5,
				)
			else:
				# Quiet-move dot
				pygame.draw.circle(
					surf, HIGHLIGHT_LEGAL_DOT,
					(SQUARE_SIZE // 2, SQUARE_SIZE // 2),
					SQUARE_SIZE // 7,
				)
			self.screen.blit(surf, (x, y))

		# Check highlight
		if self.board.is_check() and not self.game_over:
			king_sq = self.board.king(self.board.turn)
			if king_sq is not None:
				self._overlay(king_sq, HIGHLIGHT_CHECK)

	def _overlay(self, sq, colour):
		x, y = self.sq_to_screen(sq)
		surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
		surf.fill(colour)
		self.screen.blit(surf, (x, y))

	def _draw_pieces(self):
		for sq in chess.SQUARES:
			piece = self.board.piece_at(sq)
			if piece:
				x, y = self.sq_to_screen(sq)
				self.screen.blit(self.pieces[(piece.piece_type, piece.color)], (x, y))

	# ------------------------------------------------------------------
	# Side panel
	# ------------------------------------------------------------------

	def _draw_panel(self):
		pygame.draw.rect(self.screen, PANEL_BG, (BOARD_PX, 0, PANEL_WIDTH, HEIGHT))

		y = 20
		# Title
		self.screen.blit(self.font_lg.render("DeepChess", True, WHITE_CLR), (BOARD_PX + 20, y))
		y += 35
		pygame.draw.line(self.screen, DIVIDER, (BOARD_PX + 15, y), (BOARD_PX + PANEL_WIDTH - 15, y))
		y += 15

		# Engine mode + difficulty
		mode_txt = f"Engine: {self.engine.mode.capitalize()}"
		self.screen.blit(self.font_sm.render(mode_txt, True, (140, 140, 140)), (BOARD_PX + 20, y))
		y += 20
		diff_txt = f"Difficulty: {self.difficulty.capitalize()}"
		self.screen.blit(self.font_sm.render(diff_txt, True, (140, 140, 140)), (BOARD_PX + 20, y))
		y += 30

		# Status
		if self.game_over:
			status = self._game_over_text()
			self.screen.blit(self.font_md.render(status, True, (255, 200, 100)), (BOARD_PX + 20, y))
		elif self.ai_thinking:
			dots = "." * (1 + (pygame.time.get_ticks() // 400) % 3)
			self.screen.blit(
				self.font_md.render(f"Thinking{dots}", True, (150, 200, 255)),
				(BOARD_PX + 20, y),
			)
		else:
			turn = "White" if self.board.turn == chess.WHITE else "Black"
			who = "(You)" if self.board.turn == self.player_color else "(AI)"
			self.screen.blit(
				self.font_md.render(f"{turn}'s turn {who}", True, PANEL_TEXT),
				(BOARD_PX + 20, y),
			)

		if self.board.is_check() and not self.game_over:
			y += 25
			self.screen.blit(self.font_md.render("Check!", True, (255, 100, 100)), (BOARD_PX + 20, y))

		# Move counter
		y = 200
		self.screen.blit(
			self.font_sm.render(f"Move: {self.board.fullmove_number}", True, PANEL_TEXT),
			(BOARD_PX + 20, y),
		)

		# Captured pieces
		y += 30
		pygame.draw.line(self.screen, DIVIDER, (BOARD_PX + 15, y), (BOARD_PX + PANEL_WIDTH - 15, y))
		y += 10
		self.screen.blit(self.font_md.render("Captured", True, PANEL_TEXT), (BOARD_PX + 20, y))
		y += 28

		initial = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
		for color, label in [(chess.BLACK, "White took:"), (chess.WHITE, "Black took:")]:
			self.screen.blit(self.font_sm.render(label, True, PANEL_TEXT), (BOARD_PX + 20, y))
			y += 20
			xp = BOARD_PX + 20
			for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
				diff = max(0, initial.get(pt, 0) - len(self.board.pieces(pt, color)))
				for _ in range(diff):
					small = pygame.transform.smoothscale(self.pieces[(pt, color)], (24, 24))
					self.screen.blit(small, (xp, y))
					xp += 25
					if xp > BOARD_PX + PANEL_WIDTH - 30:
						y += 25
						xp = BOARD_PX + 20
			y += 30

		# Move history (last few moves)
		y += 5
		pygame.draw.line(self.screen, DIVIDER, (BOARD_PX + 15, y), (BOARD_PX + PANEL_WIDTH - 15, y))
		y += 10
		self.screen.blit(self.font_md.render("History", True, PANEL_TEXT), (BOARD_PX + 20, y))
		y += 25

		san_moves = self._get_san_history()
		# Show last ~8 full moves
		start = max(0, len(san_moves) - 16)
		for i in range(start, len(san_moves), 2):
			move_num = i // 2 + 1
			white_san = san_moves[i]
			black_san = san_moves[i + 1] if i + 1 < len(san_moves) else ""
			line = f"{move_num}. {white_san}  {black_san}"
			self.screen.blit(self.font_sm.render(line, True, (160, 160, 160)), (BOARD_PX + 20, y))
			y += 18
			if y > HEIGHT - 130:
				break

		# Controls
		y = HEIGHT - 110
		pygame.draw.line(self.screen, DIVIDER, (BOARD_PX + 15, y), (BOARD_PX + PANEL_WIDTH - 15, y))
		controls = [
			"N — New Game",
			"U — Undo Move",
			"1/2/3 — Easy / Normal / Hard",
			"Q — Quit",
		]
		for i, txt in enumerate(controls):
			self.screen.blit(
				self.font_sm.render(txt, True, (120, 120, 120)),
				(BOARD_PX + 20, y + 12 + i * 22),
			)

	def _game_over_text(self):
		if self.board.is_checkmate():
			winner = "Black" if self.board.turn == chess.WHITE else "White"
			return f"Checkmate! {winner} wins"
		if self.board.is_stalemate():
			return "Stalemate — Draw"
		if self.board.is_insufficient_material():
			return "Draw — Insufficient material"
		if self.board.can_claim_fifty_moves():
			return "Draw — 50-move rule"
		if self.board.can_claim_threefold_repetition():
			return "Draw — Repetition"
		return "Game Over"

	def _get_san_history(self):
		"""Replay the move stack to get SAN notation."""
		sans = []
		tmp = chess.Board()
		for move in self.board.move_stack:
			sans.append(tmp.san(move))
			tmp.push(move)
		return sans

	# ------------------------------------------------------------------
	# Promotion dialog
	# ------------------------------------------------------------------

	def _draw_promotion_dialog(self):
		# Dim overlay
		dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
		dim.fill((0, 0, 0, 150))
		self.screen.blit(dim, (0, 0))

		dw, dh = 320, 100
		dx = (BOARD_PX - dw) // 2
		dy = (HEIGHT - dh) // 2

		pygame.draw.rect(self.screen, PANEL_BG, (dx, dy, dw, dh), border_radius=8)
		pygame.draw.rect(self.screen, WHITE_CLR, (dx, dy, dw, dh), 2, border_radius=8)

		lbl = self.font_md.render("Promote to:", True, WHITE_CLR)
		self.screen.blit(lbl, (dx + (dw - lbl.get_width()) // 2, dy + 5))

		self.promotion_rects = {}
		for i, pt in enumerate([chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]):
			x = dx + 20 + i * 75
			y = dy + 35
			img = pygame.transform.smoothscale(self.pieces[(pt, self.player_color)], (60, 60))
			self.screen.blit(img, (x, y))
			self.promotion_rects[pt] = pygame.Rect(x, y, 60, 60)

	def _handle_promotion_click(self, pos):
		for pt, rect in self.promotion_rects.items():
			if rect.collidepoint(pos):
				from_sq, to_sq = self.promotion_pending
				move = chess.Move(from_sq, to_sq, promotion=pt)
				self.board.push(move)
				self.last_move = move
				self.selected_sq = None
				self.legal_for_selected = []
				self.promotion_pending = None
				self._check_game_over()
				if not self.game_over:
					self._start_ai()
				return True
		return False

	# ------------------------------------------------------------------
	# Click handling
	# ------------------------------------------------------------------

	def _handle_click(self, pos):
		if self.ai_thinking or self.game_over or self.promotion_pending:
			return

		sq = self.screen_to_sq(pos)
		if sq is None:
			return

		# If a piece is already selected, try to move
		if self.selected_sq is not None:
			target_move = None
			for m in self.legal_for_selected:
				if m.to_square == sq:
					target_move = m
					break

			if target_move:
				# Check for promotion
				piece = self.board.piece_at(self.selected_sq)
				if piece and piece.piece_type == chess.PAWN:
					to_rank = chess.square_rank(sq)
					if (piece.color == chess.WHITE and to_rank == 7) or \
					   (piece.color == chess.BLACK and to_rank == 0):
						self.promotion_pending = (self.selected_sq, sq)
						return

				self.board.push(target_move)
				self.last_move = target_move
				self.selected_sq = None
				self.legal_for_selected = []
				self._check_game_over()
				if not self.game_over:
					self._start_ai()
				return

			# Re-select own piece
			piece = self.board.piece_at(sq)
			if piece and piece.color == self.player_color:
				self.selected_sq = sq
				self.legal_for_selected = [m for m in self.board.legal_moves if m.from_square == sq]
				return

			# Deselect
			self.selected_sq = None
			self.legal_for_selected = []
			return

		# Nothing selected — pick a piece
		piece = self.board.piece_at(sq)
		if piece and piece.color == self.player_color and self.board.turn == self.player_color:
			self.selected_sq = sq
			self.legal_for_selected = [m for m in self.board.legal_moves if m.from_square == sq]

	# ------------------------------------------------------------------
	# AI
	# ------------------------------------------------------------------

	def _start_ai(self):
		if self.board.turn != self.player_color and not self.game_over:
			self.ai_thinking = True
			threading.Thread(target=self._ai_worker, daemon=True).start()

	def _ai_worker(self):
		move = self.engine.get_move(self.board)
		if move:
			self.board.push(move)
			self.last_move = move
		self.ai_thinking = False
		self._check_game_over()

	def _check_game_over(self):
		if self.board.is_game_over():
			self.game_over = True

	# ------------------------------------------------------------------
	# Game actions
	# ------------------------------------------------------------------

	def _new_game(self):
		if self.ai_thinking:
			return
		self.board = chess.Board()
		self.selected_sq = None
		self.legal_for_selected = []
		self.last_move = None
		self.game_over = False
		self.promotion_pending = None

	def _undo(self):
		if self.ai_thinking:
			return
		if len(self.board.move_stack) >= 2:
			self.board.pop()
			self.board.pop()
		elif len(self.board.move_stack) == 1:
			self.board.pop()
		self.last_move = self.board.peek() if self.board.move_stack else None
		self.selected_sq = None
		self.legal_for_selected = []
		self.game_over = False

	def _set_difficulty(self, diff):
		if self.ai_thinking:
			return
		self.difficulty = diff
		self.engine = Engine(difficulty=diff)

	# ------------------------------------------------------------------
	# Main loop
	# ------------------------------------------------------------------

	def run(self):
		running = True
		while running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False

				elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
					if self.promotion_pending:
						self._handle_promotion_click(event.pos)
					else:
						self._handle_click(event.pos)

				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_q:
						running = False
					elif event.key == pygame.K_n:
						self._new_game()
					elif event.key == pygame.K_u:
						self._undo()
					elif event.key == pygame.K_1:
						self._set_difficulty("easy")
					elif event.key == pygame.K_2:
						self._set_difficulty("normal")
					elif event.key == pygame.K_3:
						self._set_difficulty("hard")

			# Draw
			self._draw_board()
			self._draw_highlights()
			self._draw_pieces()
			self._draw_panel()
			if self.promotion_pending:
				self._draw_promotion_dialog()
			pygame.display.flip()
			self.clock.tick(FPS)

		pygame.quit()
		sys.exit()


# ===================================================================
# Entry point
# ===================================================================

def main():
	parser = argparse.ArgumentParser(description="DeepChess — play against the AI")
	parser.add_argument(
		"--difficulty", type=str, default="normal",
		choices=["easy", "normal", "hard"],
		help="AI difficulty level",
	)
	args = parser.parse_args()
	game = ChessGame(difficulty=args.difficulty)
	game.run()


if __name__ == "__main__":
	main()

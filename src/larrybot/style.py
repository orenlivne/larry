"""Style vector extraction from a player's games and per-move scoring.

The style model captures four dimensions of a player's tendencies:
  - aggression: preference for checks, attacks near the opponent's king
  - material_weight: tendency to capture / maintain material advantage
  - activity_weight: preference for moves that increase piece mobility
  - centrality: preference for placing pieces on central squares
"""

from __future__ import annotations

import chess

from .config import StyleVector
from .pgn_utils import iter_player_positions

# Central squares (d4, e4, d5, e5) and extended center
_CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
_EXTENDED_CENTER = _CENTER | {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4,
    chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}

# King-zone squares (3x3 around each king position)
def _king_zone(board: chess.Board, color: chess.Color) -> set[int]:
    king_sq = board.king(color)
    if king_sq is None:
        return set()
    kr, kf = chess.square_rank(king_sq), chess.square_file(king_sq)
    zone = set()
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            r, f = kr + dr, kf + df
            if 0 <= r <= 7 and 0 <= f <= 7:
                zone.add(chess.square(f, r))
    return zone


def _mobility(board: chess.Board) -> int:
    """Number of legal moves for the side to move."""
    return board.legal_moves.count()


def _centrality_score(board: chess.Board, color: chess.Color) -> float:
    """Fraction of the player's pieces on central / extended-central squares."""
    pieces = board.piece_map()
    own = [sq for sq, p in pieces.items() if p.color == color]
    if not own:
        return 0.0
    center_count = sum(1 for sq in own if sq in _CENTER) * 2
    ext_count = sum(1 for sq in own if sq in _EXTENDED_CENTER and sq not in _CENTER)
    return (center_count + ext_count) / (2 * len(own))


def _aggression_score(board: chess.Board, move: chess.Move, color: chess.Color) -> float:
    """Score how aggressive a move is (0-1)."""
    score = 0.0
    opp_color = not color

    # Giving check is aggressive
    board.push(move)
    if board.is_check():
        score += 0.4
    board.pop()

    # Attacking opponent's king zone
    opp_king_zone = _king_zone(board, opp_color)
    to_sq = move.to_square
    if to_sq in opp_king_zone:
        score += 0.3

    # Piece sacrifices (moving to a square attacked by opponent)
    if board.is_attacked_by(opp_color, to_sq):
        moving_piece = board.piece_at(move.from_square)
        if moving_piece and moving_piece.piece_type in (
            chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN,
        ):
            score += 0.3

    return min(1.0, score)


def _material_score(board: chess.Board, move: chess.Move) -> float:
    """Score material gain of a move (0-1)."""
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            values = {
                chess.PAWN: 0.2, chess.KNIGHT: 0.5, chess.BISHOP: 0.5,
                chess.ROOK: 0.7, chess.QUEEN: 1.0, chess.KING: 0.0,
            }
            return values.get(captured.piece_type, 0.0)
    return 0.0


def _activity_score(board: chess.Board, move: chess.Move) -> float:
    """Score mobility increase after the move (0-1)."""
    mob_before = _mobility(board)
    board.push(move)
    # Opponent's mobility doesn't matter; flip to get our next mobility
    board.push(chess.Move.null())
    mob_after = _mobility(board) if not board.is_check() else mob_before
    board.pop()
    board.pop()

    if mob_before == 0:
        return 0.5
    ratio = mob_after / mob_before
    return min(1.0, max(0.0, ratio - 0.5))


def _centrality_move_score(board: chess.Board, move: chess.Move, color: chess.Color) -> float:
    """Score how much a move improves centrality (0-1)."""
    before = _centrality_score(board, color)
    board.push(move)
    after = _centrality_score(board, color)
    board.pop()
    diff = after - before
    return min(1.0, max(0.0, 0.5 + diff * 5))


def compute_move_features(
    board: chess.Board, move: chess.Move, color: chess.Color,
) -> tuple[float, float, float, float]:
    """Return (aggression, material, activity, centrality) for a move."""
    return (
        _aggression_score(board, move, color),
        _material_score(board, move),
        _activity_score(board, move),
        _centrality_move_score(board, move, color),
    )


def score_move(
    board: chess.Board,
    move: chess.Move,
    color: chess.Color,
    style: StyleVector,
) -> float:
    """Score how well a move matches the player's style (higher = better fit)."""
    agg, mat, act, cen = compute_move_features(board, move, color)
    return (
        style.aggression * agg
        + style.material_weight * mat
        + style.activity_weight * act
        + style.centrality * cen
    )


class StyleAnalyzer:
    """Extract a StyleVector from a player's PGN games."""

    @staticmethod
    def from_pgn(pgn_path: str, player_name: str) -> StyleVector:
        """Compute aggregate style from every move the player made."""
        totals = [0.0, 0.0, 0.0, 0.0]
        count = 0

        for board, move, color in iter_player_positions(pgn_path, player_name):
            features = compute_move_features(board, move, color)
            for i in range(4):
                totals[i] += features[i]
            count += 1

        if count == 0:
            return StyleVector()

        avg = [t / count for t in totals]
        # Normalize to [0, 1] — values are already roughly in this range
        return StyleVector(
            aggression=min(1.0, max(0.0, avg[0])),
            material_weight=min(1.0, max(0.0, avg[1])),
            activity_weight=min(1.0, max(0.0, avg[2])),
            centrality=min(1.0, max(0.0, avg[3])),
        )

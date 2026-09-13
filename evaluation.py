"""Position evaluation.

The evaluator returns a score in centipawns from the perspective of the side
to move (negamax convention): positive means the player about to move is
better.

The default evaluator is PeSTO (Ronald Friederich's tapered piece-square
tables), a well-tested handcrafted baseline that is hard to beat without a
trained model. The Evaluator interface is deliberately tiny so a learned
model (e.g. a small NNUE loaded from model/weights.pt) can be dropped in
later without touching the search.
"""

from __future__ import annotations

import chess

from pesto import EG_PST, EG_VALUE, MG_PST, MG_VALUE

# Contribution of each piece type to the game phase (PeSTO convention):
# pawn 0, knight 1, bishop 1, rook 2, queen 4. A full starting position
# sums to 24 (phase 24 = pure middlegame, 0 = pure endgame).
_PHASE_INC = [0, 0, 1, 1, 2, 4, 0]  # indexed by chess.PieceType (1..6)
_MAX_PHASE = 24


def _build_tables() -> tuple[list, list]:
    """Precompute material+PST tables in python-chess square indexing.

    PeSTO tables are written from White's visual perspective (index 0 = a8),
    while python-chess uses 0 = a1. For a white piece the lookup square is
    sq ^ 56 (vertical flip); for black it is sq unchanged.

    Returns MG and EG tables shaped [piece_type][color][square] where color
    follows python-chess (False=black, True=white) via int() indexing.
    """
    mg: list = [None] * 7
    eg: list = [None] * 7
    for pt in range(1, 7):
        i = pt - 1
        mg_black = [MG_VALUE[i] + MG_PST[i][sq] for sq in range(64)]
        mg_white = [MG_VALUE[i] + MG_PST[i][sq ^ 56] for sq in range(64)]
        eg_black = [EG_VALUE[i] + EG_PST[i][sq] for sq in range(64)]
        eg_white = [EG_VALUE[i] + EG_PST[i][sq ^ 56] for sq in range(64)]
        mg[pt] = (mg_black, mg_white)
        eg[pt] = (eg_black, eg_white)
    return mg, eg


_MG_TABLE, _EG_TABLE = _build_tables()

# Raw material values, used by the search for move ordering and pruning
# margins (indexed by chess.PieceType, king given a huge value so it sorts
# first as a capture victim in MVV-LVA — it can never actually be captured).
PIECE_VALUE = [0, MG_VALUE[0], MG_VALUE[1], MG_VALUE[2], MG_VALUE[3], MG_VALUE[4], 20000]

# Distance of each square from the board center (0 in the middle, 6 in the
# corners). Used by the mop-up term to drive the losing king to the edge.
_CENTER_MANHATTAN = [
    max(3 - sq % 8, sq % 8 - 4, 0) + max(3 - sq // 8, sq // 8 - 4, 0)
    for sq in range(64)
]


class PestoEvaluator:
    """Tapered material + piece-square evaluation (PeSTO weights)."""

    def __init__(self, tempo: int = 10):
        # Small bonus for having the move; helps avoid oscillating scores
        # between plies and mildly encourages activity.
        self.tempo = tempo

    def evaluate(self, board: chess.Board) -> int:
        mg = 0
        eg = 0
        phase = 0
        material = 0  # white minus black, kings excluded
        occupied_white = board.occupied_co[chess.WHITE]
        for pt, bb in (
            (chess.PAWN, board.pawns),
            (chess.KNIGHT, board.knights),
            (chess.BISHOP, board.bishops),
            (chess.ROOK, board.rooks),
            (chess.QUEEN, board.queens),
            (chess.KING, board.kings),
        ):
            mg_t = _MG_TABLE[pt]
            eg_t = _EG_TABLE[pt]
            inc = _PHASE_INC[pt]
            while bb:
                sq = (bb & -bb).bit_length() - 1
                bb &= bb - 1
                if occupied_white >> sq & 1:
                    mg += mg_t[1][sq]
                    eg += eg_t[1][sq]
                    if pt != chess.KING:
                        material += PIECE_VALUE[pt]
                else:
                    mg -= mg_t[0][sq]
                    eg -= eg_t[0][sq]
                    if pt != chess.KING:
                        material -= PIECE_VALUE[pt]
                phase += inc

        if phase > _MAX_PHASE:  # early promotions can exceed 24
            phase = _MAX_PHASE
        score = (mg * phase + eg * (_MAX_PHASE - phase)) // _MAX_PHASE

        # Mop-up: with a decisive material lead, plain material counting sees
        # every winning move as equal and the engine drifts into rule draws.
        # Reward cornering the losing king and approaching with our own, with
        # the term fading in as the game empties out (phase -> 0).
        if material >= 400 or material <= -400:
            winner = chess.WHITE if material > 0 else chess.BLACK
            loser_k = board.king(not winner)
            winner_k = board.king(winner)
            if loser_k is not None and winner_k is not None:
                kings_dist = chess.square_distance(winner_k, loser_k)
                mop = 5 * (4 * _CENTER_MANHATTAN[loser_k] + 2 * (7 - kings_dist))
                mop = mop * (_MAX_PHASE - phase) // _MAX_PHASE
                score += mop if winner == chess.WHITE else -mop

        # Convert from White's perspective to side-to-move perspective.
        return (score if board.turn else -score) + self.tempo

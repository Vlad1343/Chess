"""Search: iterative deepening negamax with alpha-beta pruning.

Techniques, chosen because they are the proven core of every strong small
engine (Boychesser, Kaggle FIDE winners, sunfish-NNUE):

- iterative deepening with a soft/hard time budget
- fail-soft alpha-beta negamax
- transposition table (exact/lower/upper bounds, mate-score ply adjustment)
- quiescence search (captures only; full evasions when in check)
- move ordering: TT move, MVV-LVA captures, promotions, killers, history
- null-move pruning (with zugzwang guard), check extension
- late move reductions for quiet moves

The root loop updates the best move incrementally, so aborting on the hard
time limit still returns the best fully-searched move so far.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chess

from evaluation import _MG_TABLE, PIECE_VALUE

MATE = 100_000
MATE_BOUND = 99_000  # scores beyond this are "mate in N"
_INF = 1_000_000

_MAX_PLY = 128
_TIME_CHECK_MASK = 2047  # check the clock every 2048 nodes

_TT_EXACT, _TT_LOWER, _TT_UPPER = 0, 1, 2

# Move-ordering priorities (higher = searched earlier).
_ORDER_TT = 1 << 30
_ORDER_CAPTURE = 1 << 24
_ORDER_PROMO = 1 << 23
_ORDER_KILLER1 = 1 << 22
_ORDER_KILLER2 = (1 << 22) - 1


class _TimeUp(Exception):
    pass


@dataclass
class SearchResult:
    move: chess.Move | None
    score: int = 0
    depth: int = 0
    nodes: int = 0
    elapsed: float = 0.0
    pv: list = field(default_factory=list)


class Searcher:
    def __init__(self, evaluator, max_depth: int = 64, tt_max_entries: int = 1 << 21):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.tt_max_entries = tt_max_entries
        self.tt: dict = {}
        self.eval_cache: dict = {}
        self._reset_heuristics()

    def _reset_heuristics(self) -> None:
        self.killers = [[None, None] for _ in range(_MAX_PLY)]
        self.history: dict = {}
        self.nodes = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        board: chess.Board,
        soft_time: float = 1.0,
        hard_time: float | None = None,
        max_depth: int | None = None,
        game_history: dict | None = None,
    ) -> SearchResult:
        """Search `board` and return the best move found.

        soft_time: no new depth iteration is started after this much time.
        hard_time: the search is aborted outright at this point.
        game_history: transposition keys of positions already seen in the
            real game (the tournament interface sends bare FENs, so the
            board's own move stack can't know about them). Any search node
            matching one is scored as a draw: reaching it again would be at
            least a twofold repetition, which the opponent can convert to a
            threefold claim.
        """
        start = time.perf_counter()
        self._deadline = start + (hard_time if hard_time is not None else soft_time * 2.5)
        self._reset_heuristics()
        self._game_history = game_history or {}
        if len(self.tt) > self.tt_max_entries:
            self.tt.clear()
        if len(self.eval_cache) > self.tt_max_entries:
            self.eval_cache.clear()

        depth_limit = min(max_depth or self.max_depth, self.max_depth)
        moves = list(board.legal_moves)
        if not moves:
            return SearchResult(move=None, elapsed=time.perf_counter() - start)

        best_move = moves[0]
        best_score = -_INF
        completed_depth = 0
        base_stack_len = len(board.move_stack)

        try:
            for depth in range(1, depth_limit + 1):
                if time.perf_counter() - start > soft_time and completed_depth >= 1:
                    break
                # Aspiration window: expect the score near last iteration's;
                # a miss (fail low/high) triggers one full-width re-search.
                if depth >= 4 and abs(best_score) < MATE_BOUND:
                    alpha, beta = best_score - 60, best_score + 60
                else:
                    alpha, beta = -_INF, _INF
                score, move = self._search_root(board, moves, depth, best_move, alpha, beta)
                if move is None or score <= alpha or score >= beta:
                    score, move = self._search_root(board, moves, depth, best_move,
                                                    -_INF, _INF)
                if move is not None:
                    best_move, best_score = move, score
                completed_depth = depth
                # No point going deeper once a forced mate is found.
                if abs(best_score) >= MATE_BOUND:
                    break
        except _TimeUp:
            pass
        finally:
            # A hard-time abort raises from deep inside the tree; unwind the
            # caller's board back to the root position.
            while len(board.move_stack) > base_stack_len:
                board.pop()

        return SearchResult(
            move=best_move,
            score=best_score,
            depth=completed_depth,
            nodes=self.nodes,
            elapsed=time.perf_counter() - start,
            pv=self._extract_pv(board, best_move),
        )

    # ------------------------------------------------------------------
    # Root
    # ------------------------------------------------------------------

    def _search_root(self, board, moves, depth, prev_best, alpha=-_INF, beta=_INF):
        # Search the previous iteration's best move first: it usually stays
        # best, and it guarantees an abort mid-iteration still leaves a move
        # searched at the deepest depth.
        moves.sort(key=lambda m: (m != prev_best, -self._order_score(board, m, 0)))

        best_move = None
        for move in moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, 1, True)
            board.pop()
            if score > alpha:
                alpha = score
                best_move = move
                if alpha >= beta:  # fail high on an aspiration window
                    break
        return alpha, best_move

    # ------------------------------------------------------------------
    # Negamax
    # ------------------------------------------------------------------

    def _negamax(self, board, depth, alpha, beta, ply, null_allowed):
        self.nodes += 1
        if self.nodes & _TIME_CHECK_MASK == 0 and time.perf_counter() > self._deadline:
            raise _TimeUp

        # Draws by rule. Repetition needs >= 4 reversible half-moves.
        if board.halfmove_clock >= 100:
            return 0
        if board.halfmove_clock >= 4 and board.is_repetition(2):
            return 0
        # Repetition against the real game (positions before the search root,
        # invisible to the FEN-rooted board). Scored as draw at any depth.
        key = board._transposition_key()
        if self._game_history and key in self._game_history:
            return 0

        # Mate-distance pruning: a mate from here can never beat one already
        # found closer to the root; tighten the window accordingly.
        if alpha < ply - MATE:
            alpha = ply - MATE
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

        in_check = board.is_check()
        if in_check:
            depth += 1  # check extension: don't stand pat out of forcing lines

        if depth <= 0 or ply >= _MAX_PLY:
            return self._qsearch(board, alpha, beta, ply)

        tt_move = None
        entry = self.tt.get(key)
        if entry is not None:
            e_depth, e_score, e_flag, tt_move = entry
            if e_depth >= depth:
                score = self._score_from_tt(e_score, ply)
                if e_flag == _TT_EXACT:
                    return score
                if e_flag == _TT_LOWER and score >= beta:
                    return score
                if e_flag == _TT_UPPER and score <= alpha:
                    return score

        static_eval = None
        if not in_check and beta < MATE_BOUND:
            static_eval = self._static_eval(board, key)
            # Reverse futility pruning: if the static eval beats beta by a
            # depth-scaled margin, a full search almost never changes that.
            if depth <= 3 and static_eval - 120 * depth >= beta:
                return static_eval

        # Null-move pruning: if giving the opponent a free move still fails
        # high, this node is almost certainly >= beta. Skipped in check, at
        # low depth, and without sliders/knights (zugzwang danger).
        if (
            null_allowed
            and not in_check
            and depth >= 3
            and beta < MATE_BOUND
            and static_eval is not None
            and static_eval >= beta
            and board.occupied_co[board.turn]
            & (board.knights | board.bishops | board.rooks | board.queens)
        ):
            board.push(chess.Move.null())
            score = -self._negamax(board, depth - 1 - 2, -beta, -beta + 1, ply + 1, False)
            board.pop()
            if score >= beta:
                return beta

        best_score = -_INF
        best_move = None
        orig_alpha = alpha
        # Futility pruning: at frontier depths, quiet moves can't close a
        # large eval gap to alpha; only tactical moves get searched.
        futile = (
            not in_check
            and depth <= 2
            and static_eval is not None
            and static_eval + 150 * depth <= alpha
            and abs(alpha) < MATE_BOUND
        )

        any_legal = False
        searched = 0
        quiet_searched = 0
        for hint, move in self._ordered_moves(board, ply, tt_move):
            any_legal = True
            if hint >= _ORDER_TT:  # TT move: hint says nothing about type
                is_quiet = not board.is_capture(move) and move.promotion is None
            else:
                is_quiet = hint < _ORDER_PROMO
            if futile and is_quiet and best_move is not None and not board.gives_check(move):
                continue
            board.push(move)
            gives_check = board.is_check()
            if searched == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
            else:
                # Late move reductions: quiet moves far down the order get a
                # shallower look first, scaled by depth and quiet index.
                r = 0
                if is_quiet and depth >= 3 and not in_check and not gives_check \
                        and quiet_searched >= 2:
                    r = int(0.5 * ((depth - 1) ** 0.5 + quiet_searched ** 0.5))
                    if r > depth - 1:
                        r = depth - 1
                # PVS: scout with a null window; only a promising score earns
                # a deeper and then a full-window re-search.
                score = -self._negamax(board, depth - 1 - r, -alpha - 1, -alpha,
                                       ply + 1, True)
                if score > alpha and r:
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha,
                                           ply + 1, True)
                if alpha < score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -alpha,
                                           ply + 1, True)
            board.pop()
            searched += 1
            if is_quiet:
                quiet_searched += 1

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if is_quiet:
                    ks = self.killers[ply]
                    if ks[0] != move:
                        ks[1] = ks[0]
                        ks[0] = move
                    h_key = (board.turn, move.from_square, move.to_square)
                    self.history[h_key] = self.history.get(h_key, 0) + depth * depth
                break

        if not any_legal:
            return -(MATE - ply) if in_check else 0

        flag = (
            _TT_LOWER if best_score >= beta
            else _TT_UPPER if best_score <= orig_alpha
            else _TT_EXACT
        )
        self.tt[key] = (depth, self._score_to_tt(best_score, ply), flag, best_move)
        return best_score

    # ------------------------------------------------------------------
    # Quiescence
    # ------------------------------------------------------------------

    def _qsearch(self, board, alpha, beta, ply):
        self.nodes += 1
        if self.nodes & _TIME_CHECK_MASK == 0 and time.perf_counter() > self._deadline:
            raise _TimeUp

        in_check = board.is_check()
        if in_check:
            # Standing pat while in check is unsound; search every evasion.
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE - ply)
        else:
            stand_pat = self._static_eval(board, board._transposition_key())
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            if ply >= _MAX_PLY:
                return stand_pat
            moves = list(board.generate_legal_captures())
            moves.sort(key=lambda m: -self._mvv_lva(board, m))

        best_score = alpha if not in_check else -_INF
        for move in moves:
            if not in_check:
                # Delta pruning: even winning this victim outright can't
                # bring the score near alpha.
                victim = self._victim_value(board, move)
                if stand_pat + victim + 200 <= alpha and move.promotion is None:
                    continue
            board.push(move)
            score = -self._qsearch(board, -beta, -alpha, ply + 1)
            board.pop()
            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best_score

    def _static_eval(self, board, key) -> int:
        e = self.eval_cache.get(key)
        if e is None:
            e = self.evaluator.evaluate(board)
            self.eval_cache[key] = e
        return e

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def _ordered_moves(self, board, ply, tt_move):
        """Yield (order_hint, move) best-first. The TT move goes out before
        any move generation happens — a TT cutoff means the full legal-move
        enumeration (the most expensive part of a CPython node) never runs."""
        tt_yielded = False
        if tt_move is not None and board.is_legal(tt_move):
            tt_yielded = True
            yield _ORDER_TT, tt_move
        scored = [(self._order_score(board, m, ply), m)
                  for m in board.legal_moves
                  if not tt_yielded or m != tt_move]
        scored.sort(key=lambda t: t[0], reverse=True)
        yield from scored

    def _victim_value(self, board, move) -> int:
        if board.is_en_passant(move):
            return PIECE_VALUE[chess.PAWN]
        victim = board.piece_type_at(move.to_square)
        return PIECE_VALUE[victim] if victim else 0

    def _mvv_lva(self, board, move) -> int:
        attacker = board.piece_type_at(move.from_square) or 0
        return 10 * self._victim_value(board, move) - PIECE_VALUE[attacker] // 10

    def _order_score(self, board, move, ply, tt_move=None) -> int:
        if tt_move is not None and move == tt_move:
            return _ORDER_TT
        if board.is_capture(move):
            return _ORDER_CAPTURE + self._mvv_lva(board, move)
        if move.promotion == chess.QUEEN:
            return _ORDER_PROMO
        ks = self.killers[ply] if ply < _MAX_PLY else (None, None)
        if move == ks[0]:
            return _ORDER_KILLER1
        if move == ks[1]:
            return _ORDER_KILLER2
        # Quiets: history first; PST improvement breaks ties when history is
        # cold, so fresh nodes aren't ordered arbitrarily.
        pst = _MG_TABLE[board.piece_type_at(move.from_square)][board.turn]
        return (self.history.get((board.turn, move.from_square, move.to_square), 0) * 8
                + pst[move.to_square] - pst[move.from_square])

    # ------------------------------------------------------------------
    # Mate scores in the TT must be relative to the node, not the root.
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_tt(score: int, ply: int) -> int:
        if score >= MATE_BOUND:
            return score + ply
        if score <= -MATE_BOUND:
            return score - ply
        return score

    @staticmethod
    def _score_from_tt(score: int, ply: int) -> int:
        if score >= MATE_BOUND:
            return score - ply
        if score <= -MATE_BOUND:
            return score + ply
        return score

    def _extract_pv(self, board: chess.Board, first_move=None, limit: int = 12) -> list:
        """Walk the TT to recover the principal variation (for logging)."""
        pv = []
        b = board.copy(stack=False)
        # The root best move lives in the search result, not the TT; the
        # caller seeds the walk with it via first_move.
        if first_move is not None and first_move in b.legal_moves:
            pv.append(first_move)
            b.push(first_move)
        seen = set()
        while len(pv) < limit:
            key = b._transposition_key()
            if key in seen:
                break
            seen.add(key)
            entry = self.tt.get(key)
            if entry is None or entry[3] is None or entry[3] not in b.legal_moves:
                break
            pv.append(entry[3])
            b.push(entry[3])
        return pv

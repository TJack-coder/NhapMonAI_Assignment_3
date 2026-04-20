"""
Random rule-based chess agent (Python 3)
---------------------------------------
Requirements:
- Follow international chess rules using python-chess (`legal_moves`).
- Random + rule-based: choose randomly from all legal moves only.

Notes:
- "Rule-based" here means the agent always follows rules provided by python-chess.
- No tactical or positional evaluation is used.

Quick run:
    python random_rule_based_agent.py --games 10 --seed 42
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Optional

import chess


@dataclass
class MatchStats:
    wins: int = 0
    draws: int = 0
    losses: int = 0


class BaseAgent:
    """Simple interface for agents."""

    def choose_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError


class PureRandomAgent(BaseAgent):
    """Baseline agent: fully random over legal moves."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves are available in the current position.")
        return self.rng.choice(legal)


class RandomRuleBasedAgent(BaseAgent):
    """
    Pure random rule-based agent:
    - Gets legal moves from python-chess.
    - Picks one legal move at random.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves are available in the current position.")
        return self.rng.choice(legal_moves)


def play_one_game(
    white: BaseAgent,
    black: BaseAgent,
    max_plies: int = 300,
) -> str:
    """
    Returns PGN-style result: "1-0", "0-1", or "1/2-1/2".
    max_plies avoids extremely long games between weak agents.
    """
    board = chess.Board()

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        agent = white if board.turn == chess.WHITE else black
        move = agent.choose_move(board)

        if move not in board.legal_moves:
            # Safety guard: if an agent returns an illegal move, it loses immediately.
            return "0-1" if board.turn == chess.WHITE else "1-0"

        board.push(move)

    if not board.is_game_over(claim_draw=True):
        return "1/2-1/2"

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "1/2-1/2"
    return outcome.result()


def evaluate_against_random(
    games: int,
    seed: Optional[int] = None,
    max_plies: int = 300,
) -> MatchStats:
    """Evaluate RandomRuleBasedAgent against PureRandomAgent (both are legal-random)."""
    rng = random.Random(seed)
    stats = MatchStats()

    for i in range(games):
        # Alternate colors for fairness.
        if i % 2 == 0:
            white = RandomRuleBasedAgent(rng)
            black = PureRandomAgent(rng)
            result = play_one_game(white, black, max_plies=max_plies)
            if result == "1-0":
                stats.wins += 1
            elif result == "1/2-1/2":
                stats.draws += 1
            else:
                stats.losses += 1
        else:
            white = PureRandomAgent(rng)
            black = RandomRuleBasedAgent(rng)
            result = play_one_game(white, black, max_plies=max_plies)
            if result == "0-1":
                stats.wins += 1
            elif result == "1/2-1/2":
                stats.draws += 1
            else:
                stats.losses += 1

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure random rule-based chess agent benchmark"
    )
    parser.add_argument("--games", type=int, default=10, help="Number of evaluation games")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--max-plies",
        type=int,
        default=300,
        help="Maximum plies per game before forcing a draw",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.games <= 0:
        raise ValueError("--games must be > 0")
    if args.max_plies <= 0:
        raise ValueError("--max-plies must be > 0")

    stats = evaluate_against_random(
        games=args.games,
        seed=args.seed,
        max_plies=args.max_plies,
    )

    print("=== PURE RANDOM RULE-BASED AGENT REPORT ===")
    print(f"Games   : {args.games}")
    print(f"Wins    : {stats.wins}")
    print(f"Draws   : {stats.draws}")
    print(f"Losses  : {stats.losses}")

    print("Note: This is a pure random version with no strategic logic.")


if __name__ == "__main__":
    main()

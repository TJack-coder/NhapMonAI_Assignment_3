from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import chess

from random_rule_based_agent import RandomRuleBasedAgent

try:
    # Optional teacher/baseline from your search agent.
    from Chess_AI import evaluate_board as teacher_evaluate_board
except Exception:
    teacher_evaluate_board = None

MODEL_PATH = Path(__file__).with_name("chess_ml_model.json")

# Keep the same function name used by Chess_new.py for compatibility.
LEVEL_TO_DEPTH = {
    "poor": 1,
    "average": 2,
    "intermediate": 2,
    "good": 3,
}

# In this ML file, the displayed "depth" is only a strength tier.
LEVEL_TO_PROFILE = {
    "poor": {"top_k": 5, "epsilon": 0.30},
    "average": {"top_k": 3, "epsilon": 0.10},
    "intermediate": {"top_k": 3, "epsilon": 0.10},
    "good": {"top_k": 1, "epsilon": 0.00},
}

PIECE_WEIGHTS = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


class LinearChessModel:
    """Small trainable linear model in pure Python.

    The prediction is always from White's perspective:
    positive -> White is better
    negative -> Black is better
    """

    def __init__(self, feature_names: Sequence[str], weights: Optional[Dict[str, float]] = None, bias: float = 0.0):
        self.feature_names = list(feature_names)
        self.weights = {name: 0.0 for name in self.feature_names}
        if weights:
            self.weights.update(weights)
        self.bias = float(bias)

    def predict(self, feature_map: Dict[str, float]) -> float:
        total = self.bias
        for name in self.feature_names:
            total += self.weights.get(name, 0.0) * feature_map.get(name, 0.0)
        return total

    def train_sgd(
        self,
        dataset: Sequence[Tuple[Dict[str, float], float]],
        epochs: int = 10,
        learning_rate: float = 0.002,
        shuffle_seed: int = 42,
    ) -> None:
        if not dataset:
            return

        rng = random.Random(shuffle_seed)
        samples = list(dataset)

        for _ in range(epochs):
            rng.shuffle(samples)
            for feature_map, target in samples:
                pred = self.predict(feature_map)
                error = pred - target
                for name in self.feature_names:
                    self.weights[name] -= learning_rate * error * feature_map.get(name, 0.0)
                self.bias -= learning_rate * error

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LinearChessModel":
        return cls(
            feature_names=payload["feature_names"],
            weights=payload.get("weights", {}),
            bias=payload.get("bias", 0.0),
        )


FEATURE_NAMES = [
    "material_diff",
    "pawn_diff",
    "knight_diff",
    "bishop_diff",
    "rook_diff",
    "queen_diff",
    "mobility_diff",
    "center_control_diff",
    "attack_pressure_diff",
    "developed_minor_diff",
    "king_safety_diff",
    "white_in_check",
    "black_in_check",
    "side_to_move",
]


def get_depth_from_level(level_name: str = "average") -> int:
    return LEVEL_TO_DEPTH.get(level_name.lower(), LEVEL_TO_DEPTH["average"])


def extract_features(board: chess.Board) -> Dict[str, float]:
    white_counts = {}
    black_counts = {}
    for piece_type in PIECE_WEIGHTS:
        white_counts[piece_type] = len(board.pieces(piece_type, chess.WHITE))
        black_counts[piece_type] = len(board.pieces(piece_type, chess.BLACK))

    white_material = sum(PIECE_WEIGHTS[p] * white_counts[p] for p in PIECE_WEIGHTS)
    black_material = sum(PIECE_WEIGHTS[p] * black_counts[p] for p in PIECE_WEIGHTS)

    white_mobility = count_legal_moves_for_color(board, chess.WHITE)
    black_mobility = count_legal_moves_for_color(board, chess.BLACK)

    white_center = sum(len(board.attackers(chess.WHITE, sq)) for sq in CENTER_SQUARES)
    black_center = sum(len(board.attackers(chess.BLACK, sq)) for sq in CENTER_SQUARES)

    white_attack_pressure = total_attacks(board, chess.WHITE)
    black_attack_pressure = total_attacks(board, chess.BLACK)

    white_developed = developed_minor_pieces(board, chess.WHITE)
    black_developed = developed_minor_pieces(board, chess.BLACK)

    white_king_safety = king_safety(board, chess.WHITE)
    black_king_safety = king_safety(board, chess.BLACK)

    return {
        "material_diff": white_material - black_material,
        "pawn_diff": white_counts[chess.PAWN] - black_counts[chess.PAWN],
        "knight_diff": white_counts[chess.KNIGHT] - black_counts[chess.KNIGHT],
        "bishop_diff": white_counts[chess.BISHOP] - black_counts[chess.BISHOP],
        "rook_diff": white_counts[chess.ROOK] - black_counts[chess.ROOK],
        "queen_diff": white_counts[chess.QUEEN] - black_counts[chess.QUEEN],
        "mobility_diff": float(white_mobility - black_mobility),
        "center_control_diff": float(white_center - black_center),
        "attack_pressure_diff": float(white_attack_pressure - black_attack_pressure),
        "developed_minor_diff": float(white_developed - black_developed),
        "king_safety_diff": float(white_king_safety - black_king_safety),
        "white_in_check": 1.0 if is_color_in_check(board, chess.WHITE) else 0.0,
        "black_in_check": 1.0 if is_color_in_check(board, chess.BLACK) else 0.0,
        "side_to_move": 1.0 if board.turn == chess.WHITE else -1.0,
    }


def count_legal_moves_for_color(board: chess.Board, color: bool) -> int:
    temp = board.copy(stack=False)
    temp.turn = color
    return sum(1 for _ in temp.legal_moves)


def total_attacks(board: chess.Board, color: bool) -> int:
    return sum(len(board.attacks(square)) for square in board.piece_map() if board.color_at(square) == color)


def developed_minor_pieces(board: chess.Board, color: bool) -> int:
    if color == chess.WHITE:
        start_squares = {chess.B1, chess.G1, chess.C1, chess.F1}
    else:
        start_squares = {chess.B8, chess.G8, chess.C8, chess.F8}

    minor_squares = set(board.pieces(chess.KNIGHT, color)) | set(board.pieces(chess.BISHOP, color))
    return sum(1 for sq in minor_squares if sq not in start_squares)


def king_safety(board: chess.Board, color: bool) -> int:
    king_square = board.king(color)
    if king_square is None:
        return -999
    ring = chess.BB_KING_ATTACKS[king_square]
    unsafe = 0
    enemy = not color
    for sq in chess.SQUARES:
        if ring & chess.BB_SQUARES[sq] and board.is_attacked_by(enemy, sq):
            unsafe += 1
    return -unsafe


def is_color_in_check(board: chess.Board, color: bool) -> bool:
    temp = board.copy(stack=False)
    temp.turn = color
    return temp.is_check()


def bootstrap_model() -> LinearChessModel:
    # Sensible starting weights. Then optional training can refine them.
    weights = {
        "material_diff": 1.20,
        "pawn_diff": 0.25,
        "knight_diff": 0.60,
        "bishop_diff": 0.65,
        "rook_diff": 1.00,
        "queen_diff": 1.80,
        "mobility_diff": 0.05,
        "center_control_diff": 0.08,
        "attack_pressure_diff": 0.02,
        "developed_minor_diff": 0.15,
        "king_safety_diff": 0.20,
        "white_in_check": -0.70,
        "black_in_check": 0.70,
        "side_to_move": 0.03,
    }
    return LinearChessModel(FEATURE_NAMES, weights=weights, bias=0.0)


def save_model(model: LinearChessModel, path: Path = MODEL_PATH) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, ensure_ascii=False, indent=2)


def load_model(path: Path = MODEL_PATH) -> Optional[LinearChessModel]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return LinearChessModel.from_dict(payload)


def build_teacher_dataset(num_positions: int = 250, seed: int = 42) -> List[Tuple[Dict[str, float], float]]:
    rng = random.Random(seed)
    dataset: List[Tuple[Dict[str, float], float]] = []

    for _ in range(num_positions):
        board = chess.Board()
        plies = rng.randint(4, 24)
        for _ in range(plies):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))

        features = extract_features(board)

        if teacher_evaluate_board is not None:
            # Normalize search/static score into a smaller range for SGD.
            target = max(-20.0, min(20.0, teacher_evaluate_board(board) / 100.0))
        else:
            target = hand_label(board)

        dataset.append((features, float(target)))

    return dataset


def hand_label(board: chess.Board) -> float:
    f = extract_features(board)
    # fallback target if teacher is unavailable
    return (
        1.5 * f["material_diff"]
        + 0.08 * f["mobility_diff"]
        + 0.12 * f["center_control_diff"]
        + 0.18 * f["developed_minor_diff"]
        + 0.25 * f["king_safety_diff"]
        - 0.80 * f["white_in_check"]
        + 0.80 * f["black_in_check"]
    )


def ensure_model_ready(retrain: bool = False) -> LinearChessModel:
    if not retrain:
        loaded = load_model()
        if loaded is not None:
            return loaded

    model = bootstrap_model()
    dataset = build_teacher_dataset()
    model.train_sgd(dataset, epochs=14, learning_rate=0.0015)
    save_model(model)
    return model


def predict_board(board: chess.Board, model: Optional[LinearChessModel] = None) -> float:
    if model is None:
        model = ensure_model_ready(retrain=False)
    return model.predict(extract_features(board))


def evaluate_candidate_moves(board: chess.Board, model: Optional[LinearChessModel] = None) -> List[Tuple[chess.Move, float]]:
    if model is None:
        model = ensure_model_ready(retrain=False)

    scored_moves: List[Tuple[chess.Move, float]] = []
    for move in board.legal_moves:
        board.push(move)
        score = predict_board(board, model)
        board.pop()
        scored_moves.append((move, score))

    if board.turn == chess.WHITE:
        scored_moves.sort(key=lambda item: item[1], reverse=True)
    else:
        scored_moves.sort(key=lambda item: item[1])
    return scored_moves


def choose_move_with_profile(board: chess.Board, top_k: int, epsilon: float, model: Optional[LinearChessModel] = None) -> Optional[chess.Move]:
    scored = evaluate_candidate_moves(board, model)
    if not scored:
        return None

    if epsilon > 0.0 and random.random() < epsilon:
        k = min(top_k, len(scored))
        return random.choice([move for move, _ in scored[:k]])

    return scored[0][0]


def get_move_by_level(board: chess.Board, level_name: str = "average") -> Optional[chess.Move]:
    profile = LEVEL_TO_PROFILE.get(level_name.lower(), LEVEL_TO_PROFILE["average"])
    model = ensure_model_ready(retrain=False)
    return choose_move_with_profile(
        board,
        top_k=profile["top_k"],
        epsilon=profile["epsilon"],
        model=model,
    )


def train_model_file(num_positions: int = 400, epochs: int = 18, learning_rate: float = 0.0015, seed: int = 42) -> Path:
    model = bootstrap_model()
    dataset = build_teacher_dataset(num_positions=num_positions, seed=seed)
    model.train_sgd(dataset, epochs=epochs, learning_rate=learning_rate, shuffle_seed=seed)
    save_model(model)
    return MODEL_PATH


def play_game_vs_random(level_name: str = "average", ai_color: bool = chess.WHITE, seed: Optional[int] = None) -> str:
    board = chess.Board()
    rng = random.Random(seed)
    random_agent = RandomRuleBasedAgent(rng)

    while not board.is_game_over(claim_draw=True):
        if board.turn == ai_color:
            move = get_move_by_level(board, level_name)
        else:
            move = random_agent.choose_move(board)

        if move is None or move not in board.legal_moves:
            break
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "Draw"
    if outcome.winner == ai_color:
        return "Win"
    return "Loss"


def run_arena(level_name: str = "average", num_games: int = 10) -> Dict[str, int]:
    results = {"Win": 0, "Draw": 0, "Loss": 0}
    for i in range(num_games):
        ai_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        result = play_game_vs_random(level_name=level_name, ai_color=ai_color, seed=100 + i)
        results[result] += 1
    return results


if __name__ == "__main__":
    model_path = train_model_file()
    print(f"Saved model to: {model_path}")
    print("Arena average (10 games):", run_arena("average", 10))

import random
import time
import chess


LEVEL_TO_DEPTH = {
    "poor": 2,
    "average": 3,
    "intermediate": 3,
    "good": 5,
}


def get_depth_from_level(level_name: str = "average") -> int:
    """Map a named skill level to a search depth."""
    return LEVEL_TO_DEPTH.get(level_name.lower(), LEVEL_TO_DEPTH["average"])


def evaluate_board(board: chess.Board) -> int:
    """Evaluate the position from White's perspective.

    Positive score favors White, negative score favors Black.
    """
    if board.is_checkmate():
        return -9999 if board.turn == chess.WHITE else 9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    pawn_pst = [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0,
    ]

    knight_pst = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ]

    bishop_pst = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ]

    rook_pst = [
         0,  0,  0,  5,  5,  0,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         5, 10, 10, 10, 10, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ]

    queen_pst = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -10,  5,  5,  5,  5,  5,  0,-10,
          0,  0,  5,  5,  5,  5,  0, -5,
         -5,  0,  5,  5,  5,  5,  0, -5,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ]

    king_pst = [
         20, 30, 10,  0,  0, 10, 30, 20,
         20, 20,  0,  0,  0,  0, 20, 20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
    ]

    pst_dict = {
        chess.PAWN: pawn_pst,
        chess.KNIGHT: knight_pst,
        chess.BISHOP: bishop_pst,
        chess.ROOK: rook_pst,
        chess.QUEEN: queen_pst,
        chess.KING: king_pst,
    }

    evaluation = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        value = piece_values[piece.piece_type]
        pst = pst_dict[piece.piece_type]

        if piece.color == chess.WHITE:
            pst_value = pst[square]
            evaluation += value + pst_value
        else:
            pst_value = pst[chess.square_mirror(square)]
            evaluation -= value + pst_value

    return evaluation


def order_moves(board: chess.Board, moves):
    """Order moves using promotion, capture priority and checks."""
    piece_values = {
        chess.PAWN: 10,
        chess.KNIGHT: 30,
        chess.BISHOP: 30,
        chess.ROOK: 50,
        chess.QUEEN: 90,
        chess.KING: 900,
    }

    def move_score(move: chess.Move) -> int:
        score = 0

        if move.promotion:
            score += piece_values.get(move.promotion, 0) * 10

        if board.is_capture(move):
            target_piece = board.piece_at(move.to_square)
            attacker_piece = board.piece_at(move.from_square)

            if target_piece and attacker_piece:
                score += (
                    piece_values.get(target_piece.piece_type, 0) * 10
                    - piece_values.get(attacker_piece.piece_type, 0)
                )
            elif board.is_en_passant(move):
                score += (piece_values[chess.PAWN] * 10) - piece_values[chess.PAWN]

        if board.gives_check(move):
            score += 50

        return score

    return sorted(moves, key=move_score, reverse=True)


def minimax(board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing_player: bool) -> int:
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if is_maximizing_player:
        max_eval = -float("inf")
        for move in order_moves(board, list(board.legal_moves)):
            board.push(move)
            evaluation = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval

    min_eval = float("inf")
    for move in order_moves(board, list(board.legal_moves)):
        board.push(move)
        evaluation = minimax(board, depth - 1, alpha, beta, True)
        board.pop()

        min_eval = min(min_eval, evaluation)
        beta = min(beta, evaluation)
        if beta <= alpha:
            break
    return min_eval


def get_best_move(board: chess.Board, depth: int):
    """Return the best move for the side to move.

    The evaluation is always from White's perspective, so White maximizes and Black minimizes.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    alpha = -float("inf")
    beta = float("inf")
    best_move = None

    if board.turn == chess.WHITE:
        best_eval = -float("inf")
        for move in order_moves(board, legal_moves):
            board.push(move)
            evaluation = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            if evaluation > best_eval:
                best_eval = evaluation
                best_move = move
            alpha = max(alpha, evaluation)
    else:
        best_eval = float("inf")
        for move in order_moves(board, legal_moves):
            board.push(move)
            evaluation = minimax(board, depth - 1, alpha, beta, True)
            board.pop()

            if evaluation < best_eval:
                best_eval = evaluation
                best_move = move
            beta = min(beta, evaluation)

    return best_move


def get_move_by_level(board: chess.Board, level_name: str = "average"):
    """Return a move based on a named skill level."""
    depth = get_depth_from_level(level_name)
    return get_best_move(board, depth=depth)


def test_ai_level(board: chess.Board, level_name: str):
    ai_depth = get_depth_from_level(level_name)

    print(f"=== ĐANG TEST AI Ở MỨC ĐỘ: {level_name.upper()} (Depth = {ai_depth}) ===")
    print("Trạng thái bàn cờ ban đầu:")
    print(board)
    print("-" * 20)

    start_time = time.perf_counter()
    best_move = get_best_move(board, depth=ai_depth)
    end_time = time.perf_counter()

    execution_time = end_time - start_time

    print(f"-> AI quyết định đi nước: {best_move}")
    print(f"-> Thời gian suy nghĩ: {execution_time:.4f} giây")

    if best_move:
        board.push(best_move)
        print("\nTrạng thái bàn cờ sau khi AI đi:")
        print(board)
    else:
        print("Không tìm được nước đi (có thể đã hết cờ hoặc hòa).")

    print("=" * 50 + "\n")


def play_game_vs_random(ai_depth: int, ai_color: bool = chess.WHITE):
    """Play one full game between the searching AI and a random opponent."""
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == ai_color:
            move = get_best_move(board, depth=ai_depth)
            if move is None:
                break
            board.push(move)
        else:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)

    outcome = board.outcome()
    if outcome is None:
        return "Lỗi/Chưa kết thúc"

    if outcome.winner == ai_color:
        return "Thắng"
    if outcome.winner is None:
        return "Hòa"
    return "Thua"


def run_arena(ai_depth: int, num_games: int = 10):
    print(f"🔥 BẮT ĐẦU ĐẤU {num_games} VÁN VỚI RANDOM AGENT (AI Depth = {ai_depth}) 🔥")
    print("-" * 50)

    wins = 0
    losses = 0
    draws = 0

    start_total_time = time.perf_counter()

    for i in range(num_games):
        print(f"Đang chơi ván {i + 1}...", end=" ")
        ai_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        result = play_game_vs_random(ai_depth, ai_color=ai_color)

        if result == "Thắng":
            wins += 1
            print("✅ AI Thắng!")
        elif result == "Thua":
            losses += 1
            print("❌ AI Thua!")
        else:
            draws += 1
            print("🤝 Hòa!")

    end_total_time = time.perf_counter()

    print("-" * 50)
    print("📊 KẾT QUẢ CHUNG CUỘC:")
    print(f"Tổng số ván: {num_games}")
    print(f"Thắng: {wins} | Thua: {losses} | Hòa: {draws}")
    print(f"Tỉ lệ thắng: {(wins / num_games) * 100}%")
    print(f"Tổng thời gian chạy mô phỏng: {end_total_time - start_total_time:.2f} giây")


if __name__ == "__main__":
    test_board = chess.Board()
    test_ai_level(test_board, "poor")
    run_arena(ai_depth=get_depth_from_level("average"), num_games=10)

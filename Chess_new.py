import pygame
import chess
import sys
import random
import time

from random_rule_based_agent import RandomRuleBasedAgent
from Chess_AI import (
    get_move_by_level as get_search_move_by_level,
    get_depth_from_level as get_search_depth_from_level,
)
from Chess_ML_Method import (
    get_move_by_level as get_ml_move_by_level,
    get_depth_from_level as get_ml_depth_from_level,
)

pygame.init()

MARGIN = 30
BOARD_SIZE = 512
PANEL_WIDTH = 270
PANEL_X = BOARD_SIZE + 2 * MARGIN

WIDTH = PANEL_X + PANEL_WIDTH
HEIGHT = BOARD_SIZE + 2 * MARGIN
DIMENSION = 8
SQ_SIZE = BOARD_SIZE // DIMENSION
MAX_FPS = 15

COLOR_LIGHT = (240, 217, 181)
COLOR_DARK = (181, 136, 99)
COLOR_HIGHLIGHT = (186, 202, 68)

COLOR_MOVE_EMPTY = (255, 255, 0, 100)
COLOR_MOVE_CAPTURE = (255, 0, 0, 120)

COLOR_PANEL = (33, 37, 45)
COLOR_CARD = (48, 54, 66)
COLOR_CARD_BORDER = (74, 84, 102)
COLOR_BTN = (77, 86, 106)
COLOR_BTN_HOVER = (97, 108, 130)
COLOR_BTN_ACTIVE = (122, 145, 80)
COLOR_BTN_DISABLED = (58, 62, 74)
COLOR_BTN_BORDER = (146, 156, 177)
COLOR_TEXT = (255, 255, 255)
COLOR_MUTED = (210, 216, 230)
COLOR_WARNING = (255, 95, 95)
COLOR_RESULT_BG = (24, 28, 36, 220)
COLOR_RESULT_BORDER = (205, 175, 90)
COLOR_RESULT_TEXT = (255, 230, 140)
COLOR_SHADOW = (15, 18, 24)

PIECE_UNICODE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"
}

BTN_WIDTH, BTN_HEIGHT = 200, 38
BTN_X = PANEL_X + (PANEL_WIDTH - BTN_WIDTH) // 2

btn_undo = pygame.Rect(BTN_X, 34, BTN_WIDTH, BTN_HEIGHT)
btn_reset = pygame.Rect(BTN_X, 78, BTN_WIDTH, BTN_HEIGHT)
btn_mode_hvh = pygame.Rect(BTN_X, 128, BTN_WIDTH, BTN_HEIGHT)
btn_mode_rvm = pygame.Rect(BTN_X, 172, BTN_WIDTH, BTN_HEIGHT)
btn_mode_svm = pygame.Rect(BTN_X, 216, BTN_WIDTH, BTN_HEIGHT)
btn_benchmark = pygame.Rect(BTN_X, 260, BTN_WIDTH, BTN_HEIGHT)
btn_toggle_auto = pygame.Rect(BTN_X, 304, BTN_WIDTH, BTN_HEIGHT)
btn_cycle_level = pygame.Rect(BTN_X, 348, BTN_WIDTH, BTN_HEIGHT)

status_box = pygame.Rect(PANEL_X + 18, 392, PANEL_WIDTH - 36, 160)

POPUP_WIDTH = 320
POPUP_HEIGHT = 180
popup_rect = pygame.Rect(
    (WIDTH - POPUP_WIDTH) // 2,
    (HEIGHT - POPUP_HEIGHT) // 2,
    POPUP_WIDTH,
    POPUP_HEIGHT,
)

btn_popup_play_again = pygame.Rect(
    popup_rect.x + 35,
    popup_rect.y + 105,
    115,
    42,
)

btn_popup_close = pygame.Rect(
    popup_rect.x + 170,
    popup_rect.y + 105,
    115,
    42,
)

GAME_MODE_HVH = "hvh"
GAME_MODE_RVM = "rvm"
GAME_MODE_SVM = "svm"
AGENT_MOVE_DELAY_MS = 280
MAX_PLIES_PER_GAME = 220
RANDOM_ML_BENCHMARK_GAMES = 10
SEARCH_ML_BENCHMARK_GAMES = 4

ML_LEVELS = ["poor", "average", "good"]
ml_level_index = 1
benchmark_summary_lines = []


def format_level_name(level_name: str) -> str:
    return level_name.capitalize()


def get_current_ml_level() -> str:
    return ML_LEVELS[ml_level_index]


def cycle_ml_level() -> str:
    global ml_level_index
    ml_level_index = (ml_level_index + 1) % len(ML_LEVELS)
    return get_current_ml_level()


def clear_benchmark_summary():
    global benchmark_summary_lines
    benchmark_summary_lines = []


def set_benchmark_summary(lines):
    global benchmark_summary_lines
    benchmark_summary_lines = list(lines)


def draw_board(screen, coord_font):
    pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(0, 0, PANEL_X, HEIGHT))

    colors = [COLOR_LIGHT, COLOR_DARK]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            rect = pygame.Rect(MARGIN + c * SQ_SIZE, MARGIN + r * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            pygame.draw.rect(screen, color, rect)

    for r in range(DIMENSION):
        text = coord_font.render(str(8 - r), True, COLOR_TEXT)
        text_rect = text.get_rect(center=(MARGIN // 2, MARGIN + r * SQ_SIZE + SQ_SIZE // 2))
        screen.blit(text, text_rect)

    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for c in range(DIMENSION):
        text = coord_font.render(files[c], True, COLOR_TEXT)
        text_rect = text.get_rect(center=(MARGIN + c * SQ_SIZE + SQ_SIZE // 2, HEIGHT - MARGIN // 2))
        screen.blit(text, text_rect)


def draw_pieces(screen, board, piece_font):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            square_index = chess.square(c, 7 - r)
            piece = board.piece_at(square_index)
            if piece:
                text_surface = piece_font.render(PIECE_UNICODE[piece.symbol()], True, (0, 0, 0))
                text_rect = text_surface.get_rect(
                    center=(MARGIN + c * SQ_SIZE + SQ_SIZE // 2, MARGIN + r * SQ_SIZE + SQ_SIZE // 2)
                )
                screen.blit(text_surface, text_rect)


def draw_valid_moves(screen, board, selected_square):
    if selected_square is None:
        return

    surface = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
    for move in board.legal_moves:
        if move.from_square == selected_square:
            to_square = move.to_square
            c = chess.square_file(to_square)
            r = 7 - chess.square_rank(to_square)
            rect_x = MARGIN + c * SQ_SIZE
            rect_y = MARGIN + r * SQ_SIZE

            is_capture = board.piece_at(to_square) is not None or board.is_en_passant(move)
            surface.fill(COLOR_MOVE_CAPTURE if is_capture else COLOR_MOVE_EMPTY)
            screen.blit(surface, (rect_x, rect_y))


def draw_button(screen, ui_font, rect, text, active=False, disabled=False):
    mouse_pos = pygame.mouse.get_pos()
    hovered = rect.collidepoint(mouse_pos)

    if disabled:
        fill = COLOR_BTN_DISABLED
        border = COLOR_CARD_BORDER
    elif active:
        fill = COLOR_BTN_ACTIVE
        border = (185, 205, 120)
    elif hovered:
        fill = COLOR_BTN_HOVER
        border = (170, 180, 200)
    else:
        fill = COLOR_BTN
        border = COLOR_BTN_BORDER

    shadow_rect = rect.move(0, 4)
    pygame.draw.rect(screen, COLOR_SHADOW, shadow_rect, border_radius=12)
    pygame.draw.rect(screen, fill, rect, border_radius=12)
    pygame.draw.rect(screen, border, rect, width=2, border_radius=12)

    text_surface = ui_font.render(text, True, COLOR_TEXT)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)


def draw_status_box(screen, small_font, board, game_message, game_mode, auto_play):
    pygame.draw.rect(screen, COLOR_CARD, status_box, border_radius=14)
    pygame.draw.rect(screen, COLOR_CARD_BORDER, status_box, width=2, border_radius=14)

    mode_label = {
        GAME_MODE_HVH: "Mode: Human vs Human ",
        GAME_MODE_RVM: "Mode: Random vs ML ",
        GAME_MODE_SVM: "Mode: Search vs ML ",
    }[game_mode]

    lines = [
        (mode_label, COLOR_TEXT),
        (f"Auto: {'ON' if auto_play else 'OFF'}", COLOR_MUTED),
        (f"Turn: {'White' if board.turn else 'Black'}", COLOR_MUTED),
    ]

    if game_mode == GAME_MODE_HVH:
        lines.append(("White: Human", COLOR_MUTED))
        lines.append(("Black: Human", COLOR_MUTED))
    elif game_mode == GAME_MODE_RVM:
        ml_level = get_current_ml_level()
        lines.append(("White: Random", COLOR_MUTED))
        lines.append((f"Black: ML ({format_level_name(ml_level)})", COLOR_MUTED))
        lines.append((f"ML tier: {get_ml_depth_from_level(ml_level)}", COLOR_MUTED))
    elif game_mode == GAME_MODE_SVM:
        if benchmark_summary_lines:
            for item in benchmark_summary_lines[:3]:
                lines.append((item, COLOR_RESULT_TEXT if item.startswith("Win-Draw-Lose") else COLOR_MUTED))
        else:
            duel_level = get_current_ml_level()
            lines.append((f"White: Search", COLOR_MUTED))
            lines.append((f"Black: ML ({format_level_name(duel_level)})", COLOR_MUTED))
            lines.append((f"Search depth: {get_search_depth_from_level(duel_level)} | ML tier: {get_ml_depth_from_level(duel_level)}", COLOR_MUTED))

    if board.is_game_over():
        lines.append((game_message, COLOR_RESULT_TEXT))
    elif board.is_check():
        lines.append(("CHECK!!!", COLOR_WARNING))
    elif game_message:
        lines.append((game_message, COLOR_HIGHLIGHT))


    y = status_box.y + 10
    for text, color in lines[:8]:
        surface = small_font.render(text, True, color)
        screen.blit(surface, (status_box.x + 12, y))
        y += 18


def draw_panel(screen, ui_font, small_font, board, game_message, game_mode, auto_play):
    pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(PANEL_X, 0, PANEL_WIDTH, HEIGHT))

    draw_button(screen, ui_font, btn_undo, "Undo")
    draw_button(screen, ui_font, btn_reset, "Reset")
    draw_button(screen, ui_font, btn_mode_hvh, "Human vs Human", active=(game_mode == GAME_MODE_HVH))
    draw_button(screen, ui_font, btn_mode_rvm, "Random vs ML", active=(game_mode == GAME_MODE_RVM))
    draw_button(screen, ui_font, btn_mode_svm, "Search vs ML", active=(game_mode == GAME_MODE_SVM))

    benchmark_label = "Run Benchmark"
    draw_button(screen, ui_font, btn_benchmark, benchmark_label, disabled=(game_mode != GAME_MODE_SVM))

    auto_btn_text = "Start Auto" if not auto_play else "Stop Auto"
    draw_button(screen, ui_font, btn_toggle_auto, auto_btn_text, disabled=(game_mode == GAME_MODE_HVH))

    if game_mode != GAME_MODE_HVH:
        draw_button(screen, ui_font, btn_cycle_level, f"Level: {format_level_name(get_current_ml_level())}")

    draw_status_box(screen, small_font, board, game_message, game_mode, auto_play)


def draw_result_banner(screen, banner_font, message):
    if not message:
        return

    banner_width = 280
    banner_height = 56
    banner_x = MARGIN + (BOARD_SIZE - banner_width) // 2
    banner_y = MARGIN + 14

    banner = pygame.Surface((banner_width, banner_height), pygame.SRCALPHA)
    pygame.draw.rect(banner, COLOR_RESULT_BG, pygame.Rect(0, 0, banner_width, banner_height), border_radius=14)
    pygame.draw.rect(banner, COLOR_RESULT_BORDER, pygame.Rect(0, 0, banner_width, banner_height), width=2, border_radius=14)

    text_surface = banner_font.render(message, True, COLOR_RESULT_TEXT)
    text_rect = text_surface.get_rect(center=(banner_width // 2, banner_height // 2))
    banner.blit(text_surface, text_rect)
    screen.blit(banner, (banner_x, banner_y))


def draw_game_over_popup(screen, title_font, text_font, message):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 120))
    screen.blit(overlay, (0, 0))

    pygame.draw.rect(screen, COLOR_CARD, popup_rect, border_radius=16)
    pygame.draw.rect(screen, COLOR_RESULT_BORDER, popup_rect, width=2, border_radius=16)

    title_surface = title_font.render("Game Over", True, COLOR_RESULT_TEXT)
    title_rect = title_surface.get_rect(center=(popup_rect.centerx, popup_rect.y + 36))
    screen.blit(title_surface, title_rect)

    msg_surface = text_font.render(message, True, COLOR_TEXT)
    msg_rect = msg_surface.get_rect(center=(popup_rect.centerx, popup_rect.y + 72))
    screen.blit(msg_surface, msg_rect)

    draw_button(screen, text_font, btn_popup_play_again, "Play Again", active=True)
    draw_button(screen, text_font, btn_popup_close, "Close")


def get_game_over_message(board):
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"{winner} win"
    if board.is_stalemate():
        return "Draw by stalemate"
    if board.is_insufficient_material():
        return "Draw: Insufficient material"
    if board.can_claim_threefold_repetition():
        return "Draw: Threefold repetition"
    if board.can_claim_fifty_moves():
        return "Draw: Fifty moves"
    return "Game over"


def get_ml_move(board, level_name=None):
    level_name = level_name or get_current_ml_level()
    move = get_ml_move_by_level(board, level_name)
    if move is None:
        legal_moves = list(board.legal_moves)
        return legal_moves[0] if legal_moves else None
    return move


def get_search_move(board, level_name=None):
    level_name = level_name or get_current_ml_level()
    move = get_search_move_by_level(board, level_name)
    if move is None:
        legal_moves = list(board.legal_moves)
        return legal_moves[0] if legal_moves else None
    return move


def choose_agent_move(agent_kind: str, board: chess.Board, rng: random.Random, level_name: str):
    if agent_kind == "random":
        agent = RandomRuleBasedAgent(rng)
        return agent.choose_move(board)
    if agent_kind == "search":
        return get_search_move(board, level_name)
    if agent_kind == "ml":
        return get_ml_move(board, level_name)
    return None


def play_benchmark_game(white_kind: str, black_kind: str, level_name: str, seed: int):
    board = chess.Board()
    rng = random.Random(seed)
    side_times = {"white": 0.0, "black": 0.0}
    side_moves = {"white": 0, "black": 0}

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_PLIES_PER_GAME:
        mover_kind = white_kind if board.turn == chess.WHITE else black_kind
        side_key = "white" if board.turn == chess.WHITE else "black"

        start = time.perf_counter()
        move = choose_agent_move(mover_kind, board, rng, level_name)
        elapsed = time.perf_counter() - start

        side_times[side_key] += elapsed
        side_moves[side_key] += 1

        if move is None or move not in board.legal_moves:
            return {
                "result": "1-0" if board.turn == chess.BLACK else "0-1",
                "side_times": side_times,
                "side_moves": side_moves,
            }

        board.push(move)

    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome is not None else "1/2-1/2"
    return {"result": result, "side_times": side_times, "side_moves": side_moves}


def run_benchmark(game_mode: str, level_name: str):
    if game_mode == GAME_MODE_RVM:
        total_games = RANDOM_ML_BENCHMARK_GAMES
        pairings = [
            ("random", "ml") if i % 2 == 0 else ("ml", "random")
            for i in range(total_games)
        ]
        opponent_name = "Random"
    elif game_mode == GAME_MODE_SVM:
        total_games = SEARCH_ML_BENCHMARK_GAMES
        pairings = [
            ("search", "ml") if i % 2 == 0 else ("ml", "search")
            for i in range(total_games)
        ]
        opponent_name = "Search"
    else:
        return []

    ml_wins = 0
    ml_draws = 0
    ml_losses = 0
    ml_total_time = 0.0
    ml_total_moves = 0
    opponent_total_time = 0.0
    opponent_total_moves = 0

    for i, (white_kind, black_kind) in enumerate(pairings):
        result_data = play_benchmark_game(white_kind, black_kind, level_name, seed=500 + i)
        result = result_data["result"]

        ml_is_white = white_kind == "ml"
        if result == "1/2-1/2":
            ml_draws += 1
        elif (result == "1-0" and ml_is_white) or (result == "0-1" and not ml_is_white):
            ml_wins += 1
        else:
            ml_losses += 1

        if white_kind == "ml":
            ml_total_time += result_data["side_times"]["white"]
            ml_total_moves += result_data["side_moves"]["white"]
            opponent_total_time += result_data["side_times"]["black"]
            opponent_total_moves += result_data["side_moves"]["black"]
        else:
            ml_total_time += result_data["side_times"]["black"]
            ml_total_moves += result_data["side_moves"]["black"]
            opponent_total_time += result_data["side_times"]["white"]
            opponent_total_moves += result_data["side_moves"]["white"]

    ml_avg = ml_total_time / ml_total_moves if ml_total_moves else 0.0
    opp_avg = opponent_total_time / opponent_total_moves if opponent_total_moves else 0.0

    return [
        "White: Search | Black: ML",
        f"Win-Draw-Lose (ML): {ml_wins}-{ml_draws}-{ml_losses}",
        f"Time: Search {opp_avg:.3f}s | ML {ml_avg:.3f}s",
    ]


def get_square_under_mouse():
    mouse_pos = pygame.mouse.get_pos()
    x = mouse_pos[0] - MARGIN
    y = mouse_pos[1] - MARGIN

    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
        return chess.square(x // SQ_SIZE, 7 - (y // SQ_SIZE))
    return None


def reset_game_state(board):
    board.reset()


def set_mode(board, mode_name: str):
    reset_game_state(board)
    clear_benchmark_summary()
    return {
        GAME_MODE_HVH: "Mode: Human vs Human ",
        GAME_MODE_RVM: "Mode: Random vs ML ",
        GAME_MODE_SVM: "Mode: Search vs ML ",
    }[mode_name]


def get_auto_agent_move(board, game_mode):
    if game_mode == GAME_MODE_RVM:
        if board.turn == chess.WHITE:
            return choose_agent_move("random", board, random.Random(), get_current_ml_level())
        return get_ml_move(board)

    if game_mode == GAME_MODE_SVM:
        if board.turn == chess.WHITE:
            return get_search_move(board)
        return get_ml_move(board)

    return None


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Game")
    try:
        app_icon = pygame.image.load("icon.png")
        pygame.display.set_icon(app_icon)
    except Exception:
        print("icon.png not found, using default icon.")

    clock = pygame.time.Clock()

    pygame.font.init()
    try:
        piece_font = pygame.font.SysFont("segoeuisymbol", SQ_SIZE - 10)
    except Exception:
        piece_font = pygame.font.Font(None, SQ_SIZE - 10)

    ui_font = pygame.font.SysFont("arial", 18, bold=True)
    small_font = pygame.font.SysFont("arial", 15, bold=True)
    banner_font = pygame.font.SysFont("arial", 24, bold=True)
    popup_title_font = pygame.font.SysFont("arial", 26, bold=True)
    coord_font = pygame.font.SysFont("arial", 14, bold=True)

    board = chess.Board()

    running = True
    selected_square = None
    game_message = ""
    game_mode = GAME_MODE_HVH
    auto_play = False
    last_agent_move_ms = 0
    show_game_over_popup = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()

                if show_game_over_popup:
                    if btn_popup_play_again.collidepoint(mouse_pos):
                        board.reset()
                        game_message = ""
                        selected_square = None
                        auto_play = False
                        show_game_over_popup = False
                    elif btn_popup_close.collidepoint(mouse_pos):
                        show_game_over_popup = False
                    continue

                if btn_undo.collidepoint(mouse_pos):
                    if len(board.move_stack) > 0:
                        board.pop()
                        game_message = ""
                        selected_square = None
                        show_game_over_popup = False

                elif btn_reset.collidepoint(mouse_pos):
                    board.reset()
                    game_message = ""
                    selected_square = None
                    auto_play = False
                    show_game_over_popup = False
                    clear_benchmark_summary()

                elif btn_mode_hvh.collidepoint(mouse_pos):
                    game_mode = GAME_MODE_HVH
                    auto_play = False
                    game_message = set_mode(board, game_mode)
                    selected_square = None
                    show_game_over_popup = False

                elif btn_mode_rvm.collidepoint(mouse_pos):
                    game_mode = GAME_MODE_RVM
                    auto_play = False
                    game_message = set_mode(board, game_mode)
                    selected_square = None
                    show_game_over_popup = False

                elif btn_mode_svm.collidepoint(mouse_pos):
                    game_mode = GAME_MODE_SVM
                    auto_play = False
                    game_message = set_mode(board, game_mode)
                    selected_square = None
                    show_game_over_popup = False

                elif btn_benchmark.collidepoint(mouse_pos) and game_mode == GAME_MODE_SVM:
                    auto_play = False
                    selected_square = None
                    level_name = get_current_ml_level()
                    set_benchmark_summary(run_benchmark(game_mode, level_name))
                    game_message = "Search vs ML benchmark"

                elif btn_toggle_auto.collidepoint(mouse_pos):
                    if game_mode != GAME_MODE_HVH:
                        auto_play = not auto_play
                        last_agent_move_ms = pygame.time.get_ticks()
                        game_message = "Auto ON" if auto_play else "Auto OFF"
                        selected_square = None

                elif btn_cycle_level.collidepoint(mouse_pos) and game_mode != GAME_MODE_HVH:
                    new_level = cycle_ml_level()
                    clear_benchmark_summary()
                    game_message = f"Level -> {format_level_name(new_level)}"
                    selected_square = None

                elif mouse_pos[0] < PANEL_X and game_mode == GAME_MODE_HVH and not board.is_game_over():
                    clicked_square = get_square_under_mouse()

                    if clicked_square is not None:
                        if selected_square is None:
                            piece = board.piece_at(clicked_square)
                            if piece and piece.color == board.turn:
                                selected_square = clicked_square
                        else:
                            move = chess.Move(selected_square, clicked_square)

                            piece = board.piece_at(selected_square)
                            if piece and piece.piece_type == chess.PAWN:
                                if chess.square_rank(clicked_square) in (0, 7):
                                    move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)

                            if move in board.legal_moves:
                                board.push(move)
                                selected_square = None
                                if board.is_game_over():
                                    game_message = get_game_over_message(board)
                                    show_game_over_popup = True
                            else:
                                piece = board.piece_at(clicked_square)
                                if piece and piece.color == board.turn:
                                    selected_square = clicked_square
                                else:
                                    selected_square = None

        if auto_play and not board.is_game_over():
            now = pygame.time.get_ticks()
            if now - last_agent_move_ms >= AGENT_MOVE_DELAY_MS:
                agent_move = get_auto_agent_move(board, game_mode)

                if agent_move and agent_move in board.legal_moves:
                    board.push(agent_move)
                    selected_square = None
                    if board.is_game_over():
                        game_message = get_game_over_message(board)
                        auto_play = False
                        show_game_over_popup = True

                last_agent_move_ms = now

        draw_board(screen, coord_font)

        if selected_square is not None:
            c = chess.square_file(selected_square)
            r = 7 - chess.square_rank(selected_square)
            pygame.draw.rect(
                screen,
                COLOR_HIGHLIGHT,
                pygame.Rect(MARGIN + c * SQ_SIZE, MARGIN + r * SQ_SIZE, SQ_SIZE, SQ_SIZE),
            )

        draw_valid_moves(screen, board, selected_square)
        draw_pieces(screen, board, piece_font)
        draw_panel(screen, ui_font, small_font, board, game_message, game_mode, auto_play)

        if board.is_game_over() and game_message:
            draw_result_banner(screen, banner_font, game_message)

        if show_game_over_popup:
            draw_game_over_popup(screen, popup_title_font, small_font, game_message)

        pygame.display.flip()
        clock.tick(MAX_FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

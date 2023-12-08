import numpy as np

from nanoAlphaGo.config import WHITE, BLACK, EMPTY, SEKI, BOARD_SIZE, KOMI


NEIGHBORS = {(x, y): [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
             for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)}


def calculate_outcome_for_player(board, color):
    score = calculate_score(board._matrix)
    player_won = score[color] > score[-color]
    draw = score[color] == score[-color]
    if player_won:
        return 1
    if draw:
        return 0
    else:
        return -1


def calculate_score(board_matrix, komi=KOMI):
    working_board = board_matrix.copy()
    working_board = assign_positions_to_players(working_board)
    black_score, white_score = count_player_positions(working_board)
    return {BLACK: black_score,
            WHITE: white_score + komi}


def assign_positions_to_players(working_board):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if working_board[row, col] == EMPTY:
                set_placeholder_color(working_board, row, col)
    return working_board


def count_player_positions(working_board):
    black_score = np.count_nonzero(working_board == BLACK)
    white_score = np.count_nonzero(working_board == WHITE)
    return black_score, white_score


def set_placeholder_color(working_board, row, col):
    territory, borders = find_reached(working_board, (row, col))
    border_colors = set(working_board[b] for b in borders)
    territory_color = find_placeholder_color(border_colors)

    for position in territory:
        working_board[position] = territory_color


def find_placeholder_color(border_colors):
    if BLACK in border_colors and WHITE not in border_colors:
        territory_color = BLACK
    elif WHITE in border_colors and BLACK not in border_colors:
        territory_color = WHITE
    else:
        territory_color = SEKI
    return territory_color


def find_reached(board_matrix, position):
    color = board_matrix[position]
    chain = set([position])
    reached = set()
    frontier = [position]
    while frontier:
        current = frontier.pop()
        explore_current_neighbours(current,
                                   board_matrix,
                                   color,
                                   chain,
                                   reached,
                                   frontier)
    return chain, reached


def explore_current_neighbours(current,
                               board_matrix,
                               color,
                               chain,
                               reached,
                               frontier):
    chain.add(current)
    for n in NEIGHBORS[current]:
        if position_is_within_board(n):
            if board_matrix[n] == color and not n in chain:
                frontier.append(n)
            elif board_matrix[n] != color:
                reached.add(n)


def position_is_within_board(pos):
    x_on_board = 0 <= pos[0] < BOARD_SIZE
    y_on_board = 0 <= pos[1] < BOARD_SIZE
    pos_is_on_the_board = x_on_board and y_on_board
    return pos_is_on_the_board



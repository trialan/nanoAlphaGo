from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import BLACK, WHITE, BOARD_SIZE
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.graphics.rendering import display_board


def play_turn(policy, board, history):
    move = policy.generate_move(board)
    board.apply_move(move, policy.color)
    history.append(move)
    display_board(board)


if __name__ == '__main__':
    board = GoBoard(BOARD_SIZE)
    black_policy = PolicyNN(color=BLACK)
    white_policy = PolicyNN(color=WHITE)

    game_history = []

    for _ in range(20):
        play_turn(white_policy, board, game_history)
        play_turn(black_policy, board, game_history)
        print(game_history)



from nanoAlphaGo.board import GoBoard
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rendering import display_board

BLACK = 1
WHITE = -1


def play_turn(policy, board, history):
    move = policy.generate_move(board)
    board.apply_move(move, policy.color)
    history.append(move)
    display_board(board)


if __name__ == '__main__':
    board = GoBoard(size=19)
    black_policy = PolicyNN(board, color=BLACK)
    white_policy = PolicyNN(board, color=WHITE)

    game_history = []

    for _ in range(2):
        play_turn(white_policy, board, game_history)
        play_turn(black_policy, board, game_history)
        print(game_history)



from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import BLACK, WHITE, BOARD_SIZE
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.graphics.rendering import display_board


def play_turn(policy, board, history):
    move, _ = policy.get_policy_output(board)
    board.apply_move(move, policy.color)
    history.append(move)
    display_board(board)
    return move


if __name__ == '__main__':
    for _ in range(5):
        print("======NEW GAME=======")
        board = GoBoard(BOARD_SIZE)
        black_policy = PolicyNN(color=BLACK)
        white_policy = PolicyNN(color=WHITE)

        game_history = []

        for _ in range(2):
            move_black = play_turn(black_policy, board, game_history)
            move_white = play_turn(white_policy, board, game_history)
            print(move_black)
            print(move_white)



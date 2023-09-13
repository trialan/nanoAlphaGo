from nanoAlphaGo.board import GoBoard
from nanoAlphaGo.policy import PolicyNN
from nanoAlphaGo.rendering import display_board

BLACK = 1
WHITE = -1

if __name__ == '__main__':
    board = GoBoard(size=19)
    black_policy = PolicyNN(board, color=BLACK)
    white_policy = PolicyNN(board, color=WHITE)

    for _ in range(5):
        black_move = black_policy.generate_move(board)
        board.apply_move(black_move, BLACK)
        display_board(board)
        white_move = white_policy.generate_move(board)
        board.apply_move(white_move, WHITE)
        display_board(board)



import numpy as np

from nanoAlphaGo.board import GoBoard


def test_board_initialisation():
    board = GoBoard(size=19)
    assert_board_is_initially_empty(board.board)


def test_getting_legal_moves():
    board = GoBoard(size=9)

    assert_first_move_options_are_correct(board)

    board = _setup_a_simple_board()
    moves_for_white = board.possible_moves(-1)
    assert len(moves_for_white) == 9*9 + 1 - 3

    _check_the_no_suicide_rule()


def test_counting_the_number_of_liberties():
    board = _setup_a_complicated_board()
    assert board.count_liberties((5,3)) == 4
    assert board.count_liberties((1,0)) == 1


def _check_the_no_suicide_rule():
    board = _setup_a_board_where_suicide_would_be_possible()
    suicide_move = (1,1)
    color = 1
    assert not board.is_valid_move(suicide_move, color)


def assert_board_is_initially_empty(board):
    row_wise_sum = np.sum(board, axis=1)
    assert np.sum(row_wise_sum) == 0


def assert_first_move_options_are_correct(board):
    moves_for_black = board.possible_moves(1)
    moves_for_white = board.possible_moves(-1)

    #First move is any intersection or pass
    assert len(moves_for_black) == 9*9 + 1
    assert len(moves_for_white) == 9*9 + 1


def _setup_a_simple_board():
    board = GoBoard(size=9)
    board.board = np.array([[1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,-1,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0]])
    return board


def _setup_a_board_where_suicide_would_be_possible():
    board = GoBoard(size=9)
    board.board = np.array([[1,1,-1,0,0,0,0,0,0],
                            [1,0,-1,0,0,0,0,0,0],
                            [-1,-1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,-1,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0]])
    return board


def _setup_a_complicated_board():
    board = GoBoard(size=9)
    board.board = np.array([[1,1,-1,0,0,0,0,0,0],
                            [1,0,-1,0,0,0,0,0,0],
                            [-1,-1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,-1,-1,0,0,-1,0,0],
                            [0,0,-1, 1,-1,0,0,0,0],
                            [0,-1,1,1,1,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0]])
    return board



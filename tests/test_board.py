import numpy as np
import pytest

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import BLACK, WHITE


def test_board_initialisation():
    board = GoBoard(size=19)
    assert_board_is_initially_empty(board)


def test_counting_the_number_of_liberties():
    board = _setup_a_complicated_board()
    assert board.count_liberties((5,3)) == 4
    assert board.count_liberties((1,0)) == 1
    assert board.count_liberties((6,4)) == 4
    assert board.count_liberties((4,6)) == 4 #Test group rule


def test_getting_legal_moves():
    board = GoBoard(size=9)
    assert_first_move_options_are_correct(board)
    board = _setup_a_simple_board()
    moves_for_white = board.legal_moves(WHITE)
    assert len(moves_for_white) == 9*9 + 1 - 3


def test_we_dont_allow_illegal_moves():
    _check_the_no_suicide_rule()
    _check_that_intersections_must_be_empty()
    _check_that_position_must_be_on_board()


def test_making_a_move_on_the_board():
    board = _setup_a_simple_board()
    board.apply_move(30, BLACK)
    assert board._matrix[3,3] == BLACK
    assert_that_tensor_is_same_board_as_matrix(board)

    with pytest.raises(AssertionError):
        """ Can't make illegal moves. """
        board.apply_move(30, WHITE)


def _check_the_no_suicide_rule():
    board = _setup_a_board_where_suicide_would_be_possible()
    suicide_move = (1,1)
    color = 1
    assert not board.is_valid_move(suicide_move, color)


def _check_that_position_must_be_on_board():
    board = GoBoard(size=9)
    bad_moves = [(11,3), (-1, 3)]
    assert_these_are_all_bad_moves(board, bad_moves)


def _check_that_intersections_must_be_empty():
    board = _setup_a_simple_board()
    bad_moves = [(0,0), (4,6), (7,1)]
    assert_these_are_all_bad_moves(board, bad_moves)


def assert_these_are_all_bad_moves(board, bad_moves):
    for bad_move in bad_moves:
        assert not board.is_valid_move(bad_move, 1)
        assert not board.is_valid_move(bad_move, -1)


def assert_that_tensor_is_same_board_as_matrix(board):
    assert np.array_equal(board.tensor[0].numpy(),
                          board._matrix)


def assert_board_is_initially_empty(board):
    assert board._matrix.sum() == 0


def assert_first_move_options_are_correct(board):
    moves_for_black = board.legal_moves(1)
    moves_for_white = board.legal_moves(-1)

    #First move is any intersection or pass
    assert len(moves_for_black) == 9*9 + 1
    assert len(moves_for_white) == 9*9 + 1


def _setup_a_simple_board():
    board = GoBoard(size=9)
    board._matrix = np.array([[1,0,0,0,0,0,0,0,0],
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
    board._matrix = np.array([[1,1,-1,0,0,0,0,0,0],
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
    board._matrix = np.array([[1,1,-1,0,0,0,0,0,0],
                            [1,0,-1,0,0,0,0,0,0],
                            [-1,-1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,-1,-1,0,0,-1,0,0],
                            [0,0,-1, 1,-1,0,0,0,0],
                            [0,-1,1,1,1,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0]])
    return board



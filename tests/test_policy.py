import numpy as np

from nanoAlphaGo.board import GoBoard
from nanoAlphaGo.rl.policy import _index_to_move, PolicyNN, _legal_move_mask


def test_that_model_is_outputting_moves_as_coordinates():
    board = GoBoard()
    policy = PolicyNN(color=1)
    for _ in range(10):
        move = policy.generate_move(board)
        assert_move_is_a_pair_of_coordinates(move)


def test_that_index_to_move_mapping_is_unique():
    board = GoBoard()
    coordinates = []
    for i in range(board.size**2+1):
        move = _index_to_move(i)
        coordinates.append(move)
    assert len(set(coordinates)) == board.size**2 + 1


def test_we_are_masking_the_right_number_of_moves():
    board = GoBoard(size=3)
    mask = _legal_move_mask(board, 1)
    assert sum(mask) == 3 * 3 + 1

    board = _setup_a_complicated_board()
    mask = _legal_move_mask(board, -1)
    number_of_stones_on_board = np.count_nonzero(board.board)
    assert sum(mask) == 9 * 9 + 1 - number_of_stones_on_board


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


def assert_move_is_a_pair_of_coordinates(move):
    assert type(move) == tuple
    assert len(move) == 2
    assert type(move[0]) == int
    assert type(move[1]) == int

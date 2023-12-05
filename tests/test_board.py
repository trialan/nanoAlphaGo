import numpy as np
import torch
import pytest

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import BLACK, WHITE
from nanoAlphaGo.game.scoring import NEIGHBORS, position_is_within_board
from nanoAlphaGo.rl.masking import generate_mask


def test_board_initialisation():
    board = GoBoard(size=19)
    assert_board_is_initially_empty(board)


def test_getting_legal_moves():
    board = GoBoard(size=9)
    assert_first_move_options_are_correct(board)
    board = _setup_a_simple_board()
    moves_for_white = board.legal_moves(WHITE)
    assert len(moves_for_white) == 9*9 - 3


def test_violating_ko_rule():
    board = GoBoard(size=2)

    board.previous_boards.append(np.array([[0,1], [0,0]]))
    board.previous_boards.append(np.array([[0,1], [1,0]]))
    board.previous_boards.append(np.array([[1,-1], [1,0]]))

    board._matrix = np.array([[0,0],[1,0]])
    ko_violating_move = (0,1)
    assert board.violates_ko_rule(ko_violating_move, color=1) is True


def test_counting_the_number_of_liberties():
    board = _setup_a_complicated_board()
    assert board.count_liberties((5,3)) == 4
    assert board.count_liberties((6,4)) == 4
    assert board.count_liberties((4,6)) == 4 #Test group rule
    assert board.count_liberties((1,0)) == 1

    board = _setup_two_eyed_group_board()
    assert board.count_liberties((0,1)) == 2
    assert board.count_liberties((3,0)) == 2
    board.apply_move(18, BLACK) #18 is int representation of (2,0)
    assert board.count_liberties((0,1)) == 1
    assert board.count_liberties((3,0)) == 1


def test_certain_boards_for_legal_moves():
    """ These boards occurred during games that had a suspicious # of moves """
    tensors = torch.tensor([[[ 1.,  1.,  1.,  1.,  1.,  0.,  0., -1., -1.],
                      [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  0.],
                      [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  0.],
                      [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                      [ 1.,  0.,  0., -1.,  0.,  1.,  1.,  1.,  0.],
                      [ 1., -1., -1.,  1.,  1., -1.,  1.,  1.,  1.],
                      [ 1.,  1., -1.,  1., -1., -1.,  1.,  0.,  0.],
                      [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
                      [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.]]])
    board = GoBoard(initial_state_matrix=tensors[0].numpy())
    expected_legal_moves = [
                (0,5), (0,6),
                (1,4), (1,7), (1,8),
                (2,1), (2,5), (2,8),
                (4,1), (4,2), (4,4), (4,8),
                (6,7), (6,8),
                (7,1), (7,5), (7,7),
                (8,7), (8,8)
            ]
    """ On this board these moves are illegal for white:
    (1,4), (7,1), (7,5), (4,8), : suicide """
    white_suicide_moves = set({64, 68, 44, 13, 19, 23})

    black_legal_moves = board.legal_moves(BLACK)
    white_legal_moves = board.legal_moves(WHITE)
    legal_moves = set(black_legal_moves).union(set(white_legal_moves))
    assert sorted(legal_moves) == sorted(expected_legal_moves)

    expected_ixs = sorted([a*9+b for (a,b) in expected_legal_moves])
    black_mask = generate_mask(tensors, BLACK).numpy()
    nonzero_mask_ixs = sorted(list(np.nonzero(black_mask)[0]))
    assert expected_ixs == nonzero_mask_ixs[:-1] #mask includes 'PASS' move
    white_mask = generate_mask(tensors, WHITE).numpy()
    nonzero_mask_ixs = sorted(list(np.nonzero(white_mask)[0]))
    assert set(expected_ixs) - set(nonzero_mask_ixs[:-1]) == white_suicide_moves


def test_suicide_rule_special_case():
    """ If a group is surrounded but has one 'eye' it is legal for the
        surrounding player to place his stone inside the eye and thus
        capture the group. Simplistic application of 'no suicide' rule
        would prevent this.

        Reference:
        https://www.pandanet.co.jp/English/learning_go/learning_go_7.html
        """
    board = _setup_one_eyed_group_board()
    legal_moves = board.legal_moves(BLACK)
    assert_neighbours_have_just_one_liberty(board, (3,3))
    assert (3,3) in legal_moves


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
    """ board.legal_moves doesn't return PASS """
    moves_for_black = board.legal_moves(1)
    moves_for_white = board.legal_moves(-1)

    #First move is any intersection or pass
    assert len(moves_for_black) == 9*9
    assert len(moves_for_white) == 9*9


def _setup_a_simple_board():
    board = GoBoard(size=9)
    board_matrix = np.array([[1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,-1,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0]])
    for x in range(board.size):
        for y in range(board.size):
            color = board_matrix[x, y]
            if color != 0:
                index = x * board.size + y
                board.apply_move(index, color)
    return board


def _setup_a_board_where_suicide_would_be_possible():
    board = GoBoard(size=9)
    board_matrix = np.array([[1,1,-1,0,0,0,0,0,0],
                            [1,0,-1,0,0,0,0,0,0],
                            [-1,-1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,-1,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0]])
    for x in range(board.size):
        for y in range(board.size):
            color = board_matrix[x, y]
            if color != 0:
                index = x * board.size + y
                board.apply_move(index, color)
    return board


def _setup_a_complicated_board():
    board = GoBoard(size=9)
    board_matrix = np.array([[1,1,-1,0,0,0,0,0,0],
                            [1,0,-1,0,0,0,0,0,0],
                            [-1,-1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,-1,-1,0,0,-1,0,0],
                            [0,0,-1, 1,-1,0,0,0,0],
                            [0,-1,1,1,1,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0]])
    for x in range(board.size):
        for y in range(board.size):
            color = board_matrix[x, y]
            if color != 0:
                index = x * board.size + y
                board.apply_move(index, color)
    return board


def assert_neighbours_have_just_one_liberty(board, position):
    neighbours = NEIGHBORS[position]
    for n in neighbours:
        x,y = n
        assert board.count_liberties((x,y)) == 1


def _setup_one_eyed_group_board():
    """ White has a fully surrounded group with a single eye, this can be
    captured by black in a single move """
    matrix = np.zeros((9,9))
    for col in [1,2,3,4,5]:
        matrix[1,col] = BLACK
    for row in [2,3,4]:
        matrix[row,1] = BLACK
    for col in [2,3,4]:
        matrix[5,col] = BLACK
    for row in [1,2,3,4]:
        matrix[row,5] = BLACK
    for col in [2,3,4]:
        matrix[2,col] = WHITE
    matrix[3,2] = WHITE
    matrix[3,4] = WHITE
    for col in [2,3,4]:
        matrix[4,col] = WHITE
    board = GoBoard(initial_state_matrix=matrix)
    return board


def _setup_two_eyed_group_board():
    matrix = np.zeros((9,9))
    for row in [1,3]:
        matrix[row,0] = BLACK
    for row in [0,1,2,3]:
        matrix[row,1] = BLACK
    for row in [0,1,2,3]:
        matrix[row,2] = WHITE
    for col in [0,1]:
        matrix[4,col] = WHITE
    board = GoBoard(initial_state_matrix=matrix)
    return board



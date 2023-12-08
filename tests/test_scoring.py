import numpy as  np

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import WHITE, BLACK
from nanoAlphaGo.game.scoring import calculate_score, find_reached
from nanoAlphaGo.graphics.rendering import display_board


def test_calculating_tricky_scores():
    matrix = np.array([[ 1.,  1.,  1.,  1., -1., -1., -1.,  1.,  1.],
                   [ 1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.],
                   [ 1.,  1., -1., -1.,  1., -1., -1.,  0.,  1.],
                   [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1.],
                   [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  0.],
                   [ 1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1.],
                   [-1.,  1., -1., -1.,  1.,  1., -1., -1.,  0.],
                   [-1., -1.,  0.,  1., -1.,  1.,  1., -1.,  1.],
                   [ 1.,  1.,  1.,  1.,  0., -1.,  1., -1.,  1.]])
    n_white_stones = np.sum(matrix == WHITE)
    n_black_stones = np.sum(matrix == BLACK)
    white_territory = 0
    black_territory = 1
    expected_white_score = n_white_stones + white_territory
    expected_black_score = n_black_stones + black_territory
    score = calculate_score(matrix, komi=0)
    assert score[BLACK] == expected_black_score
    assert score[WHITE] == expected_white_score

    matrix = np.array([[ 1.,  1.,  1., -1., -1., -1., -1., -1., -1.],
               [ 1.,  1.,  1., -1., -1., -1., -1.,  0., -1.],
               [ 1.,  1.,  1., -1.,  0., -1., -1., -1., -1.],
               [ 1., -1.,  1.,  1., -1., -1., -1., -1., -1.],
               [-1., -1., -1., -1., -1., -1.,  0., -1., -1.],
               [-1.,  0., -1.,  1., -1.,  0.,  1.,  0.,  1.],
               [ 1.,  1.,  1.,  1., -1.,  1., -1., -1., -1.],
               [ 0.,  1., -1.,  0.,  1., -1.,  0., -1.,  0.],
               [-1.,  0.,  1., -1.,  0., -1., -1., -1.,  1.]])
    n_white_stones = 45
    n_black_stones = 24
    white_territory = 3
    black_territory = 0

    expected_white_score = n_white_stones + white_territory
    expected_black_score = n_black_stones + black_territory
    score = calculate_score(matrix, komi=0)
    assert score[BLACK] == expected_black_score
    assert score[WHITE] == expected_white_score


def test_calculating_score_case_one():
    board = GoBoard()
    board._matrix[0,1] = 1
    board._matrix[0,2] = 1
    board._matrix[1,0] = 1
    board._matrix[1,3] = 1
    board._matrix[2,1] = 1
    board._matrix[2,2] = 1

    board._matrix[0,4] = -1
    board._matrix[1,4] = -1

    score = calculate_score(board._matrix, komi=0)
    assert score[WHITE] == 2
    assert score[BLACK] == 9


def test_calculating_score_case_two():
    board = GoBoard()
    board._matrix[0,1] = 1
    board._matrix[0,2] = 1
    board._matrix[1,0] = 1
    board._matrix[1,3] = 1
    board._matrix[2,1] = 1
    board._matrix[2,2] = 1

    board._matrix[0,-2] = -1
    board._matrix[1,-1] = -1

    board._matrix[6,8] = -1
    board._matrix[6,7] = -1
    board._matrix[6,6] = -1
    board._matrix[6,5] = -1
    board._matrix[6,4] = -1
    board._matrix[7,4] = -1
    board._matrix[8,4] = -1

    score = calculate_score(board._matrix, komi=0)
    assert score[WHITE] == 18
    assert score[BLACK] == 9


def test_calculating_score_case_three():
    board = GoBoard()

    for i in range(board.size):
        board._matrix[1,i] = -1

    for j in range(board.size):
        board._matrix[7,j] = 1

    score = calculate_score(board._matrix, komi=0)
    assert score[BLACK] == 18
    assert score[WHITE] == 18


def test_finding_the_chains_and_boundaries():
    board = GoBoard()

    board._matrix[0,5] = 1
    board._matrix[4,2] = 1
    board._matrix[4,3] = 1
    board._matrix[5,2] = 1

    chain, reached = find_reached(board._matrix, (0,5))
    assert chain == {(0,5)}
    assert reached == {(0,4), (0,6), (1,5)}

    chain, reached = find_reached(board._matrix, (4,2))
    assert chain == {(5,2), (4,2), (4,3)}
    assert reached == {(6,2), (5,3), (5,1), (4,4), (3,3), (3,2), (4,1)}

    for position in chain:
        c, r = find_reached(board._matrix, position)
        assert c == chain
        assert r == reached



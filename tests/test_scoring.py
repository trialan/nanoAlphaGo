import numpy as  np

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import WHITE, BLACK
from nanoAlphaGo.game.scoring import calculate_score, find_reached


def test_finding_the_chains_and_boundaries():
    board = GoBoard()

    board.matrix[0,5] = 1
    board.matrix[4,2] = 1
    board.matrix[4,3] = 1
    board.matrix[5,2] = 1

    chain, reached = find_reached(board.matrix, (0,5))
    assert chain == {(0,5)}
    assert reached == {(0,4), (0,6), (1,5)}

    chain, reached = find_reached(board.matrix, (4,2))
    assert chain == {(5,2), (4,2), (4,3)}
    assert reached == {(6,2), (5,3), (5,1), (4,4), (3,3), (3,2), (4,1)}

    for position in chain:
        c, r = find_reached(board.matrix, position)
        assert c == chain
        assert r == reached


def test_calculating_score_case_one():
    board = GoBoard()
    board.matrix[0,1] = 1
    board.matrix[0,2] = 1
    board.matrix[1,0] = 1
    board.matrix[1,3] = 1
    board.matrix[2,1] = 1
    board.matrix[2,2] = 1

    board.matrix[0,4] = -1
    board.matrix[1,4] = -1

    score = calculate_score(board.matrix, komi=0)
    assert score[WHITE] == 2
    assert score[BLACK] == 9


def test_calculating_score_case_two():
    board = GoBoard()
    board.matrix[0,1] = 1
    board.matrix[0,2] = 1
    board.matrix[1,0] = 1
    board.matrix[1,3] = 1
    board.matrix[2,1] = 1
    board.matrix[2,2] = 1

    board.matrix[0,-2] = -1
    board.matrix[1,-1] = -1

    board.matrix[6,8] = -1
    board.matrix[6,7] = -1
    board.matrix[6,6] = -1
    board.matrix[6,5] = -1
    board.matrix[6,4] = -1
    board.matrix[7,4] = -1
    board.matrix[8,4] = -1

    score = calculate_score(board.matrix, komi=0)
    assert score[WHITE] == 18
    assert score[BLACK] == 9


def test_calculating_score_case_three():
    board = GoBoard()

    for i in range(board.size):
        board.matrix[1,i] = -1

    for j in range(board.size):
        board.matrix[7,j] = 1

    score = calculate_score(board.matrix, komi=0)
    assert score[BLACK] == 18
    assert score[WHITE] == 18


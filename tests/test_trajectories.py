import random
import numpy as np
import pytest

from nanoAlphaGo.config import WHITE, BLACK
from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.trajectories import collect_trajectories, game_is_over


@pytest.mark.skip
def test_collecting_trajectories():
    policy = PolicyNN(WHITE)
    trajectories = collect_trajectories(policy, n_trajectories=5)

    assert len(trajectories) == 5
    assert set(trajectories[0].keys()) == {'rewards',
                                           'moves',
                                           'board_states',
                                           "move_probs"}
    assert all([sum(t['rewards']) in {-1,0,1} for t in trajectories])


def test_checking_if_a_game_is_over():
    board = GoBoard()
    consecutive_passes = 2
    assert game_is_over(board, consecutive_passes, WHITE)


def _setup_a_complicated_board():
    board = GoBoard(size=9)
    board.matrix = np.array([[1,1,-1,0,0,0,0,0,0],
                            [1,0,-1,0,0,0,0,0,0],
                            [-1,-1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,-1,-1,0,0,-1,0,0],
                            [0,0,-1, 1,-1,0,0,0,0],
                            [0,-1,1,1,1,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0]])
    return board

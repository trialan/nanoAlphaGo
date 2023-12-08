import numpy as np
import pytest
import random
import torch

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import WHITE, BLACK, BOARD_SIZE
from nanoAlphaGo.rl.policy import PolicyNN, assert_sum_is_less_than_or_equal_to_one
from nanoAlphaGo.rl.trajectories import collect_trajectories, game_is_over, st_collect_trajectories
from nanoAlphaGo.rl.utils import change_wandb_mode_for_testing


change_wandb_mode_for_testing("disabled")

def test_playing_single_game():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    policy = PolicyNN(WHITE)
    #multiprocessed version
    trajectories = collect_trajectories(policy, n_trajectories=1)
    final_reward = trajectories[0]['rewards'][-1]
    #single threaded version (kept for debuggin purposes)
    trajectories = st_collect_trajectories(policy, n_trajectories=1)


def test_collecting_trajectories():
    policy = PolicyNN(WHITE)
    N = 5
    trajectories = st_collect_trajectories(policy, n_trajectories=N)

    assert len(trajectories) == N
    assert set(trajectories[0].keys()) == {'rewards',
                                           'moves',
                                           'board_states',
                                           'move_probs'}

    assert_all_moves_are_on_the_board(trajectories)
    assert_all_move_probs_are_probs(trajectories)
    assert_board_states_are_correct_dimensions(trajectories)
    assert_rewards_are_right_type(trajectories)
    assert_trajectories_are_different(trajectories)


def assert_trajectories_are_different(trajectories):
    assert not np.array_equal(trajectories[0]['moves'].numpy(),
                              trajectories[1]['moves'].numpy())


def test_checking_if_a_game_is_over():
    board = GoBoard()
    game_data = {"consecutive_passes" : 2,
                 "player": WHITE,
                 "turn": 5,
                 "moves": [1,2,3,4,5]}
    assert game_is_over(board, game_data)

    board._matrix = np.ones((board.size, board.size))
    game_data['consecutive_passes'] = 0
    assert game_is_over(board, game_data)

    board = GoBoard()
    game_data = {"consecutive_passes" : 0,
                 "player": WHITE,
                 "turn": 105,
                 "moves": [1,2,3]}
    assert game_is_over(board, game_data)
    game_data['turn'] = 65
    assert not game_is_over(board, game_data)

def assert_all_moves_are_on_the_board(trajectories):
    for t in trajectories:
        for m in t['moves']:
            assert m <= BOARD_SIZE * BOARD_SIZE


def assert_all_move_probs_are_probs(trajectories):
    for t in trajectories:
        for p in t['move_probs']:
            assert_sum_is_less_than_or_equal_to_one(p)


def assert_board_states_are_correct_dimensions(trajectories):
    for t in trajectories:
        assert all([s.shape == torch.Size([1,BOARD_SIZE,BOARD_SIZE])
                    for s in t['board_states']])


def assert_rewards_are_right_type(trajectories):
    assert all([sum(t['rewards']).item() in {-1,0,1} for t in trajectories])


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


change_wandb_mode_for_testing("enabled")

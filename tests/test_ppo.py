import numpy as np
import os
import pytest
import subprocess
import torch
import wandb

from nanoAlphaGo.config import WHITE, RL_params
from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.ppo import ppo_train, multiply_with_dim_correction
from nanoAlphaGo.rl.utils import add_rewards_to_go_to_trajectories, change_wandb_mode_for_testing
from nanoAlphaGo.rl.trajectories import collect_trajectories
from nanoAlphaGo.rl.value import ValueNN

change_wandb_mode_for_testing("disabled")

@pytest.mark.skip()
def test_ppo_training_runs():
    RL_params["n_trajectories"] = 2
    ppo_train(PolicyNN(WHITE), ValueNN(), n_loops=1)


def test_computing_rewards_to_go():
    policy = PolicyNN(WHITE)
    trajectories = collect_trajectories(policy, n_trajectories=2)
    add_rewards_to_go_to_trajectories(trajectories)
    assert_trajectories_have_right_dimensions(trajectories)


def test_pytorch_multiplication_in_advantages():
    batch_size = 2
    action_space_size = 3 #is 82 in Go, 5 for readability
    ratios = torch.rand((batch_size, 1, action_space_size))
    adv = torch.rand((batch_size))

    term = multiply_with_dim_correction(ratios, adv)

    assert np.isclose(ratios[0][0][0] * adv[0], term[0][0][0])
    assert np.isclose(ratios[0][0][1] * adv[0], term[0][0][1])
    assert np.isclose(ratios[0][0][2] * adv[0], term[0][0][2])

    assert np.isclose(ratios[1][0][0] * adv[1], term[1][0][0])
    assert np.isclose(ratios[1][0][1] * adv[1], term[1][0][1])
    assert np.isclose(ratios[1][0][2] * adv[1], term[1][0][2])


def assert_trajectories_have_right_dimensions(trajectories):
    keys = trajectories[0].keys()
    for t in trajectories:
        lengths = set(len(t[k]) for k in keys)
        assert len(lengths) == 1


change_wandb_mode_for_testing("enabled")

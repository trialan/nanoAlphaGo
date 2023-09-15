import random
import numpy as np

from nanoAlphaGo.config import WHITE
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.trajectories import collect_trajectories


def test_collecting_trajectories():
    policy = PolicyNN(WHITE)
    trajectories = collect_trajectories(policy, n_trajectories=5)

    assert len(trajectories) == 5
    assert set(trajectories[0].keys()) == {'rewards',
                                           'moves',
                                           'board_states'}
    assert all([sum(t['rewards']) in {-1,0,1} for t in trajectories])



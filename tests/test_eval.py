import numpy as np
import pytest

from nanoAlphaGo.config import WHITE
from nanoAlphaGo.rl.eval import  performance_against_random_policy
from nanoAlphaGo.rl.policy import PolicyNN


@pytest.mark.skip
def test_calculating_performance():
    """ Bad test, should check ideal win rate or something. Tricky. """
    policy = PolicyNN(color=WHITE)
    performance = performance_against_random_policy(policy, n_games=2)
    assert "win_rate" in performance
    assert "std_err" in performance


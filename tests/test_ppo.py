import pytest

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import WHITE
from nanoAlphaGo.rl.ppo import ppo_train
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.value import value_function, ValueNN


def test_ppo_training_runs():
    board = GoBoard()
    ppo_train(PolicyNN(WHITE), ValueNN(board), n_loops=1)

import time
import numpy as np
import wandb

from nanoAlphaGo.rl.ppo import ppo_train
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.value import ValueNN
from nanoAlphaGo.rl.eval import estimate_win_rate
from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import WHITE, KOMI, RL_params


if __name__ == '__main__':
    wandb.init(project='RL Go', entity='thomasrialan')

    policy_network = PolicyNN(WHITE)
    value_network = ValueNN()

    for i in range(100):
        performance = estimate_win_rate(policy_network)
        ppo_train(policy_network, value_network, n_loops=400)

    wandb.finish()


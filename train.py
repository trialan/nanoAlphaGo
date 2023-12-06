import time
import numpy as np
import wandb

from nanoAlphaGo.rl.ppo import ppo_train
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.value import ValueNN
from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import WHITE, KOMI, RL_params


if __name__ == '__main__':
    wandb.init(project='RL Go', entity='thomasrialan')

    policy_network = PolicyNN(WHITE)
    value_network = ValueNN()
    ppo_train(policy_network, value_network, n_loops=RL_params["n_ppo_loops"])

    wandb.finish()


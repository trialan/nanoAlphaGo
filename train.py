import time
import wandb

from nanoAlphaGo.rl.ppo import ppo_train
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.value import ValueNN
from nanoAlphaGo.rl.eval import performance_against_random_policy
from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import WHITE

EPOCHS = 1000

if __name__ == '__main__':
    board = GoBoard()

    policy_network = PolicyNN(WHITE)
    value_network = ValueNN(board)

    wandb.init(project='RL Go', entity='thomasrialan')
    for _ in range(EPOCHS):
        performance = performance_against_random_policy(policy_network,
                                                        n_games=1000)
        ppo_train(policy_network, value_network, n_loops=5)
        print(performance)


    wandb.finish()

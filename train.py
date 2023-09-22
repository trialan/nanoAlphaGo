from nanoAlphaGo.rl.ppo import ppo_train
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.value import ValueNN
from nanoAlphaGo.rl.eval import performance_against_random_policy
from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import WHITE

EPOCHS = 5

if __name__ == '__main__':
    board = GoBoard()

    policy_network = PolicyNN(WHITE)
    value_network = ValueNN(board)

    for _ in range(EPOCHS):
        ppo_train(policy_network, value_network, n_loops=2)
        performance = performance_against_random_policy(policy_network,
                                                        n_games=2)
        print(performance)



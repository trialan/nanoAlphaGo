import time
import numpy as np
import wandb

from nanoAlphaGo.rl.ppo import ppo_train
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.value import ValueNN
from nanoAlphaGo.rl.eval import performance_against_random_policy
from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import WHITE, KOMI, RL_params

EPOCHS = 1000
N_GAMES = 10

if __name__ == '__main__':
    print(f"Komi: {KOMI}")
    print(f"N games: {N_GAMES}")
    print(f"Params: {RL_params}")
    from tqdm import tqdm

    performances = []
    for i in range(50):
        print(i)
        #np.random.seed(np.random.randint(100))
        policy_network = PolicyNN(WHITE)
        #value_network = ValueNN()
        performance = performance_against_random_policy(policy_network,
                                                        n_games=N_GAMES)
        #ppo_train(policy_network, value_network, n_loops=5)
        print(performance)
        performances.append(performance)

    rates = [p['win_rate'] for p in performances]
    mean = np.mean(rates)
    se = np.std(rates) / len(rates) ** 0.5
    print("STATS")
    print(f"Mean: {mean}, SE: {se}")

    plt.figure(figsize=(8, 6))
    plt.hist(rates, bins=5, color='blue', alpha=0.7)
    plt.title('Histogram of Win Rates')
    plt.xlabel('Win Rate')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()



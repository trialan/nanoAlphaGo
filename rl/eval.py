import math
import numpy as np
import wandb

from nanoAlphaGo.rl.trajectories import collect_trajectories


def performance_against_random_policy(policy, n_games):
    trajectories = collect_trajectories(policy, n_trajectories=n_games)
    game_outcomes = [t['rewards'][-1] for t in trajectories]
    win_count = game_outcomes.count(1)
    win_percentage = win_count / n_games
    standard_error = get_std_err(game_outcomes)
    performance = {"win_rate": win_percentage,
                   "std_err": standard_error}
    #wandb.log(performance)
    return performance


def get_std_err(game_outcomes):
    wins = [w.item() if w==1 else 0 for w in game_outcomes]
    print(wins)
    std_err = np.std(wins) / len(wins) ** 0.5
    return std_err



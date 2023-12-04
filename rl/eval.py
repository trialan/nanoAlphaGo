import numpy as np
from tqdm import tqdm
import wandb

from nanoAlphaGo.rl.trajectories import collect_trajectories


def estimate_win_rate(policy, n_samples=10, n_games=1000):
    rates = []
    for i in tqdm(range(n_samples), desc="Estimating win rate"):
        performance = performance_against_random_policy(policy, n_games)
        rates.append(performance['win_rate'])
    out = {"win_rate_mean": np.mean(rates),
           "win_rate_se": np.std(rates) / len(rates) ** 0.5}
    wandb.log(out)
    return out


def performance_against_random_policy(policy, n_games):
    trajectories = collect_trajectories(policy, n_trajectories=n_games)
    game_outcomes = [t['rewards'][-1] for t in trajectories]
    win_count = game_outcomes.count(1)
    win_percentage = win_count / n_games
    standard_error = get_std_err(game_outcomes)
    performance = {"win_rate": win_percentage,
                   "std_err": standard_error}
    return performance


def get_std_err(game_outcomes):
    wins = [w.item() if w==1 else 0 for w in game_outcomes]
    std_err = np.std(wins) / len(wins) ** 0.5
    return std_err



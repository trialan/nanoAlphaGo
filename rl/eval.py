import math
import wandb

from nanoAlphaGo.rl.trajectories import collect_trajectories


def performance_against_random_policy(policy, n_games):
    trajectories = collect_trajectories(policy, n_trajectories=n_games)
    game_outcomes = [t['rewards'][-1] for t in trajectories]

    win_count = game_outcomes.count(1)
    total_games = len(game_outcomes)
    win_percentage = win_count / total_games if total_games > 0 else 0

    standard_error = get_std_err(total_games, win_percentage)

    performance = {"win_rate": win_percentage,
                   "std_err": standard_error}
    wandb.log(performance)

    return performance


def get_std_err(total_games, win_percentage):
    if total_games > 0 and win_percentage > 0 and win_percentage < 1:
        standard_error = math.sqrt(win_percentage * (1 - win_percentage)) / total_games
    else:
        standard_error = 0
    return standard_error



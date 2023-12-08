WHITE = -1
BLACK = 1
EMPTY = 0
SEKI = 2  # We'll use this value for the dame or seki situations
BOARD_SIZE = 5
PASS = BOARD_SIZE * BOARD_SIZE
KOMI = 0.

RL_params = {
            "total_games": 50000,
            "n_trajectories_per_step": 50*3,
            "learning_rate_policy": 0.0003,
            "learning_rate_value": 0.0003,
            "epsilon": 0.2,
            "gamma": 0.99,
            "lambda": 0.95,
            }
RL_params['n_ppo_loops'] = int(RL_params["total_games"]/RL_params["n_trajectories_per_step"])
SEED = 142

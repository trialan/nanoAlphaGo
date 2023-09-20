import torch

from nanoAlphaGo.config import BOARD_SIZE, PASS, RL_params

gamma = RL_params["gamma"]
lambda_ = RL_params["lambda"]


def numerically_stable_tensor_division(a, b):
    tiny_number = 1e-10
    return a / (b + tiny_number)


def _nn_tensor_from_matrix(board_matrix):
    """ Conv2D wants a 4D input. """
    x = torch.tensor(board_matrix, dtype=torch.float32)
    x = x.unsqueeze(0)
    return x


def _index_to_move(index):
    """ Returns a unique set of coordinates for each integer. """
    if index == BOARD_SIZE * BOARD_SIZE:
        return PASS
    else:
        x, y = divmod(index, BOARD_SIZE)
        return (x, y)


def add_rewards_to_go_to_trajectories(trajectories):
    rewards_to_go_list = []
    for trajectory in trajectories:
        trajectory = _compute_rtg_single_trajectory(trajectory)
    return trajectories


def compute_advantages(trajectories, valueNN):
    advantages_list = []
    for trajectory in trajectories:
        rewards_to_go = trajectory['rewards_to_go']
        states = trajectory['board_states']

        values = valueNN(states)

        advantages = []
        gae = 0

        for t in reversed(range(len(rewards_to_go))):
            delta = rewards_to_go[t] - values[t]
            if t < len(rewards_to_go) - 1:
                delta += gamma * values[t+1]
            gae = delta + gamma * lambda_ * gae
            advantages.insert(0, gae)

        advantages_list.append(torch.tensor(advantages))

    num_states = sum(len(t['board_states']) for t in trajectories)
    advantages = torch.cat(advantages_list)

    check_advantages(advantages, correct_len=num_states)
    return advantages


def check_advantages(advantages, correct_len):
    assert len(advantages) == correct_len
    assert not torch.isnan(advantages).any()
    max_val = advantages.max().item()
    min_val = advantages.min().item()
    assert max_val < 1e10 and min_val > -1e10


def _compute_rtg_single_trajectory(trajectory):
    outcome = trajectory['rewards'][-1]
    n_zeros = len(trajectory['board_states']) - 1
    rewards_to_go = [0 for _ in range(n_zeros)] + [outcome]
    trajectory['rewards_to_go'] = torch.tensor(rewards_to_go,
                                               dtype=torch.float32)
    return trajectory



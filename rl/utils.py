import torch
import random
import numpy as np
import wandb
import subprocess

from nanoAlphaGo.config import BOARD_SIZE, PASS, RL_params

gamma = RL_params["gamma"]
lambda_ = RL_params["lambda"]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


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
    device = get_device_of_trajectory(trajectories[0])

    advantages_list = []
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        states = trajectory['board_states']
        values = valueNN(states).squeeze(1)
        advantages = tr_calculate_advantages(rewards, values)
        #adv_2 = calculate_advantages(rewards, values)
        advantages_list.append(advantages)

    num_states = sum(len(t['board_states']) for t in trajectories)
    advantages = torch.cat(advantages_list).to(device)
    wandb.log({"advantages": advantages})

    check_advantages(advantages, correct_len=num_states)
    return advantages


def special_adv(rewards, values):
    adv = np.append((values[1:] - values[:-1]).detach().numpy(),0)
    return torch.tensor(adv)


def tr_calculate_advantages(rewards, values):
    n_steps = len(rewards)
    deltas = [0]
    for t in range(n_steps - 1):
        delta_t = rewards[t] + gamma * values[t+1] - values[t]
        deltas.append(delta_t.item())
    advantages = []
    T = n_steps
    for t in range(T):
        delta_ixs = list(range(t, T))
        a_t_delta = np.array([deltas[ix] for ix in delta_ixs])
        gl_exp = [x for x in range(T-t)]
        a_t_gl = np.repeat(gamma*lambda_, T-t) #last valid ix is T-t-1
        scale_term = a_t_gl ** gl_exp
        a_t_terms = a_t_delta * scale_term
        advantages.append(sum(a_t_terms))
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages


def calculate_advantages(rewards, values):
    advantages = []
    advantage = 0
    next_value = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + next_value * gamma - v
        advantage = td_error + advantage * gamma * lambda_
        next_value = v
        advantages.insert(0, advantage)
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages


def get_device_of_trajectory(trajectory):
    return trajectory['board_states'][0].device


def check_advantages(advantages, correct_len):
    assert len(advantages) == correct_len
    assert not torch.isnan(advantages).any()
    max_val = advantages.max().item()
    min_val = advantages.min().item()
    assert max_val < 1e10 and min_val > -1e10


def _compute_rtg_single_trajectory(trajectory):
    device = get_device_of_trajectory(trajectory)
    outcome = trajectory['rewards'][-1]
    n_zeros = len(trajectory['board_states']) - 1
    rewards_to_go = [outcome for _ in range(n_zeros)] + [outcome]
    trajectory['rewards_to_go'] = torch.tensor(rewards_to_go,
                                               dtype=torch.float32).to(device)
    return trajectory


def change_wandb_mode_for_testing(mode):
    assert mode in ["enabled", "disabled"]
    subprocess.run(["wandb", mode])
    if mode == "disabled":
        wandb.init()



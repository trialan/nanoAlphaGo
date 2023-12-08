import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

from nanoAlphaGo.config import RL_params
from nanoAlphaGo.rl.debugging import check_no_nan_gradients
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.trajectories import collect_trajectories
from nanoAlphaGo.rl.utils import (add_rewards_to_go_to_trajectories,
                                  compute_advantages,
                                  numerically_stable_tensor_division)
from nanoAlphaGo.rl.value import ValueNN


n_trajectories = RL_params["n_trajectories_per_step"]
lr_policy = RL_params["learning_rate_policy"]
lr_value = RL_params["learning_rate_value"]
epsilon = RL_params["epsilon"]


def ppo_train(policy_net, value_net, n_loops):
    policy_opt, value_opt = setup_optimizers(policy_net, value_net)
    for loop in tqdm(range(n_loops), desc="PPO Steps"):
        trajectories = collect_trajectories(policy_net, n_trajectories)
        add_rewards_to_go_to_trajectories(trajectories)
        advantages = compute_advantages(trajectories, value_net)
        if loop > n_loops/2:
            update_policy(policy_net, policy_opt, trajectories, advantages)
        update_value_function(value_net, value_opt, trajectories)


def update_policy(policy_net, optimizer, trajectories, advantages):
    optimizer.zero_grad()
    loss = compute_policy_loss(policy_net, trajectories, advantages)
    wandb.log({"policy objective": loss.item()})
    perform_backprop(policy_net, optimizer, loss)


def update_value_function(value_net, optimizer, trajectories):
    optimizer.zero_grad()
    loss = compute_value_loss(value_net, trajectories)
    wandb.log({"value function loss": loss.item()})
    perform_backprop(value_net, optimizer, loss)


def perform_backprop(network, optimizer, loss):
    loss.backward()
    check_no_nan_gradients(network)
    optimizer.step()


def compute_value_loss(value_net, trajectories):
    states = torch.cat([t['board_states'] for t in trajectories])
    rewards_to_go = [t['rewards_to_go'] for t in trajectories]
    rewards_to_go = torch.cat(rewards_to_go)
    values = value_net(states).squeeze()
    wandb.log({"values": values})
    loss = F.mse_loss(values, rewards_to_go)
    return loss


def compute_policy_loss(policy_net, trajectories, advantages):
    ratio = compute_loss_ratio(trajectories, policy_net)
    clip_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clip_ratio_times_adv = multiply_with_dim_correction(clip_ratio, advantages)
    ratio_times_adv = multiply_with_dim_correction(ratio, advantages)
    loss = torch.min(ratio_times_adv, clip_ratio_times_adv).mean()
    return loss


def compute_loss_ratio(trajectories, policy_net):
    states = torch.cat([t['board_states'] for t in trajectories])
    old_probs = torch.cat([t['move_probs'] for t in trajectories])
    new_probs = policy_net(states).unsqueeze(1)
    ratio = numerically_stable_tensor_division(new_probs, old_probs)
    return ratio


def multiply_with_dim_correction(ratios, advantages):
    reshaped_adv = advantages.unsqueeze(1).unsqueeze(2)
    product = ratios * reshaped_adv
    return product


def setup_optimizers(policy, value):
    policy_optimizer = optim.Adam(policy.parameters(),
                                  lr=lr_policy,
                                  maximize=True)
    value_optimizer = optim.Adam(value.parameters(),
                                 lr=lr_value,
                                 maximize=False)
    return policy_optimizer, value_optimizer



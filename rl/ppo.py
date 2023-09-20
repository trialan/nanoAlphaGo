import torch
import torch.nn.functional as F
import torch.optim as optim

from nanoAlphaGo.config import RL_params
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.trajectories import collect_trajectories
from nanoAlphaGo.rl.value import ValueNN
from nanoAlphaGo.rl.debugging import check_nn_gradients, check_advantages
from nanoAlphaGo.rl.utils import compute_rewards_to_go, compute_advantages


n_trajectories = RL_params["n_trajectories"]
learning_rate_policy = RL_params["learning_rate_policy"]
learning_rate_value = RL_params["learning_rate_value"]
epsilon = RL_params["epsilon"]


def ppo_train(policyNN, valueNN, n_loops=5):
    optimizers = setup_optimizers(policyNN, valueNN)
    policy_optimizer, value_optimizer = optimizers
    for k in range(n_loops):
        trajectories = collect_trajectories(policyNN, n_trajectories)
        trajectories = compute_rewards_to_go(trajectories)
        advantages = compute_advantages(trajectories,
                                        valueNN)
        update_policy_ppoclip(policy_optimizer,
                              policyNN,
                              trajectories,
                              advantages)
        update_value_function_mse(value_optimizer,
                                  valueNN,
                                  trajectories)


def update_policy_ppoclip(optimizer, policyNN, trajectories, advantages):
    assert len(trajectories) == len(advantages)

    states, actions, old_probs = zip(*[(t['board_states'], t['moves'], t['move_probs']) for t in trajectories])
    states = torch.cat(states)
    actions = torch.cat(actions)
    old_probs = torch.cat(old_probs)

    advantages = torch.cat(advantages)
    check_advantages(advantages, correct_len=len(old_probs))


    optimizer.zero_grad()
    new_probs = policyNN(states).unsqueeze(1)
    epsilon_2 = 1e-10
    ratio = new_probs / (old_probs + epsilon_2)
    clip_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clip_adv = _mult_ratio_and_reward(clip_ratio, advantages)
    ratio_times_adv = _mult_ratio_and_reward(ratio, advantages)
    loss = -torch.min(ratio_times_adv, clip_adv).mean()
    loss.backward()
    check_nn_gradients(policyNN)
    optimizer.step()


def update_value_function_mse(optimizer, valueNN, trajectories):
    optimizer.zero_grad()
    states = torch.cat([t['board_states'] for t in trajectories])
    rewards_to_go = [t['rewards_to_go'] for t in trajectories]
    rewards_to_go = torch.cat(rewards_to_go)
    values = valueNN(states).squeeze()
    loss = F.mse_loss(values, rewards_to_go)
    loss.backward()
    optimizer.step()


def setup_optimizers(policyNN, valueNN):
    policy_optimizer = optim.Adam(policyNN.parameters(),
                                  lr=learning_rate_policy)
    value_optimizer = optim.Adam(valueNN.parameters(),
                                 lr=learning_rate_value)
    return policy_optimizer, value_optimizer


def _mult_ratio_and_reward(ratios, advantages):
    reshaped_adv = advantages.unsqueeze(1).unsqueeze(2)
    product = ratios * reshaped_adv
    return product


if __name__ == '__main__':
    from nanoAlphaGo.game.board import GoBoard
    from nanoAlphaGo.config import WHITE
    board = GoBoard()
    ppo_train(PolicyNN(WHITE), ValueNN(board))


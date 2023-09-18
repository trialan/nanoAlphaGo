import torch
import torch.nn.functional as F
import torch.optim as optim

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import WHITE
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.rl.trajectories import collect_trajectories
from nanoAlphaGo.rl.value import value_function, ValueNN


n_trajectories = 3
learning_rate_policy = 0.001
learning_rate_value = 0.001
epsilon = 0.2
policy_epochs = 4
value_epochs = 4
gamma=0.99
lambda_=0.95


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
                                  trajectories,
                                  rewards_to_go)


def compute_rewards_to_go(trajectories):
    rewards_to_go_list = []
    for trajectory in trajectories:
        trajectory = _compute_rtg_single_trajectory(trajectory)
    return trajectories


def compute_advantages(trajectories, valueNN):
    """ GAE + set final advantage to game outcome (questionable).  """
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
    num_advantages = sum(len(a) for a in advantages_list)
    assert num_advantages == num_states

    return advantages_list



def update_policy_ppoclip(optimizer, policyNN, trajectories, advantages):
    assert len(trajectories) == len(advantages)
    for i in range(policy_epochs):
        optimizer.zero_grad()
        states, actions, old_probs = zip(*[(t['board_states'], t['moves'], t['move_probs']) for t in trajectories])
        states = torch.cat(states)
        actions = torch.cat(actions)
        old_probs = torch.cat(old_probs)
        advantages = torch.cat(advantages)
        print(f"Happy: {i}")
        assert len(advantages) == len(old_probs)
        new_probs = policyNN(states).unsqueeze(1)
        ratio = new_probs / old_probs
        clip_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clip_adv = _mult_ratio_and_reward(clip_ratio, advantages)
        ratio_times_adv = _mult_ratio_and_reward(ratio, advantages)
        loss = -torch.min(ratio_times_adv, clip_adv).mean()
        loss.backward()
        optimizer.step()


def update_value_function_mse(optimizer, valueNN, trajectories, rewards_to_go):
    for _ in range(value_epochs):
        optimizer.zero_grad()
        states = torch.cat([t['states'] for t in trajectories])
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


def _compute_rtg_single_trajectory(trajectory):
    outcome = trajectory['rewards'][-1]
    n_zeros = len(trajectory['board_states']) - 1
    rewards_to_go = [0 for _ in range(n_zeros)] + [outcome]
    trajectory['rewards_to_go'] = rewards_to_go
    return trajectory


def _mult_ratio_and_reward(ratios, advantages):
    reshaped_adv = advantages.unsqueeze(1).unsqueeze(2)
    product = ratios * reshaped_adv
    return product


if __name__ == '__main__':
    board = GoBoard()
    ppo_train(PolicyNN(WHITE), ValueNN(board))


import torch
import torch.nn.functional as F
import torch.optim as optim

from nanoAlphaGo.rl.trajectories import collect_trajectories

n_trajectories = 10
learning_rate_policy = 0.001
learning_rate_value = 0.001
epsilon = 0.2
policy_epochs = 4
value_epochs = 4

policy_optimizer = optim.Adam(policyNN.parameters(),
                              lr=learning_rate_policy)
value_optimizer = optim.Adam(valueNN.parameters(),
                             lr=learning_rate_value)


def ppo_train(policyNN, valueNN):
    for k in range(n_trajectories):
        trajectories = collect_trajectories(policyNN, n_trajectories)
        rewards_to_go = compute_reward_to_go(trajectories)
        advantages = compute_advantages(rewards_to_go,
                                        valueNN,
                                        trajectories)
        update_policy_ppoclip(policy_optimizer,
                              policyNN,
                              trajectories,
                              advantages)
        update_value_function_mse(value_optimizer,
                                  valueNN,
                                  trajectories,
                                  rewards_to_go)



def compute_reward_to_go(trajectories):
    pass


def compute_advantages(rewards_to_go, valueNN, trajectories):
    pass


def update_policy_ppoclip(optimizer, policyNN, trajectories, advantages, epsilon):
    for _ in range(policy_epochs):
        optimizer.zero_grad()
        states, actions, old_probs = zip(*[(t['states'], t['actions'], t['action_probs']) for t in trajectories])
        states = torch.cat(states)
        actions = torch.cat(actions)
        old_probs = torch.cat(old_probs)
        advantages = torch.cat(advantages)
        new_probs = policyNN(states).gather(1, actions)
        ratio = new_probs / old_probs
        clip_adv = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
        loss = -torch.min(ratio * advantages, clip_adv).mean()
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


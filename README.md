# Teaching an agent the game of Go with Proximal Policy Optimization (PPO)
Inspired by AlphaGo Zero and nanoGPT, this is a minimalistic implementation for
teaching a neural net to play the game of Go purely through self-play. No
training on human games is involved, as this is deemed more elegant.

The main features of the code base are:
1. An implementation of PPO
2. Some Go logic
3. (WIP) Monte-Carlo tree search

# Performance of the agent
We evaluate how our policy fares against a random policy (later, we can replace
this policy with our current best policy network and iteratively improve) by
having them play 1000 games after every N training steps.

```python
board = GoBoard()
policy_network = PolicyNN(WHITE)
value_network = ValueNN(board)
for _ in range(EPOCHS):
    performance = performance_against_random_policy(policy_network,
                                                    n_games=1000)
    ppo_train(policy_network, value_network, n_loops=100)
```

In the plots below (WIP) we see how performance evolves with training.

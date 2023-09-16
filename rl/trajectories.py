""" A trajectory can be defined as the sequence:

    Tau = (s_0, a_0, s_1, a_1, ... , s_T, a_T)

    Where s_t is the state at time t, and a_t the action taken by
    the policy in this state. Because the reward only occurs at
    the end of the game we will do trajectories until the game is over,
    which means both players pass (draw), one resigns (un-implemented
    as of 14/09/23) or there are no more legal moves.

    Our RL agent will always play white for now (14/09/23), this gives it
    a slight disadvantage as black plays first (this is typically adjusted
    by imposing a komi. """

import numpy as np
import torch

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import BLACK, WHITE, PASS
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.graphics.rendering import display_board
from nanoAlphaGo.game.scoring import calculate_outcome_for_player


def collect_trajectories(policyNN, n_trajectories):
    trajectories = [play_game(policyNN) for _ in range(n_trajectories)]
    return trajectories


def play_game(policy):
    board = GoBoard()
    adversary = PolicyNN(BLACK)
    players = {WHITE: policy,
               BLACK: adversary}

    moves = []
    rewards = []
    board_states = []
    policy_probs = []
    consecutive_passes = 0

    player = BLACK
    while not game_is_over(board, consecutive_passes, player):
        network = players[player]
        board_state = board.tensor.clone()
        batch = board_state.unsqueeze(0)
        probs = network.forward(batch)
        move = network.get_move_as_int_from_prob_dist(probs, board_state)

        if move == PASS:
            consecutive_passes += 1
        else:
            consecutive_passes = 0
            board.apply_move(move, player)

        moves.append(move)
        rewards.append(0)
        board_states.append(board_state)
        policy_probs.append(probs)
        player = -player

    rewards[-1] = calculate_outcome_for_player(board, policy.color)
    trajectory = {"rewards": torch.tensor(rewards),
                  "moves": torch.tensor(moves),
                  "board_states": torch.stack(board_states),
                  "move_probs": torch.stack(policy_probs)}
    return trajectory


def game_is_over(board, consecutive_passes, turn):
    players_both_passed = consecutive_passes > 1
    no_legal_moves = len(board.legal_moves(turn)) == 0
    return players_both_passed or no_legal_moves


if __name__ == '__main__':
    policy = PolicyNN(color=WHITE)
    t = play_game(policy)



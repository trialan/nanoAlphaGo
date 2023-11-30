""" A trajectory can be defined as the sequence:

    Tau = (s_0, a_0, s_1, a_1, ... , s_T, a_T)

    Where s_t is the state at time t, and a_t the action taken by
    the policy in this state. Because the reward only occurs at
    the end of the game we will do trajectories until the game is over,
    which means both players pass (draw), one resigns (un-implemented
    as of 14/09/23) or there are no more legal moves.

    Our RL agent will always play white for now (14/09/23), this gives it
    a slight disadvantage as black plays first (this is typically adjusted
    by imposing a komi). """


import numpy as np
import torch
from tqdm import tqdm

from nanoAlphaGo.config import BLACK, WHITE, PASS
from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.game.scoring import calculate_outcome_for_player
from nanoAlphaGo.graphics.rendering import display_board
from nanoAlphaGo.rl.policy import PolicyNN


def collect_trajectories(policyNN, n_trajectories):
    trajectories = [play_game(policyNN) for _ in
                    tqdm(range(n_trajectories),
                         desc=f"Collecting {n_trajectories} trajectories")]
    return trajectories


def play_game(policy):
    board = GoBoard()
    adversary = PolicyNN(BLACK)
    game_data = initialise_game_data()

    while not game_is_over(board, game_data):
        play_turn(board, policy, adversary, game_data)
        switch_player(game_data)

    game_outcome = calculate_outcome_for_player(board, policy.color)
    trajectory = build_trajectory(game_data, game_outcome, policy.device)
    return trajectory


def initialise_game_data():
    game_data = {
        'moves': [],
        'rewards': [],
        'board_states': [],
        'policy_probs': [],
        'consecutive_passes': 0,
        'player': BLACK,
    }
    return game_data


def game_is_over(board, game_data):
    """ Two consecutive passes, or only legal move is PASS  """
    players_both_passed = game_data["consecutive_passes"] > 1
    no_legal_moves = board.legal_moves(game_data["player"]) == [81]
    return players_both_passed or no_legal_moves


def play_turn(board, policy, adversary, game_data):
    player = game_data["player"]
    network = get_player_network(policy, adversary, player)
    move, board_state, probs = compute_move(network, board)
    update_consecutive_passes(move, game_data)
    if move != PASS:
        board.apply_move(move, player)
    game_data = update_game_data(game_data, move, board_state, probs)
    return move, board_state, probs


def switch_player(game_data):
    game_data["player"] = -game_data["player"]


def build_trajectory(game_data, game_outcome, device):
    game_data["rewards"][-1] = game_outcome
    trajectory = {"rewards": torch.tensor(game_data["rewards"]).to(device),
                  "moves": torch.tensor(game_data["moves"]).to(device),
                  "board_states": torch.stack(game_data["board_states"]).to(device),
                  "move_probs": torch.stack(game_data["policy_probs"]).to(device)}
    return trajectory


def update_game_data(game_data, move, board_state, probs):
    game_data["moves"].append(move)
    game_data["rewards"].append(0)
    game_data["board_states"].append(board_state)
    game_data["policy_probs"].append(probs)
    return game_data


def get_player_network(policy, adversary, player):
    players = {WHITE: policy, BLACK: adversary}
    return players[player]


def compute_move(network, board):
    board_state = board.tensor.clone()
    batch = board_state.unsqueeze(0)
    probs = network.forward(batch)
    move = network.get_move_as_int_from_prob_dist(probs, board_state)
    return move, board_state, probs


def update_consecutive_passes(move, game_data):
    if move == PASS:
        game_data["consecutive_passes"] += 1
    else:
        game_data["consecutive_passes"] = 0


